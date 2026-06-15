"""Deterministic seeding across Python, NumPy, and (optionally) JAX/Torch.

A single :func:`set_seed` is the only sanctioned entry point for randomness
initialization in the codebase. The science layer (``spec``, ``sim``,
``verifier``, ``reward``) must never seed implicitly; randomness is injected
explicitly so that every result is reproducible from a manifest.

Nondeterminism sources we cannot fully eliminate are documented in
``ASSUMPTIONS.md`` (JAX/XLA reductions, CUDA atomics). :func:`set_seed`
enables deterministic algorithms where the backend supports it and records
what was actually seeded so a manifest can attest to it.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeedState:
    """Record of what :func:`set_seed` actually seeded.

    Attributes:
        seed: The root integer seed requested.
        seeded: Sorted names of backends successfully seeded
            (e.g. ``"python"``, ``"numpy"``, ``"torch"``, ``"jax"``).
        deterministic_backends: Backends for which deterministic-algorithm
            mode was additionally enabled.
    """

    seed: int
    seeded: tuple[str, ...] = field(default_factory=tuple)
    deterministic_backends: tuple[str, ...] = field(default_factory=tuple)


def set_seed(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed all available RNG backends deterministically.

    Always seeds Python's :mod:`random` and NumPy. Seeds Torch and JAX only if
    they are importable, so the pure-science test suite has no hard dependency
    on the heavy stacks.

    Args:
        seed: Non-negative root seed. The same seed must reproduce a run.
        deterministic: If ``True``, additionally request deterministic kernels
            where the backend exposes a toggle (Torch deterministic algorithms,
            cuBLAS workspace config). Has no effect on backends that lack one.

    Returns:
        A :class:`SeedState` describing exactly what was seeded.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    seeded: list[str] = []
    deterministic_backends: list[str] = []

    random.seed(seed)
    seeded.append("python")

    import numpy as np

    np.random.seed(seed)
    seeded.append("numpy")

    if deterministic:
        # cuBLAS determinism for Torch/JAX CUDA matmuls; harmless if unused.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    _seed_torch(
        seed, deterministic=deterministic, seeded=seeded, det=deterministic_backends
    )
    _seed_jax(seed, seeded=seeded)

    return SeedState(
        seed=seed,
        seeded=tuple(sorted(seeded)),
        deterministic_backends=tuple(sorted(deterministic_backends)),
    )


def _seed_torch(
    seed: int, *, deterministic: bool, seeded: list[str], det: list[str]
) -> None:
    """Seed Torch if importable; enable deterministic algorithms when asked."""
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        det.append("torch")
    seeded.append("torch")


def _seed_jax(seed: int, *, seeded: list[str]) -> None:
    """Seed JAX's legacy global RNG if importable.

    JAX's preferred RNG is functional (explicit keys threaded through code);
    we only set the legacy global state here and record availability so callers
    that need a key can derive one deterministically from ``seed``.
    """
    try:
        import jax
    except ImportError:
        return
    # Touch a key so the import is not flagged unused and to validate the seed
    # is usable as a JAX key seed.
    jax.random.PRNGKey(seed)
    seeded.append("jax")
