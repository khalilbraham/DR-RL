"""Config loading and content-hashing built on OmegaConf.

All tolerances, weights, thresholds, and seeds live in ``configs/`` and reach
the code only through here. The config hash is part of every run manifest, so
two runs with identical configs are provably comparable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file into an OmegaConf ``DictConfig``.

    Args:
        path: Path to a ``.yaml`` config file.

    Returns:
        The parsed config.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the file parses to a list rather than a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    cfg = OmegaConf.load(p)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"expected a mapping config at {p}, got {type(cfg).__name__}")
    return cfg


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Resolve interpolations and convert a config to a plain ``dict``.

    Args:
        cfg: The config to materialize.

    Returns:
        A fully-resolved, JSON-serializable mapping.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):  # pragma: no cover - defensive; type-forbidden
        raise TypeError("resolved config is not a mapping")
    # Keys come back as Any from OmegaConf; normalize to str for a stable hash.
    return {str(k): v for k, v in container.items()}


def config_hash(cfg: DictConfig, *, length: int = 12) -> str:
    """Return a stable short hash of a resolved config.

    The hash is order-independent (keys are sorted) so semantically identical
    configs hash identically regardless of authoring order.

    Args:
        cfg: The config to hash.
        length: Number of leading hex characters to return.

    Returns:
        A hex digest prefix of the requested length.
    """
    payload = json.dumps(to_container(cfg), sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]
