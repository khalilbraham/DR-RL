"""Run manifests: a tamper-evident record of *how* a result was produced.

Every entrypoint writes a manifest capturing the git SHA, config hash, root
seed, key library versions, and hardware. A skeptical reviewer can read a
manifest and know the exact provenance of any number in the repo.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Libraries whose versions materially affect numerical results.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pydantic",
    "pint",
    "sympy",
    "omegaconf",
    "hydra-core",
    "jax",
    "jaxlib",
    "diffrax",
    "torch",
    "transformers",
    "trl",
)


def _git_sha(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit SHA, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def _git_dirty(cwd: str | Path | None = None) -> bool | None:
    """Return whether the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(out.stdout.strip())


def _library_versions() -> dict[str, str]:
    """Map tracked package name -> installed version (absent ones omitted)."""
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


@dataclass(frozen=True)
class RunManifest:
    """Provenance record for a single run.

    Attributes:
        seed: Root seed used.
        config_hash: Short hash of the resolved config (see
            :func:`drrl.utils.config.config_hash`).
        git_sha: Commit SHA, or ``None`` outside a git checkout.
        git_dirty: Whether the tree had uncommitted changes at run time.
        timestamp_utc: ISO-8601 UTC creation time.
        platform: Human-readable OS/CPU string.
        python_version: Interpreter version.
        libraries: Tracked library versions.
        extra: Free-form additional metadata (e.g. hardware, run name).
    """

    seed: int
    config_hash: str
    git_sha: str | None
    git_dirty: bool | None
    timestamp_utc: str
    platform: str
    python_version: str
    libraries: dict[str, str] = field(default_factory=dict)
    extra: dict[str, str] = field(default_factory=dict)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the manifest to a JSON string."""
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        """Write the manifest JSON to ``path`` (creating parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p


def build_manifest(
    *,
    seed: int,
    config_hash: str,
    repo_root: str | Path | None = None,
    extra: dict[str, str] | None = None,
) -> RunManifest:
    """Assemble a :class:`RunManifest` from the live environment.

    Args:
        seed: Root seed used for the run.
        config_hash: Short config hash from :func:`drrl.utils.config.config_hash`.
        repo_root: Directory to query git from; defaults to the process CWD.
        extra: Optional extra metadata to embed (e.g. ``{"gpu": "A100"}``).

    Returns:
        A frozen, serializable manifest.
    """
    return RunManifest(
        seed=seed,
        config_hash=config_hash,
        git_sha=_git_sha(repo_root),
        git_dirty=_git_dirty(repo_root),
        timestamp_utc=datetime.now(UTC).isoformat(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        libraries=_library_versions(),
        extra=dict(extra or {}),
    )
