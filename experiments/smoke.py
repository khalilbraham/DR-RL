"""Sanity entrypoint: compose config, seed, and write a run manifest.

This exercises the Phase-0 plumbing end to end (Hydra compose -> seed ->
manifest) so ``make smoke`` proves the foundations are wired correctly. It
runs no science.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from drrl.utils.config import config_hash
from drrl.utils.logging import configure_logging, get_logger
from drrl.utils.manifest import build_manifest
from drrl.utils.seed import set_seed

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs"


def main() -> Path:
    """Compose the root config, seed RNGs, and write a manifest.

    Returns:
        Path to the written manifest JSON.
    """
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="config")

    configure_logging(level=cfg.run.log_level)
    log = get_logger(__name__)

    seed_state = set_seed(int(cfg.seed))
    cfg_hash = config_hash(cfg)
    log.info("seeded backends=%s config_hash=%s", seed_state.seeded, cfg_hash)

    manifest = build_manifest(
        seed=int(cfg.seed), config_hash=cfg_hash, repo_root=_REPO_ROOT
    )
    out_dir = _REPO_ROOT / str(cfg.run.output_dir)
    path = manifest.write(out_dir / "manifest.json")
    log.info("wrote manifest -> %s", path)
    return path


if __name__ == "__main__":
    main()
