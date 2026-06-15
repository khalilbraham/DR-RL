"""Shared fixtures: tolerances come from config, never hardcoded in tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from drrl.utils.config import load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tol() -> DictConfig:
    """The project tolerance config (``configs/tolerances/default.yaml``)."""
    return load_config(_REPO_ROOT / "configs" / "tolerances" / "default.yaml")
