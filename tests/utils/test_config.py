"""Tests for config loading and hashing."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from drrl.utils.config import config_hash, load_config, to_container


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_load_config_reads_mapping(tmp_path: Path):
    p = _write(tmp_path / "c.yaml", "a: 1\nb:\n  c: 2\n")
    cfg = load_config(p)
    assert cfg.a == 1
    assert cfg.b.c == 2


def test_load_config_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_rejects_list(tmp_path: Path):
    p = _write(tmp_path / "list.yaml", "- 1\n- 2\n")
    with pytest.raises(TypeError):
        load_config(p)


def test_to_container_resolves_interpolation():
    cfg = OmegaConf.create({"name": "x", "dir": "out/${name}"})
    container = to_container(cfg)
    assert container["dir"] == "out/x"


def test_config_hash_is_order_independent():
    a = OmegaConf.create({"x": 1, "y": 2})
    b = OmegaConf.create({"y": 2, "x": 1})
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_content():
    a = OmegaConf.create({"x": 1})
    b = OmegaConf.create({"x": 2})
    assert config_hash(a) != config_hash(b)


def test_config_hash_length():
    cfg = OmegaConf.create({"x": 1})
    assert len(config_hash(cfg, length=8)) == 8
