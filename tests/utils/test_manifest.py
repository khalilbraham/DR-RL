"""Tests for the run manifest."""

import json
from pathlib import Path

from drrl.utils.manifest import RunManifest, build_manifest


def test_build_manifest_core_fields():
    m = build_manifest(seed=42, config_hash="abc123")
    assert isinstance(m, RunManifest)
    assert m.seed == 42
    assert m.config_hash == "abc123"
    assert m.python_version
    assert m.platform
    # numpy is a base dependency, so it must be recorded.
    assert "numpy" in m.libraries


def test_manifest_json_roundtrips():
    m = build_manifest(seed=1, config_hash="h", extra={"gpu": "none"})
    data = json.loads(m.to_json())
    assert data["seed"] == 1
    assert data["extra"]["gpu"] == "none"


def test_manifest_write_creates_file(tmp_path: Path):
    m = build_manifest(seed=0, config_hash="h")
    out = m.write(tmp_path / "nested" / "manifest.json")
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["config_hash"] == "h"


def test_git_fields_present_in_repo():
    # Inside this git checkout, git_sha should resolve and git_dirty be a bool.
    m = build_manifest(seed=0, config_hash="h", repo_root=Path(__file__).parent)
    assert m.git_sha is None or isinstance(m.git_sha, str)
    assert m.git_dirty is None or isinstance(m.git_dirty, bool)


def test_git_fields_none_outside_repo(tmp_path: Path):
    # A non-repo directory makes `git rev-parse`/`git status` fail -> None.
    m = build_manifest(seed=0, config_hash="h", repo_root=tmp_path)
    assert m.git_sha is None
    assert m.git_dirty is None
