"""End-to-end Phase-0 plumbing test: the smoke entrypoint runs and writes a manifest."""

import json

from experiments.smoke import main


def test_smoke_writes_valid_manifest():
    path = main()
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["seed"] == 0
    assert "config_hash" in data
    assert data["python_version"]
