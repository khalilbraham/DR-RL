"""Cover the wandb tracker adapter by injecting a fake wandb module."""

import sys
import types

from drrl.utils.tracking import get_tracker


def _fake_wandb() -> tuple[types.ModuleType, dict[str, object]]:
    logs: list[tuple[int | None, dict[str, float]]] = []
    state: dict[str, object] = {"logs": logs, "finished": False}

    class _Run:
        def log(self, metrics: dict[str, float], step: int | None = None) -> None:
            logs.append((step, metrics))

        def finish(self) -> None:
            state["finished"] = True

    mod = types.ModuleType("wandb")
    mod.init = lambda project=None, config=None: _Run()  # type: ignore[attr-defined]
    return mod, state


def test_wandb_tracker_logs_and_finishes(monkeypatch):
    fake, state = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    tracker = get_tracker("wandb", project="drrl", config={"lr": 1.0})
    tracker.log({"loss": 0.5}, step=3)
    tracker.finish()
    assert state["logs"] == [(3, {"loss": 0.5})]
    assert state["finished"] is True
