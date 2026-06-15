"""Thin experiment-tracking adapter with a local no-op fallback.

The codebase never hard-depends on Weights & Biases (or any tracker). Call
:func:`get_tracker`; if the optional ``track`` extra is not installed, you get
a :class:`NullTracker` that records to memory, so tests and offline runs work
unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    """Minimal tracking interface consumed by the RL/bench layers."""

    def log(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log a dict of scalar metrics at an optional step."""
        ...

    def finish(self) -> None:
        """Flush and close the tracker."""
        ...


class NullTracker:
    """In-memory no-op tracker used when no backend is configured.

    Keeps the last logged metrics so tests can assert on logging behavior
    without a network dependency.
    """

    def __init__(self) -> None:
        """Initialize with an empty history."""
        self.history: list[tuple[int | None, dict[str, float]]] = []

    def log(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Append metrics to the in-memory history."""
        self.history.append((step, dict(metrics)))

    def finish(self) -> None:
        """No-op flush."""


def get_tracker(
    backend: str = "null",
    *,
    project: str | None = None,
    config: dict[str, Any] | None = None,
) -> Tracker:
    """Return a tracker for ``backend``.

    Args:
        backend: ``"null"`` (default) or ``"wandb"``.
        project: Project name for remote backends.
        config: Run config to record with remote backends.

    Returns:
        A :class:`Tracker`. Falls back to :class:`NullTracker` when the
        requested backend is unavailable.
    """
    if backend == "wandb":
        try:
            import wandb
        except ImportError:
            return NullTracker()
        run = wandb.init(project=project, config=config or {})
        return _WandbTracker(run)
    return NullTracker()


class _WandbTracker:
    """Adapter wrapping a live ``wandb`` run behind :class:`Tracker`."""

    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        self._run.log(metrics, step=step)

    def finish(self) -> None:
        self._run.finish()
