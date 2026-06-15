"""Backend Protocol and shared model->numeric compilation.

The vector-field/observation compilation from a :class:`~drrl.spec.model.ModelSpec`
is backend-agnostic and lives here; concrete backends (diffrax, scipy) consume
it. Keeping the compilation in one place guarantees both backends integrate the
*same* system, which is what makes the cross-check meaningful.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from drrl.sim.result import SimulationResult
from drrl.spec.model import Design, ModelSpec


@dataclass(frozen=True)
class SimConfig:
    """Integration controls (populated from ``configs/sim``).

    Attributes:
        solver: Solver name (e.g. ``"Kvaerno5"``, ``"Dopri5"``).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        max_steps: Max integrator steps per segment.
    """

    solver: str = "Kvaerno5"
    rtol: float = 1e-8
    atol: float = 1e-10
    max_steps: int = 100_000


class Backend(Protocol):
    """A forward simulator for a :class:`ModelSpec` under a :class:`Design`."""

    def simulate(
        self, spec: ModelSpec, design: Design, *, with_sensitivities: bool = False
    ) -> SimulationResult:
        """Integrate ``spec`` under ``design`` and return predictions.

        Args:
            spec: The model.
            design: Dosing + sampling design.
            with_sensitivities: If ``True``, also return ``d observed/d theta``.

        Returns:
            A :class:`SimulationResult`.
        """
        ...


@dataclass(frozen=True)
class CompiledModel:
    """Numeric handles compiled from a spec, shared across backends.

    Attributes:
        state_names: Compartment names (state order).
        param_names: Parameter names (theta order).
        theta_natural: Natural-space parameter values, shape ``(n_params,)``.
        n_states: Number of states.
        obs_state_index: Index of the observed compartment.
        divide_by_index: Index of the volume parameter to divide by, or ``None``.
        log_transform: Whether the observation applies ``log``.
        scheduled_doses: ``(time, state_index, amount)`` boluses, time-sorted.
    """

    state_names: tuple[str, ...]
    param_names: tuple[str, ...]
    theta_natural: np.ndarray
    n_states: int
    obs_state_index: int
    divide_by_index: int | None
    log_transform: bool
    scheduled_doses: tuple[tuple[float, int, float], ...]


def compile_model(spec: ModelSpec, design: Design) -> CompiledModel:
    """Compile static numeric metadata from a spec + design.

    Raises:
        NotImplementedError: For dose routes not yet supported (Phase 1 supports
            ``iv_bolus`` only).
    """
    state_names = spec.state_names
    param_names = spec.param_names
    state_index = {n: i for i, n in enumerate(state_names)}
    param_index = {n: i for i, n in enumerate(param_names)}

    theta = np.array([p.natural_value for p in spec.parameters], dtype=np.float64)

    obs = spec.observation
    divide_idx = param_index[obs.divide_by] if obs.divide_by is not None else None

    doses: list[tuple[float, int, float]] = []
    for d in design.doses:
        if d.route != "iv_bolus":
            raise NotImplementedError(
                f"dose route {d.route!r} is not supported in Phase 1 (iv_bolus only)"
            )
        doses.append((float(d.time), state_index[d.compartment], float(d.amount)))
    doses.sort(key=lambda x: x[0])

    return CompiledModel(
        state_names=state_names,
        param_names=param_names,
        theta_natural=theta,
        n_states=len(state_names),
        obs_state_index=state_index[obs.state],
        divide_by_index=divide_idx,
        log_transform=obs.transform == "log",
        scheduled_doses=tuple(doses),
    )


def dose_schedule(
    compiled: CompiledModel,
) -> tuple[Sequence[tuple[int, float]], Sequence[tuple[float, int, float]]]:
    """Split doses into t==0 boluses and future (t>0) boluses.

    Returns:
        ``(initial, future)`` where ``initial`` is ``(state_index, amount)`` for
        boluses at ``t == 0`` and ``future`` is the time-sorted ``(t, idx, amt)``
        list of boluses at ``t > 0``.
    """
    initial = [(idx, amt) for (t, idx, amt) in compiled.scheduled_doses if t == 0.0]
    future = [(t, idx, amt) for (t, idx, amt) in compiled.scheduled_doses if t > 0.0]
    return initial, future


def apply_observation(states: np.ndarray, compiled: CompiledModel) -> np.ndarray:
    """Map state trajectories to the observed (transformed) quantity.

    Args:
        states: Array shape ``(T, n_states)``.
        compiled: Compiled model metadata.

    Returns:
        Observed prediction, shape ``(T,)``.
    """
    y = states[:, compiled.obs_state_index]
    if compiled.divide_by_index is not None:
        y = y / compiled.theta_natural[compiled.divide_by_index]
    if compiled.log_transform:
        y = np.log(y)
    return y


# A pure-NumPy vector field f(y, theta) -> dy, used by the scipy backend.
NumpyVectorField = Callable[[np.ndarray, np.ndarray], np.ndarray]
