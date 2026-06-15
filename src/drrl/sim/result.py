"""Simulation result value object (NumPy at the layer boundary)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationResult:
    """Output of a forward simulation.

    Attributes:
        times: Sample times, shape ``(T,)``.
        states: Compartment states at ``times``, shape ``(T, n_states)``.
        observed: Observed (transformed) prediction, shape ``(T,)``.
        sensitivities: ``d observed / d theta``, shape ``(T, n_params)``, or
            ``None`` if not requested.
        integrator_ok: Whether every integration segment succeeded.
        diagnostics: Backend-specific diagnostics (solver, steps, etc.).
    """

    times: FloatArray
    states: FloatArray
    observed: FloatArray
    sensitivities: FloatArray | None
    integrator_ok: bool
    diagnostics: dict[str, object] = field(default_factory=dict)
