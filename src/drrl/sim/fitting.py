"""Least-squares best-fit of a model's parameters to observed data.

Used so the reward scores each *committed structure at its best fit* rather than
at fixed default parameters — otherwise "does structure X fit" conflates the
structure with arbitrary defaults. Fitting is done in log-parameter space (PK
parameters are positive) on the observed design; generalization is then scored
on a held-out battery by the caller.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from drrl.sim.backend import Backend
from drrl.spec.model import Design, ModelSpec

FloatArray = NDArray[np.float64]


def set_natural_params(spec: ModelSpec, values: FloatArray) -> ModelSpec:
    """Return a copy of ``spec`` with parameters set to natural-space ``values``."""
    new_params = tuple(
        p.model_copy(update={"value": float(values[j]), "coord": "linear"})
        for j, p in enumerate(spec.parameters)
    )
    return spec.model_copy(update={"parameters": new_params})


def fit_params(
    template: ModelSpec,
    target_observed: FloatArray,
    design: Design,
    backend: Backend,
    *,
    max_nfev: int = 200,
) -> tuple[ModelSpec, float]:
    """Best-fit ``template``'s parameters to ``target_observed`` on ``design``.

    Args:
        template: Model whose structure is fixed; its parameters are the start.
        target_observed: Target observations on ``design`` (e.g. reference data).
        design: The design the target was measured on.
        backend: Simulator.
        max_nfev: Max least-squares function evaluations.

    Returns:
        ``(fitted_spec, cost)`` — the best-fit model and the final residual cost.
        Falls back to the template if the fit fails to integrate.
    """
    p0 = np.array([p.natural_value for p in template.parameters], dtype=np.float64)
    log_p0 = np.log(np.clip(p0, 1e-12, None))
    scale = np.abs(target_observed) + 1e-9  # ~proportional weighting

    def residual(log_p: FloatArray) -> FloatArray:
        cand = set_natural_params(template, np.exp(log_p))
        try:
            pred = backend.simulate(cand, design).observed
        except Exception:
            return np.full_like(target_observed, 1e3)
        return (pred - target_observed) / scale

    try:
        sol = least_squares(residual, log_p0, method="lm", max_nfev=max_nfev)
        fitted = set_natural_params(template, np.exp(sol.x))
        return fitted, float(sol.cost)
    except Exception:
        return template, float("inf")
