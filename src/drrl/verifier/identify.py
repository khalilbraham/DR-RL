"""CORE: parameterization-invariant, prediction-based identifiability.

A parameter direction is *identifiable* if moving along it changes predictions
on a **held-out design battery**. We measure this through the noise-normalized
prediction-sensitivity matrix ``S = (d observed / d theta) / sigma`` stacked over
the battery:

- The number of prediction-affecting independent directions is ``rank(S)``.
  Rank is invariant under any smooth invertible reparameterization
  (``rank(S·M) = rank(S)`` for invertible ``M``), so the identifiability score
  ``rank(S) / n_params`` is parameterization-invariant by construction. This is
  the test that proves the metric measures the science, not the coordinates.
- The null space of ``S`` gives the *flat / sloppy* directions — combinations of
  parameters that do not affect predictions.

We deliberately do **not** use ``kappa(S)`` (the FIM condition number): it changes
under coordinate scaling and so fails reparameterization invariance.

The rank verdict is a cheap surrogate for the inner loop;
:func:`prediction_change_along` provides the prediction-grounded spot-check
(move along a direction, measure the RMS standardized prediction change) used
to calibrate it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from drrl.sim.backend import Backend
from drrl.spec.model import Design, ErrorModel, ModelSpec
from drrl.verifier.report import IdentifyReport

FloatArray = NDArray[np.float64]


def observation_sigma(
    observed: FloatArray, error: ErrorModel, *, floor: float
) -> FloatArray:
    """Per-observation noise SD from the error model (with a floor).

    Args:
        observed: Predicted observations.
        error: The residual error model.
        floor: Minimum SD, used where the model gives zero (degenerate noise).

    Returns:
        Standard deviations, same shape as ``observed``.
    """
    add = error.sigma_add
    prop = error.sigma_prop
    sigma = np.sqrt(add**2 + (prop * np.abs(observed)) ** 2)
    return np.where(sigma <= 0.0, floor, sigma)


def sensitivity_matrix(
    spec: ModelSpec,
    battery: list[Design],
    backend: Backend,
    *,
    sigma_floor: float = 1e-9,
) -> FloatArray:
    """Noise-normalized prediction-sensitivity matrix stacked over a battery.

    Args:
        spec: The model.
        battery: Held-out designs to evaluate predictions on.
        backend: Simulator (must provide sensitivities).
        sigma_floor: Noise floor for degenerate error models.

    Returns:
        Array ``S`` of shape ``(sum_designs T_d, n_params)``.

    Raises:
        ValueError: If ``battery`` is empty or a backend returns no sensitivities.
    """
    if not battery:
        raise ValueError("identifiability needs a non-empty design battery")
    blocks: list[FloatArray] = []
    for design in battery:
        result = backend.simulate(spec, design, with_sensitivities=True)
        if result.sensitivities is None:
            raise ValueError("backend did not return sensitivities")
        sigma = observation_sigma(
            result.observed, spec.observation.error, floor=sigma_floor
        )
        blocks.append(np.asarray(result.sensitivities) / sigma[:, None])
    return np.vstack(blocks)


def _normalize_sign(vec: FloatArray) -> tuple[float, ...]:
    """Fix the sign of a unit vector (largest entry positive) for stability."""
    idx = int(np.argmax(np.abs(vec)))
    sign = 1.0 if vec[idx] >= 0 else -1.0
    return tuple(float(x) for x in (vec * sign))


def identifiability(
    spec: ModelSpec,
    battery: list[Design],
    backend: Backend,
    *,
    rank_rtol: float = 1e-6,
    sigma_floor: float = 1e-9,
) -> IdentifyReport:
    """Compute the parameterization-invariant identifiability report.

    Args:
        spec: The model.
        battery: Held-out designs (must differ from the training design).
        backend: Simulator with sensitivities.
        rank_rtol: Singular-value cutoff relative to the largest, for rank.
        sigma_floor: Noise floor for degenerate error models.

    Returns:
        An :class:`IdentifyReport`.
    """
    names = spec.param_names
    n_params = len(names)
    s_matrix = sensitivity_matrix(spec, battery, backend, sigma_floor=sigma_floor)

    _, sv, vt = np.linalg.svd(s_matrix, full_matrices=True)
    smax = float(sv[0]) if sv.size else 0.0
    tol = rank_rtol * smax
    rank = int((sv > tol).sum()) if smax > 0 else 0
    fraction = rank / n_params if n_params else 0.0

    col_norms = np.linalg.norm(s_matrix, axis=0)
    prediction_affecting = {
        name: bool(col_norms[j] > tol) for j, name in enumerate(names)
    }

    null_rows = vt[rank:] if rank < n_params else np.empty((0, n_params))
    null_dirs = tuple(_normalize_sign(np.asarray(row)) for row in null_rows)

    return IdentifyReport(
        param_names=names,
        n_params=n_params,
        identifiable_rank=rank,
        identifiable_fraction=fraction,
        score=fraction,
        prediction_affecting=prediction_affecting,
        nonidentifiable_directions=null_dirs,
    )


def prediction_change_along(
    spec: ModelSpec,
    direction: FloatArray,
    battery: list[Design],
    backend: Backend,
    *,
    step: float = 1e-3,
    sigma_floor: float = 1e-9,
) -> float:
    """RMS standardized prediction change when moving theta along ``direction``.

    This is the prediction-grounded definition of (non-)identifiability: a flat
    direction yields ~0 change; an identifiable direction yields a large change.
    The step is relative to each parameter's natural value, so the probe is
    scale-aware.

    Args:
        spec: The model.
        direction: Unit vector in parameter space (length ``n_params``).
        battery: Held-out designs.
        backend: Simulator.
        step: Relative step size along ``direction``.
        sigma_floor: Noise floor.

    Returns:
        RMS over the battery of ``(y(theta+) - y(theta-)) / (2 sigma)``.
    """
    theta = np.array([p.natural_value for p in spec.parameters], dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    # Move as a scalar multiple of d (a true step *along* the direction); scaling
    # each component by |theta| would rotate off the direction and break the
    # correspondence with the SVD null space.
    delta = step * float(np.linalg.norm(theta)) * d
    spec_plus = _perturbed(spec, theta + delta)
    spec_minus = _perturbed(spec, theta - delta)

    sq: list[float] = []
    for design in battery:
        rp = backend.simulate(spec_plus, design)
        rm = backend.simulate(spec_minus, design)
        sigma = observation_sigma(
            rp.observed, spec.observation.error, floor=sigma_floor
        )
        sq.extend(((rp.observed - rm.observed) / (2.0 * sigma)).tolist())
    return float(np.sqrt(np.mean(np.square(sq)))) if sq else 0.0


def _perturbed(spec: ModelSpec, theta_natural: FloatArray) -> ModelSpec:
    """Copy ``spec`` with parameters set to ``theta_natural`` (linear coord)."""
    new_params = tuple(
        p.model_copy(update={"value": float(theta_natural[j]), "coord": "linear"})
        for j, p in enumerate(spec.parameters)
    )
    return spec.model_copy(update={"parameters": new_params})
