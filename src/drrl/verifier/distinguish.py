"""CORE: calibrated predictive distinguishability oracle.

Decides whether a candidate model is *observationally indistinguishable* from a
reference at the operative noise level and design.

- **Linear fast path (exact):** for linear compartmental systems, two models have
  the same input-output behaviour iff their Markov-parameter sequences
  ``{c A^k b}`` agree (realization theory — equivalent to transfer-function
  identity, but pure NumPy and numerically robust). This is the ground truth the
  general oracle is calibrated against.
- **General predictive path:** simulate both at the operative noise level over a
  design battery and compute the noncentrality
  ``lambda = sum ((y_ref - y_cand)/sigma)^2``.
  Declare *distinguishable* iff ``lambda`` exceeds the chi-square quantile at the
  configured nominal false-positive rate.

The admissible (equivalence-class) set is assembled at runtime from an open
competitor set (see :mod:`drrl.verifier.competitors`), never a hardcoded list.
"""

from __future__ import annotations

import numpy as np
import sympy
from numpy.typing import NDArray
from scipy.stats import chi2

from drrl.sim.backend import Backend
from drrl.spec.model import Design, ModelSpec
from drrl.verifier.identify import observation_sigma
from drrl.verifier.report import DistinguishReport

FloatArray = NDArray[np.float64]


def is_linear(spec: ModelSpec) -> bool:
    """Whether every ODE RHS is linear in the states (constant Jacobian)."""
    syms = spec.symbols
    state_syms = [syms[n] for n in spec.state_names]
    state_set = set(state_syms)
    for expr in spec.rhs_exprs().values():
        for s in state_syms:
            if sympy.diff(expr, s).free_symbols & state_set:
                return False
    return True


def _linear_system(
    spec: ModelSpec, dose_compartment: str
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return numeric ``(A, b, c)`` for the linear realization at natural params."""
    syms = spec.symbols
    states = list(spec.state_names)
    n = len(states)
    rhs = spec.rhs_exprs()
    psub = {syms[p.name]: p.natural_value for p in spec.parameters}
    a = np.array(
        [
            [
                float(sympy.diff(rhs[states[i]], syms[states[j]]).subs(psub))
                for j in range(n)
            ]
            for i in range(n)
        ],
        dtype=np.float64,
    )
    b = np.zeros(n)
    b[states.index(dose_compartment)] = 1.0
    obs = spec.observation
    volume = float(psub[syms[obs.divide_by]]) if obs.divide_by is not None else 1.0
    c = np.zeros(n)
    c[states.index(obs.state)] = 1.0 / volume
    return a, b, c


def markov_parameters(
    spec: ModelSpec, dose_compartment: str, n_markov: int
) -> FloatArray:
    """Markov parameters ``c A^k b`` for ``k = 0 .. n_markov-1`` (linear systems)."""
    a, b, c = _linear_system(spec, dose_compartment)
    powers = np.empty(n_markov)
    mat = np.eye(a.shape[0])
    for k in range(n_markov):
        powers[k] = float(c @ mat @ b)
        mat = mat @ a
    return powers


def linear_indistinguishable(
    ref: ModelSpec,
    cand: ModelSpec,
    ref_dose: str,
    cand_dose: str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> bool:
    """Exact linear verdict: equal Markov sequences => indistinguishable."""
    order = max(len(ref.state_names), len(cand.state_names))
    n_markov = 2 * order + 1
    mr = markov_parameters(ref, ref_dose, n_markov)
    mc = markov_parameters(cand, cand_dose, n_markov)
    return bool(np.allclose(mr, mc, rtol=rtol, atol=atol))


def predictive_noncentrality(
    ref: ModelSpec,
    cand: ModelSpec,
    battery: list[Design],
    backend: Backend,
    *,
    sigma_floor: float = 1e-9,
) -> tuple[float, int]:
    """Noise-normalized ``(lambda, n_points)`` between ref and candidate predictions."""
    lam = 0.0
    n_points = 0
    for design in battery:
        yr = backend.simulate(ref, design).observed
        yc = backend.simulate(cand, design).observed
        sigma = observation_sigma(yr, ref.observation.error, floor=sigma_floor)
        lam += float(np.sum(((yr - yc) / sigma) ** 2))
        n_points += yr.size
    return lam, n_points


def predictive_threshold(n_points: int, *, alpha: float) -> float:
    """Chi-square upper quantile at level ``alpha`` with ``n_points`` dof."""
    return float(chi2.ppf(1.0 - alpha, df=max(n_points, 1)))


def distinguish(
    ref: ModelSpec,
    cand: ModelSpec,
    battery: list[Design],
    backend: Backend,
    *,
    alpha: float = 0.05,
    sigma_floor: float = 1e-9,
    markov_rtol: float = 1e-6,
    markov_atol: float = 1e-9,
    admissible: tuple[str, ...] = (),
) -> DistinguishReport:
    """Distinguishability verdict for ``cand`` vs ``ref`` over a design battery.

    Uses the exact linear Markov criterion when both models are linear; otherwise
    the calibrated predictive noncentrality test.

    Args:
        ref: Reference model.
        cand: Candidate model.
        battery: Designs (valid for both models, i.e. shared compartment names).
        backend: Simulator.
        alpha: Nominal false-positive rate for the predictive test.
        sigma_floor: Noise floor.
        markov_rtol: Relative tol for Markov-parameter equality.
        markov_atol: Absolute tol for Markov-parameter equality.
        admissible: Canonical keys of the admissible set to record on the report.

    Returns:
        A :class:`DistinguishReport`.
    """
    dose = battery[0].doses[0].compartment
    if is_linear(ref) and is_linear(cand):
        indist = linear_indistinguishable(
            ref, cand, dose, dose, rtol=markov_rtol, atol=markov_atol
        )
        order = max(len(ref.state_names), len(cand.state_names))
        mr = markov_parameters(ref, dose, 2 * order + 1)
        mc = markov_parameters(cand, dose, 2 * order + 1)
        stat = float(np.max(np.abs(mr - mc)))
        return DistinguishReport(
            verdict="indistinguishable" if indist else "distinguishable",
            method="transfer_function",
            statistic=stat,
            threshold=markov_atol + markov_rtol * float(np.max(np.abs(mr))),
            admissible=admissible,
        )

    lam, n_points = predictive_noncentrality(
        ref, cand, battery, backend, sigma_floor=sigma_floor
    )
    thresh = predictive_threshold(n_points, alpha=alpha)
    return DistinguishReport(
        verdict="distinguishable" if lam > thresh else "indistinguishable",
        method="predictive",
        statistic=lam,
        threshold=thresh,
        admissible=admissible,
    )


def predictive_distinguishable(
    ref: ModelSpec,
    cand: ModelSpec,
    battery: list[Design],
    backend: Backend,
    *,
    alpha: float = 0.05,
    sigma_floor: float = 1e-9,
) -> bool:
    """General predictive verdict (ignores the linear fast path) — for calibration."""
    lam, n_points = predictive_noncentrality(
        ref, cand, battery, backend, sigma_floor=sigma_floor
    )
    return lam > predictive_threshold(n_points, alpha=alpha)
