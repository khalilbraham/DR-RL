"""Derived PK/PD metrics: AUC, Cmax, Tmax, terminal half-life, CL; Hill/Emax.

Metrics are computed from a forward simulation (PK) or as closed-form response
functions (PD). They feed the PK/PD reward terms and the benchmark metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from drrl.sim.backend import Backend
from drrl.spec.model import Design, ModelSpec
from drrl.verifier.report import PkPdReport

FloatArray = NDArray[np.float64]


def hill(
    conc: FloatArray, emax: float, ec50: float, hill_coef: float = 1.0
) -> FloatArray:
    """Hill (sigmoidal Emax) response.

    ``E(C) = Emax * C^h / (EC50^h + C^h)``.

    Args:
        conc: Concentrations (>= 0).
        emax: Maximum effect.
        ec50: Concentration at half-maximal effect.
        hill_coef: Hill coefficient ``h`` (1.0 = simple Emax).

    Returns:
        Effect at each concentration. ``E(EC50) == Emax/2`` for any ``h``.
    """
    c = np.asarray(conc, dtype=np.float64)
    num = emax * np.power(c, hill_coef)
    den = ec50**hill_coef + np.power(c, hill_coef)
    return np.asarray(num / den, dtype=np.float64)


def _terminal_slope(times: FloatArray, conc: FloatArray, *, frac: float = 0.4) -> float:
    """Estimate the terminal elimination rate ``lambda_z`` (log-linear tail fit).

    Returns ``nan`` if the tail is not positive/decreasing enough to fit.
    """
    t = np.asarray(times, dtype=np.float64)
    c = np.asarray(conc, dtype=np.float64)
    t_cut = t[0] + (1.0 - frac) * (t[-1] - t[0])
    mask = (t >= t_cut) & (c > 0)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(t[mask], np.log(c[mask]), 1)
    return float(-slope) if slope < 0 else float("nan")


def compute_pk_metrics(
    times: FloatArray, conc: FloatArray, *, dose: float | None = None
) -> dict[str, float]:
    """Compute PK metrics from a concentration-time curve.

    Args:
        times: Sample times (ascending), shape ``(T,)``.
        conc: Concentrations at ``times``, shape ``(T,)``.
        dose: Administered dose; enables ``cl`` (= dose / AUC_inf).

    Returns:
        Mapping with ``auc`` (0..t_end), ``auc_inf``, ``cmax``, ``tmax``,
        ``thalf``, and ``cl`` (when ``dose`` given and AUC_inf finite).
    """
    t = np.asarray(times, dtype=np.float64)
    c = np.asarray(conc, dtype=np.float64)
    auc_t = float(np.trapezoid(c, t))
    cmax = float(np.max(c))
    tmax = float(t[int(np.argmax(c))])
    lambda_z = _terminal_slope(t, c)
    metrics: dict[str, float] = {"auc": auc_t, "cmax": cmax, "tmax": tmax}
    if np.isfinite(lambda_z) and lambda_z > 0:
        thalf = float(np.log(2.0) / lambda_z)
        auc_inf = auc_t + float(c[-1]) / lambda_z
        metrics["thalf"] = thalf
        metrics["auc_inf"] = auc_inf
        if dose is not None and auc_inf > 0:
            metrics["cl"] = dose / auc_inf
    return metrics


def pkpd_metrics(
    spec: ModelSpec,
    design: Design,
    backend: Backend,
    *,
    dose: float | None = None,
    horizon: float | None = None,
    n_grid: int = 2000,
) -> PkPdReport:
    """Simulate on a fine grid and compute PK metrics.

    Args:
        spec: The model.
        design: Dosing design (doses define initial conditions).
        backend: Simulator.
        dose: Administered dose for ``cl`` (defaults to total bolus amount).
        horizon: End time for the fine grid (defaults to a long multiple so the
            terminal phase is captured).
        n_grid: Number of fine grid points.

    Returns:
        A :class:`PkPdReport`.
    """
    base = max(design.sample_times) if design.sample_times else 1.0
    t_end = horizon if horizon is not None else base * 5.0
    fine = Design(
        doses=design.doses, sample_times=tuple(np.linspace(0.0, float(t_end), n_grid))
    )
    result = backend.simulate(spec, fine)
    total_dose = (
        dose if dose is not None else float(sum(d.amount for d in design.doses))
    )
    metrics = compute_pk_metrics(result.times, result.observed, dose=total_dose)
    return PkPdReport(metrics=metrics)
