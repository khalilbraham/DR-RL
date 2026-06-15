"""Invariant #1: the simulator reproduces closed-form analytic solutions.

Tested against the closed forms, not against the simulator itself. Both backends
(diffrax and the independent scipy cross-check) must pass.
"""

import numpy as np
import pytest
from omegaconf import DictConfig

from drrl.sim import SimConfig, get_backend
from tests.factories import iv_bolus_design, one_comp_macro, two_comp_micro

# Tight integrator settings so numerical error is far below the assert tolerance.
# These are solver controls, not scientific thresholds.
_TIGHT = SimConfig(rtol=1e-11, atol=1e-13)
_BACKENDS = ["diffrax", "scipy"]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_one_compartment_iv_bolus_exponential(backend: str, tol: DictConfig):
    cl, v, dose = 2.0, 10.0, 100.0
    spec = one_comp_macro(cl=cl, v=v)
    times = tuple(np.linspace(0.5, 8.0, 16))
    design = iv_bolus_design(times, amount=dose)

    result = get_backend(backend, _TIGHT).simulate(spec, design)
    expected = (dose / v) * np.exp(-(cl / v) * np.asarray(times))

    assert result.integrator_ok
    np.testing.assert_allclose(
        result.observed, expected, rtol=tol.analytic_rtol, atol=tol.analytic_atol
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_one_compartment_auc(backend: str, tol: DictConfig):
    # AUC_0->inf = Dose / CL for a 1-compartment IV bolus.
    cl, v, dose = 2.0, 10.0, 100.0
    spec = one_comp_macro(cl=cl, v=v)
    grid = tuple(np.linspace(0.0, 300.0, 60001))
    design = iv_bolus_design(grid, amount=dose)

    result = get_backend(backend, _TIGHT).simulate(spec, design)
    auc = float(np.trapezoid(result.observed, np.asarray(grid)))
    assert auc == pytest.approx(dose / cl, rel=tol.analytic_rtol)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_two_compartment_biexponential(backend: str, tol: DictConfig):
    spec = two_comp_micro()
    k10, k12, k21, v1, dose = 0.3, 0.2, 0.1, 8.0, 100.0
    times = tuple(np.linspace(0.5, 8.0, 16))
    design = iv_bolus_design(times, amount=dose, into="C")

    result = get_backend(backend, _TIGHT).simulate(spec, design)

    s = k10 + k12 + k21
    p = k10 * k21
    alpha = (s + np.sqrt(s * s - 4 * p)) / 2
    beta = (s - np.sqrt(s * s - 4 * p)) / 2
    t = np.asarray(times)
    expected = (dose / v1) * (
        ((k21 - alpha) / (beta - alpha)) * np.exp(-alpha * t)
        + ((k21 - beta) / (alpha - beta)) * np.exp(-beta * t)
    )

    assert result.integrator_ok
    np.testing.assert_allclose(
        result.observed, expected, rtol=tol.analytic_rtol, atol=tol.analytic_atol
    )


def test_backends_agree_on_states(tol: DictConfig):
    spec = two_comp_micro()
    times = tuple(np.linspace(0.5, 10.0, 20))
    design = iv_bolus_design(times, into="C")
    rd = get_backend("diffrax", _TIGHT).simulate(spec, design)
    rs = get_backend("scipy", _TIGHT).simulate(spec, design)
    np.testing.assert_allclose(rd.states, rs.states, rtol=1e-6, atol=1e-8)
