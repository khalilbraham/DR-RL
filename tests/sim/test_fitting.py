"""Tests for least-squares best-fit parameter estimation."""

import numpy as np

from drrl.sim import SimConfig, fit_params, get_backend, set_natural_params
from tests.factories import iv_bolus_design, one_comp_macro

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-10, atol=1e-12))


def test_fit_recovers_known_parameters():
    truth = one_comp_macro(cl=3.0, v=12.0)
    design = iv_bolus_design(tuple(np.linspace(0.5, 12.0, 16)))
    target = _BACKEND.simulate(truth, design).observed

    # Start from different defaults; fitting should recover CL, V.
    template = one_comp_macro(cl=1.0, v=5.0)
    fitted, cost = fit_params(template, target, design, _BACKEND)

    values = {p.name: p.natural_value for p in fitted.parameters}
    assert abs(values["CL"] - 3.0) < 0.05
    assert abs(values["V"] - 12.0) < 0.2
    assert cost < 1e-6


def test_set_natural_params_roundtrip():
    spec = one_comp_macro()
    out = set_natural_params(spec, np.array([5.0, 20.0]))
    vals = {p.name: p.value for p in out.parameters}
    assert vals["CL"] == 5.0 and vals["V"] == 20.0
    assert all(p.coord == "linear" for p in out.parameters)
