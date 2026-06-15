"""Tests for PK metrics and PD response functions."""

import numpy as np
import pytest
from omegaconf import DictConfig

from drrl.sim import SimConfig, get_backend
from drrl.verifier import compute_pk_metrics, hill, pkpd_metrics
from tests.factories import iv_bolus_design, one_comp_macro

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-10, atol=1e-12))


def test_hill_half_max_at_ec50():
    e = hill(np.array([50.0]), emax=100.0, ec50=50.0, hill_coef=1.0)
    assert e[0] == pytest.approx(50.0)


def test_hill_steepness_with_coefficient():
    # Higher Hill coefficient -> sharper transition (lower below EC50).
    low = hill(np.array([25.0]), 100.0, 50.0, hill_coef=4.0)[0]
    lin = hill(np.array([25.0]), 100.0, 50.0, hill_coef=1.0)[0]
    assert low < lin


def test_pk_metrics_one_compartment(tol: DictConfig):
    cl, v, dose = 2.0, 10.0, 100.0
    spec = one_comp_macro(cl=cl, v=v)
    design = iv_bolus_design(tuple(np.linspace(0.5, 12.0, 24)), amount=dose)
    report = pkpd_metrics(spec, design, _BACKEND, dose=dose, horizon=80.0)
    m = report.metrics
    # AUC_inf = Dose/CL = 50; CL recovered; Cmax = Dose/V = 10 at t=0;
    # t1/2 = ln2 / (CL/V) = ln2 / 0.2.
    assert m["auc_inf"] == pytest.approx(50.0, rel=1e-2)
    assert m["cl"] == pytest.approx(cl, rel=1e-2)
    assert m["cmax"] == pytest.approx(dose / v, rel=1e-2)
    assert m["thalf"] == pytest.approx(np.log(2.0) / (cl / v), rel=1e-2)


def test_compute_pk_metrics_direct():
    t = np.linspace(0.0, 50.0, 5000)
    c = 10.0 * np.exp(-0.2 * t)  # 1-comp, V=10, ke=0.2, dose=100
    m = compute_pk_metrics(t, c, dose=100.0)
    assert m["cmax"] == pytest.approx(10.0, rel=1e-3)
    assert m["thalf"] == pytest.approx(np.log(2) / 0.2, rel=1e-2)
    assert m["cl"] == pytest.approx(2.0, rel=1e-2)
