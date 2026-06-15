"""Invariant #2 (full): parameterization-invariant, prediction-based identifiability."""

import numpy as np
import pytest
from omegaconf import DictConfig

from drrl.sim import SimConfig, get_backend
from drrl.verifier import identifiability, prediction_change_along
from tests.factories import (
    confound_product,
    confound_product_reparam,
    held_out_battery,
    one_comp_macro,
    one_comp_micro,
    two_comp_macro,
    two_comp_micro,
)

# scipy backend (finite-difference sensitivities): fast and robust, and gives
# clean rank separation here. The diffrax autodiff path is validated in the sim
# tests; identifiability is backend-agnostic.
_BACKEND = get_backend("scipy", SimConfig(rtol=1e-11, atol=1e-13))


def _report(spec, into="A1", tol=None):
    kw = {}
    if tol is not None:
        kw = {
            "rank_rtol": tol.identify_rank_rtol,
            "sigma_floor": tol.identify_sigma_floor,
        }
    return identifiability(spec, held_out_battery(into), _BACKEND, **kw)


def test_fully_identifiable_models_have_fraction_one(tol: DictConfig):
    assert _report(one_comp_macro(), tol=tol).identifiable_fraction == pytest.approx(
        1.0
    )
    r2 = _report(two_comp_micro(), into="C", tol=tol)
    assert r2.identifiable_fraction == pytest.approx(1.0)
    assert r2.identifiable_rank == 4


def test_identifiability_is_reparameterization_invariant(tol: DictConfig):
    # (CL,V) vs (k10,V): different parameter sets, same identifiability score.
    macro = _report(one_comp_macro(), tol=tol)
    micro = _report(one_comp_micro(), tol=tol)
    assert macro.score == pytest.approx(micro.score)

    # 2-compartment macro vs micro.
    t_macro = _report(two_comp_macro(), into="A1", tol=tol)
    t_micro = _report(two_comp_micro(), into="C", tol=tol)
    assert t_macro.score == pytest.approx(t_micro.score)

    # Deficient-rank reparameterization: (p,q,V) confound vs (r=p*q, q, V).
    conf = _report(confound_product(), tol=tol)
    conf_re = _report(confound_product_reparam(), tol=tol)
    assert conf.identifiable_fraction == pytest.approx(2 / 3)
    assert conf_re.identifiable_fraction == pytest.approx(2 / 3)
    assert conf.score == pytest.approx(conf_re.score)


def test_detects_nonidentifiable_confound(tol: DictConfig):
    report = _report(confound_product(), tol=tol)
    assert report.identifiable_rank == 2
    assert report.n_params == 3
    assert len(report.nonidentifiable_directions) == 1
    # The flat direction lies in the (p, q) plane: V component ~ 0 (FD noise),
    # while the p,q components are O(1).
    null = np.array(report.nonidentifiable_directions[0])
    assert abs(null[2]) < 1e-4
    assert abs(null[0]) > 0.1 and abs(null[1]) > 0.1


def test_prediction_change_grounds_the_rank_verdict(tol: DictConfig):
    # Moving along the flat (null) direction barely changes predictions;
    # moving along an identifiable axis changes them a lot.
    spec = confound_product()
    battery = held_out_battery()
    report = identifiability(
        spec,
        battery,
        _BACKEND,
        rank_rtol=tol.identify_rank_rtol,
        sigma_floor=tol.identify_sigma_floor,
    )
    flat = np.array(report.nonidentifiable_directions[0])
    flat_change = prediction_change_along(spec, flat, battery, _BACKEND)

    identifiable_axis = np.array([0.0, 0.0, 1.0])  # V affects predictions
    id_change = prediction_change_along(spec, identifiable_axis, battery, _BACKEND)

    assert flat_change < tol.identify_flat_threshold
    assert id_change > 10 * flat_change


def test_prediction_affecting_flags(tol: DictConfig):
    report = _report(one_comp_macro(), tol=tol)
    assert all(report.prediction_affecting.values())


def test_empty_battery_raises():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="non-empty"):
        identifiability(one_comp_macro(), [], _BACKEND)
