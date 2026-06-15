"""Reward composition: gating, distinguishability-relativity, NaN-safety."""

import pytest
from omegaconf import DictConfig

from drrl.reward import RewardContext, RewardWeights, compose_reward
from drrl.reward.breakdown import RewardBreakdown, RewardError, safe01
from drrl.sim import SimConfig, get_backend
from drrl.verifier import verify
from drrl.verifier.report import VerifierReport
from tests.factories import (
    iv_bolus_design,
    one_comp_macro,
    one_comp_micro,
    with_prop_error,
)

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-10, atol=1e-12))
_HIDDEN = [
    iv_bolus_design((0.5, 1.0, 2.0, 4.0, 8.0)),
    iv_bolus_design((0.25, 0.75, 1.5, 3.0, 6.0, 12.0), amount=150.0),
]


def _weights(tol_cfg: DictConfig | None = None) -> RewardWeights:
    return RewardWeights()


def _report(spec) -> VerifierReport:
    return verify(spec, _HIDDEN[0], _BACKEND, hidden_battery=_HIDDEN)


def test_safe01_handles_nan_and_range():
    assert safe01(float("nan")) == 0.0
    assert safe01(float("inf")) == 0.0
    assert safe01(-1.0) == 0.0
    assert safe01(2.0) == 1.0
    assert safe01(0.3) == 0.3


def test_assert_no_nan_raises():
    bad = RewardBreakdown(
        gate=1.0,
        r_eq=float("nan"),
        r_fit=0.0,
        r_pk=0.0,
        r_pd=0.0,
        r_identify=0.0,
        r_parsimony=0.0,
        r_abstain=0.0,
        r_expl=0.0,
        total=0.0,
    )
    with pytest.raises(RewardError):
        bad.assert_no_nan()


def test_tier_a_failure_zeroes_reward():
    ref = with_prop_error(one_comp_macro())
    # A units-broken candidate: clearance used as a first-order rate.
    from drrl.spec import (
        Compartment,
        ModelSpec,
        ObservationModel,
        ODETerm,
        Parameter,
        Unit,
    )

    broken = ModelSpec(
        compartments=(Compartment(name="A1", unit=Unit(expr="mg")),),
        odes=(ODETerm(target="A1", expr="-CL*A1"),),  # L/h * mg != mg/h
        parameters=(
            Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
            Parameter(name="V", value=10.0, unit=Unit(expr="L")),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )
    report = _report(broken)
    ctx = RewardContext(candidate=broken, reference=ref, hidden_battery=tuple(_HIDDEN))
    rb = compose_reward(report, ctx, _weights(), _BACKEND)
    assert rb.gate == 0.0
    assert rb.total == 0.0


def test_indistinguishable_candidate_beats_distinguishable():
    ref = with_prop_error(one_comp_macro(cl=2.0, v=10.0))
    good = with_prop_error(one_comp_micro(ke=0.2, v=10.0))  # == ref (reparam)
    bad = with_prop_error(one_comp_micro(ke=0.6, v=10.0))  # different elimination

    ctx_good = RewardContext(
        candidate=good, reference=ref, hidden_battery=tuple(_HIDDEN)
    )
    ctx_bad = RewardContext(candidate=bad, reference=ref, hidden_battery=tuple(_HIDDEN))
    rb_good = compose_reward(_report(good), ctx_good, _weights(), _BACKEND)
    rb_bad = compose_reward(_report(bad), ctx_bad, _weights(), _BACKEND)

    assert rb_good.r_eq == pytest.approx(1.0)
    assert rb_good.total > rb_bad.total


def test_weights_load_from_config(tol: DictConfig):
    from pathlib import Path

    from drrl.utils.config import load_config

    cfg = load_config(
        Path(__file__).resolve().parents[2] / "configs" / "reward" / "default.yaml"
    )
    w = RewardWeights.from_config(cfg)
    assert w.w_eq == pytest.approx(0.25)
    assert w.wrong_commit_penalty == pytest.approx(1.0)
    assert w.over_abstain_penalty == pytest.approx(0.3)
