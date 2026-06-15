"""Reward-hacking adversarial tests (acceptance criteria #4 and #5)."""

from drrl.reward import RewardContext, RewardWeights, compose_reward
from drrl.sim import SimConfig, get_backend
from drrl.verifier import verify
from tests.factories import (
    add_unused_param,
    iv_bolus_design,
    one_comp_macro,
    two_comp_macro,
    two_comp_micro_matched,
    with_prop_error,
)

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-10, atol=1e-12))
_HIDDEN = [
    iv_bolus_design((0.5, 1.0, 2.0, 4.0, 8.0, 12.0)),
    iv_bolus_design((0.25, 0.75, 1.5, 3.0, 6.0), amount=150.0),
]


def _reward(candidate, reference, weights, **ctx_kw):
    report = verify(candidate, _HIDDEN[0], _BACKEND, hidden_battery=_HIDDEN)
    ctx = RewardContext(
        candidate=candidate,
        reference=reference,
        hidden_battery=tuple(_HIDDEN),
        **ctx_kw,
    )
    return compose_reward(report, ctx, weights, _BACKEND)


def test_parsimony_cannot_collapse_model():
    # Reference is a 2-compartment model.
    ref = with_prop_error(two_comp_macro())
    correct = with_prop_error(two_comp_micro_matched())  # == ref, 4 params
    overcomplex = with_prop_error(
        add_unused_param(two_comp_micro_matched())
    )  # 5 params, == fit
    empty = with_prop_error(one_comp_macro())  # under-fits the 2c data

    # Crank parsimony weight absurdly high to give the empty model every chance.
    greedy = RewardWeights(
        w_eq=0.1,
        w_fit=0.1,
        w_pk=0.0,
        w_pd=0.0,
        w_identify=0.0,
        w_parsimony=0.7,
        w_abstain=0.05,
        w_expl=0.05,
    )

    rb_correct = _reward(correct, ref, greedy)
    rb_over = _reward(overcomplex, ref, greedy)
    rb_empty = _reward(empty, ref, greedy)

    # Parsimony only tie-breaks among fit-adequate models: the simpler correct
    # model is not beaten by the fit-equivalent richer one.
    assert rb_correct.total >= rb_over.total
    assert rb_over.r_parsimony < rb_correct.r_parsimony
    # The empty model under-fits, so parsimony is gated to zero and it cannot win,
    # even with a dominant parsimony weight.
    assert rb_empty.r_parsimony == 0.0
    assert rb_empty.total < rb_correct.total


def test_fit_uses_hidden_design_battery():
    # The agent "observed" one design; fit must be scored on a disjoint hidden set.
    observed = iv_bolus_design((100.0,))  # a single late point the agent saw
    ref = with_prop_error(one_comp_macro(cl=2.0, v=10.0))
    candidate = with_prop_error(one_comp_macro(cl=4.0, v=10.0))  # different from ref

    observed_times = {observed.sample_times}
    hidden_times = {d.sample_times for d in _HIDDEN}
    assert observed_times.isdisjoint(hidden_times)  # genuinely held out

    # Record every design the backend simulates during reward composition.
    simulated: list[tuple[float, ...]] = []

    class _Recorder:
        def simulate(self, spec, design, *, with_sensitivities=False):
            simulated.append(design.sample_times)
            return _BACKEND.simulate(
                spec, design, with_sensitivities=with_sensitivities
            )

    report = verify(candidate, _HIDDEN[0], _Recorder(), hidden_battery=_HIDDEN)
    ctx = RewardContext(
        candidate=candidate,
        reference=ref,
        hidden_battery=tuple(_HIDDEN),
        observed_designs=(observed,),
    )
    compose_reward(report, ctx, RewardWeights(), _Recorder())

    # The observed design's sampling grid was never simulated for scoring.
    assert observed.sample_times not in simulated
    # The hidden battery grids WERE simulated (fit is genuinely scored on them).
    assert hidden_times.issubset(set(simulated))


def test_abstention_penalty_is_asymmetric():
    ref = with_prop_error(two_comp_macro())
    wrong = with_prop_error(one_comp_macro())  # distinguishable from ref

    # Confident wrong commit.
    rb_commit = _reward(wrong, ref, RewardWeights(), action="commit")
    # Over-abstaining when the set is NOT ambiguous (few admissible).
    rb_overabstain = _reward(
        wrong, ref, RewardWeights(), action="abstain", admissible_keys=("only_one",)
    )

    assert rb_commit.r_abstain < rb_overabstain.r_abstain  # wrong commit hurts more


def test_correct_abstention_when_ambiguous_is_rewarded():
    ref = with_prop_error(one_comp_macro())
    candidate = with_prop_error(one_comp_macro())
    proposed = iv_bolus_design((0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
    rb = _reward(
        candidate,
        ref,
        RewardWeights(),
        action="abstain",
        proposed_design=proposed,
        admissible_keys=("a", "b", "c"),  # large admissible set => abstention is right
    )
    assert rb.r_abstain == 1.0
