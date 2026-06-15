"""CPU tests for the GRPO reward, diversity bonus, and ablation modes.

These prove the GRPO loop's *scientific* core without a GPU: the verifier-grounded
reward correctly ranks decisions, and the manipulation check is visible at the
reward level (removing r_identify raises the reward of a fit-but-flat commit).
"""

import pytest

from drrl.reward import RewardWeights
from drrl.rl import build_registry, parse_completion, score_decisions, weights_for_mode
from drrl.rl.decision import Decision
from drrl.rl.grpo import make_reward_funcs
from drrl.sim import SimConfig, get_backend

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-8, atol=1e-10))
_REG = build_registry()


def _reward(mode: str, case_id: str, completion: str) -> float:
    funcs = make_reward_funcs(_REG, mode=mode, backend=_BACKEND)  # type: ignore[arg-type]
    case = _REG.get(case_id)
    return funcs[0]([case.prompt], [completion], case_id=[case_id])[0]


# --- parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("reasoning...\nANSWER: two_compartment", ("two_compartment", False, True)),
        ("ANSWER: abstain", (None, True, True)),
        ("I think 2c is best. ANSWER: 2-compartment", ("two_compartment", False, True)),
        ("ANSWER: michaelis-menten", ("michaelis_menten", False, True)),
        ("no answer here", (None, False, False)),
    ],
)
def test_parse_completion(text, expected):
    d = parse_completion(text)
    assert (d.structure, d.abstain, d.valid) == expected


# --- reward ranking ----------------------------------------------------------


def test_correct_commit_outranks_wrong_on_commit_case():
    r_correct = _reward("full", "commit_2c", "ANSWER: two_compartment")
    r_wrong = _reward("full", "commit_2c", "ANSWER: one_compartment")
    assert r_correct > r_wrong


def test_invalid_answer_scores_zero():
    assert _reward("full", "commit_1c", "I am not sure.") == 0.0


def test_manipulation_check_at_reward_level():
    # On a Michaelis-Menten case dosed below saturation, committing MM is a
    # "fit-but-flat" choice (parameters practically non-identifiable). r_identify
    # penalizes it: the margin by which abstaining beats the flat commit is larger
    # WITH r_identify than without -> removing r_identify lets fit-but-flat return.
    margin_full = _reward("full", "abstain_mm_low", "ANSWER: abstain") - _reward(
        "full", "abstain_mm_low", "ANSWER: michaelis_menten"
    )
    margin_noid = _reward("no_identify", "abstain_mm_low", "ANSWER: abstain") - _reward(
        "no_identify", "abstain_mm_low", "ANSWER: michaelis_menten"
    )
    assert margin_full > margin_noid


def test_abstain_is_best_action_on_ambiguous_case():
    rewards = {
        a: _reward("full", "abstain_mm_low", f"ANSWER: {a}")
        for a in ("abstain", "one_compartment", "two_compartment", "michaelis_menten")
    }
    assert max(rewards, key=lambda k: rewards[k]) == "abstain"


# --- diversity bonus ---------------------------------------------------------


def test_diversity_bonus_rewards_varied_group():
    funcs = make_reward_funcs(_REG, mode="full", backend=_BACKEND, diversity_weight=0.5)
    diversity = funcs[1]
    prompt = _REG.get("commit_1c").prompt
    same = diversity([prompt] * 4, ["ANSWER: one_compartment"] * 4)
    varied = diversity(
        [prompt] * 4,
        [
            "ANSWER: one_compartment",
            "ANSWER: two_compartment",
            "ANSWER: michaelis_menten",
            "ANSWER: abstain",
        ],
    )
    assert max(same) == 0.0
    assert min(varied) > 0.0


# --- ablation modes ----------------------------------------------------------


def test_weights_for_mode():
    base = RewardWeights()
    assert weights_for_mode("full", base) == base
    assert weights_for_mode("no_identify", base).w_identify == 0.0
    fit = weights_for_mode("fit_only", base)
    assert fit.w_eq == 0.0 and fit.w_abstain == 0.0 and fit.w_identify == 0.0
    assert fit.w_fit > 0.0


# --- evaluation metrics ------------------------------------------------------


def test_score_decisions_metrics():
    # A "good" policy: commits correctly, abstains on ambiguous cases.
    decisions: dict[str, Decision] = {}
    for cid, case in _REG.cases.items():
        if case.correct_action == "commit":
            decisions[cid] = Decision(case.correct_structure, abstain=False, valid=True)
        else:
            decisions[cid] = Decision(None, abstain=True, valid=True)
    report = score_decisions(decisions, _REG)
    assert report.commit_accuracy == pytest.approx(1.0)
    assert report.abstain_rate_on_abstain_cases == pytest.approx(1.0)
    assert report.flat_commit_rate == 0.0
    assert report.correct_action_rate == pytest.approx(1.0)
