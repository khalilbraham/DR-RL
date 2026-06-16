"""Tests for the process reward model and best-first edit search (Phase 6)."""

from drrl.data.synth.library import michaelis_menten, two_compartment
from drrl.env import ModelEditEnv, Observation
from drrl.reward import RewardWeights
from drrl.rl import (
    ProcessRewardModel,
    best_first_repair,
    observation_potential,
    one_shot_commit,
)
from drrl.sim import SimConfig, get_backend
from drrl.spec import Design, Dose, Unit

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-9, atol=1e-12))
_MG = Unit(expr="mg")


def _obs(
    structure="one_compartment", *, gates_ok=True, ident=1.0, rmse=0.0
) -> Observation:
    return Observation(
        turn=0,
        turns_left=5,
        structure=structure,
        tierA_gates={"schema": True, "units": gates_ok, "execution": True},
        feedback="",
        identifiable_fraction=ident,
        observed_rmse=rmse,
    )


def _env(initial="one_compartment", reference=None) -> ModelEditEnv:
    obs = Design(
        doses=(Dose(compartment="A1", amount=100.0, unit=_MG),),
        sample_times=(0.5, 1.0, 2.0, 4.0, 8.0),
    )
    hidden = [
        Design(
            doses=(Dose(compartment="A1", amount=a, unit=_MG),),
            sample_times=(0.25, 0.75, 1.5, 3.0, 6.0),
        )
        for a in (50.0, 150.0)
    ]
    return ModelEditEnv(
        reference or two_compartment(),
        obs,
        hidden,
        _BACKEND,
        RewardWeights(),
        max_turns=6,
        initial_structure=initial,
    )


def test_potential_zero_when_tier_a_fails():
    assert observation_potential(_obs(gates_ok=False)) == 0.0


def test_potential_rewards_fit_and_identifiability():
    good = observation_potential(_obs(ident=1.0, rmse=0.0))
    poor_fit = observation_potential(_obs(ident=1.0, rmse=5.0))
    poor_id = observation_potential(_obs(ident=0.3, rmse=0.0))
    assert good > poor_fit
    assert good > poor_id


def test_step_reward_positive_for_improving_edit():
    prm = ProcessRewardModel()
    before = _obs(rmse=5.0)  # bad fit
    after = _obs(rmse=0.0)  # good fit
    assert prm.step_reward(before, after) > 0


def test_best_first_repair_beats_one_shot_two_compartment():
    env = _env(initial="one_compartment", reference=two_compartment())
    one_shot = one_shot_commit(env)
    res = best_first_repair(env, ProcessRewardModel())
    assert res.committed_structure == "two_compartment"
    assert res.terminal_reward > one_shot


def test_best_first_repair_finds_michaelis_menten():
    env = _env(initial="one_compartment", reference=michaelis_menten(vmax=8.0, km=3.0))
    one_shot = one_shot_commit(env)
    res = best_first_repair(env, ProcessRewardModel())
    assert res.committed_structure == "michaelis_menten"
    assert res.terminal_reward >= one_shot


def test_repair_no_op_when_already_correct():
    env = _env(initial="two_compartment", reference=two_compartment())
    res = best_first_repair(env, ProcessRewardModel())
    assert res.committed_structure == "two_compartment"
