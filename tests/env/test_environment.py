"""Tests for the model-edit environment: determinism, partial obs, terminals."""

import pytest

from drrl.data.synth.library import two_compartment
from drrl.env import Action, AddCompartment, Commit, ModelEditEnv, Observation
from drrl.policy import GoToStructurePolicy, ScriptedPolicy
from drrl.reward import RewardWeights
from drrl.rl import rollout
from drrl.sim import SimConfig, get_backend
from drrl.spec import Design, Dose, Unit

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-9, atol=1e-12))
_MG = Unit(expr="mg")


def _designs():
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
    return obs, hidden


def _env(max_turns: int = 6) -> ModelEditEnv:
    obs, hidden = _designs()
    return ModelEditEnv(
        two_compartment(), obs, hidden, _BACKEND, RewardWeights(), max_turns=max_turns
    )


def test_episode_is_deterministic_and_replayable():
    script: list[Action] = [AddCompartment(), Commit()]
    r1 = rollout(_env(), ScriptedPolicy(list(script)))
    r2 = rollout(_env(), ScriptedPolicy(list(script)))
    assert r1.reward == r2.reward
    assert r1.committed_structure == r2.committed_structure
    # Observation sequences match step by step.
    assert [o.structure for o, _ in r1.steps] == [o.structure for o, _ in r2.steps]


def test_partial_observability_hides_oracle():
    obs_fields = set(Observation.__dataclass_fields__)
    for forbidden in ("reference", "hidden_battery", "admissible_keys", "reward"):
        assert forbidden not in obs_fields


def test_correct_commit_beats_wrong_commit():
    correct = rollout(_env(), GoToStructurePolicy("two_compartment"))
    wrong = rollout(_env(), GoToStructurePolicy("one_compartment"))
    assert correct.committed_structure == "two_compartment"
    assert correct.reward > wrong.reward


def test_step_after_termination_raises():
    env = _env()
    env.reset()
    env.step(Commit())
    with pytest.raises(RuntimeError):
        env.step(Commit())


def test_forced_commit_when_out_of_turns():
    env = _env(max_turns=1)
    env.reset()
    # One structural edit consumes the only turn -> forced terminal.
    _obs, _reward, done, info = env.step(AddCompartment())
    assert done
    assert info.get("forced_commit") is True


def test_abstain_terminal_produces_reward():
    from drrl.env import Abstain

    env = _env()
    env.reset()
    probe = Design(
        doses=(Dose(compartment="A1", amount=100.0, unit=_MG),),
        sample_times=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    _obs, _reward, done, info = env.step(Abstain(proposed_design=probe))
    assert done
    assert "reward_breakdown" in info
