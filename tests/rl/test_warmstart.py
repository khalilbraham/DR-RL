"""Tests for rejection sampling and SFT warm-start (Phase 4 DoD)."""

from drrl.data.synth.library import two_compartment
from drrl.env import ModelEditEnv
from drrl.env.state import ModelState, build_spec
from drrl.policy import GoToStructurePolicy, RandomEditPolicy
from drrl.reward import RewardWeights
from drrl.rl import build_sft_dataset, fit_sft, rejection_sample, rollout
from drrl.sim import SimConfig, get_backend
from drrl.spec import Design, Dose, Unit
from drrl.verifier import tier_a_gates

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


def _env_factory():
    obs, hidden = _designs()
    return ModelEditEnv(
        two_compartment(), obs, hidden, _BACKEND, RewardWeights(), max_turns=6
    )


def test_rejection_sampling_accepts_correct_trajectories():
    stats = rejection_sample(
        _env_factory,
        lambda _i: GoToStructurePolicy("two_compartment"),
        n_samples=3,
        reward_threshold=0.8,
    )
    assert stats.acceptance_rate == 1.0
    assert all(t.committed_structure == "two_compartment" for t in stats.accepted)


def test_rejection_sampling_rejects_wrong_trajectories():
    stats = rejection_sample(
        _env_factory,
        lambda _i: GoToStructurePolicy("one_compartment"),
        n_samples=3,
        reward_threshold=0.8,
    )
    assert stats.n_accepted == 0


def test_sft_targets_are_canonicalized():
    stats = rejection_sample(
        _env_factory,
        lambda _i: GoToStructurePolicy("two_compartment"),
        n_samples=2,
        reward_threshold=0.8,
    )
    dataset = build_sft_dataset(stats.accepted)
    assert dataset
    expected_key = build_spec(ModelState.initial("two_compartment")).canonicalize().key
    assert all(ex.target_canonical_key == expected_key for ex in dataset)


def test_sft_policy_produces_tier_a_valid_specs():
    # RS -> SFT -> the fitted policy should commit Tier-A-valid models.
    stats = rejection_sample(
        _env_factory,
        lambda _i: GoToStructurePolicy("two_compartment"),
        n_samples=2,
        reward_threshold=0.8,
    )
    policy = fit_sft(build_sft_dataset(stats.accepted))

    traj = rollout(_env_factory(), policy)
    committed = build_spec(ModelState.initial(traj.committed_structure))  # type: ignore[arg-type]
    gates = tier_a_gates(committed, _designs()[0], _BACKEND)
    assert all(gates.values())
    assert traj.committed_structure == "two_compartment"


def test_random_policy_is_seed_deterministic():
    r1 = rollout(_env_factory(), RandomEditPolicy(seed=7))
    r2 = rollout(_env_factory(), RandomEditPolicy(seed=7))
    assert [type(a).__name__ for _, a in r1.steps] == [
        type(a).__name__ for _, a in r2.steps
    ]
