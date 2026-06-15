"""Tests for the stub policies."""

from drrl.env import AddCompartment, Commit, Observation, SwapKinetics
from drrl.policy import GoToStructurePolicy, ScriptedPolicy


def _obs(structure: str) -> Observation:
    return Observation(
        turn=0,
        turns_left=5,
        structure=structure,  # type: ignore[arg-type]
        tierA_gates={"schema": True},
        feedback="",
        identifiable_fraction=1.0,
        observed_rmse=0.0,
    )


def test_scripted_policy_replays_then_commits():
    p = ScriptedPolicy([AddCompartment()])
    assert isinstance(p.act(_obs("one_compartment")), AddCompartment)
    assert isinstance(p.act(_obs("two_compartment")), Commit)  # exhausted -> Commit


def test_goto_structure_navigates_and_commits():
    p = GoToStructurePolicy("two_compartment")
    assert isinstance(p.act(_obs("one_compartment")), AddCompartment)
    assert isinstance(p.act(_obs("two_compartment")), Commit)


def test_goto_michaelis_from_one():
    p = GoToStructurePolicy("michaelis_menten")
    assert isinstance(p.act(_obs("one_compartment")), SwapKinetics)
    assert isinstance(p.act(_obs("michaelis_menten")), Commit)


def test_goto_abstain_variant():
    from drrl.env import Abstain

    p = GoToStructurePolicy("one_compartment", abstain=True)
    assert isinstance(p.act(_obs("one_compartment")), Abstain)
