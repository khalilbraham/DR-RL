"""Policy interface and CPU-only stub policies.

The RL/env loop is driven through a single :class:`Policy` Protocol so it is
fully unit-testable without an LLM. The real LLM-backed proposer (Phase 5)
implements the same interface behind a thin adapter.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from drrl.env.actions import (
    Abstain,
    Action,
    AddCompartment,
    Commit,
    RemoveCompartment,
    SwapKinetics,
)
from drrl.env.environment import Observation
from drrl.env.state import Structure, valid_structural_edits
from drrl.spec import Design, Dose, Unit


@runtime_checkable
class Policy(Protocol):
    """Maps a partial observation to an action."""

    def act(self, obs: Observation) -> Action:
        """Choose an action for ``obs``."""
        ...


class ScriptedPolicy:
    """Replays a fixed action sequence (deterministic; for tests/imitation)."""

    def __init__(self, actions: list[Action]) -> None:
        """Initialize with the action script."""
        self._actions = list(actions)
        self._i = 0

    def act(self, obs: Observation) -> Action:
        """Return the next scripted action (commits if the script is exhausted)."""
        if self._i >= len(self._actions):
            return Commit()
        action = self._actions[self._i]
        self._i += 1
        return action

    def reset(self) -> None:
        """Rewind the script."""
        self._i = 0


def _edit_toward(current: Structure, target: Structure) -> Action:
    """One edit moving ``current`` toward ``target`` (or Commit if equal)."""
    if current == target:
        return Commit()
    if target == "two_compartment":
        return AddCompartment() if current == "one_compartment" else SwapKinetics()
    if target == "michaelis_menten":
        return SwapKinetics() if current == "one_compartment" else RemoveCompartment()
    # target == one_compartment
    return RemoveCompartment() if current == "two_compartment" else SwapKinetics()


class GoToStructurePolicy:
    """Navigates to a target structure (>=2 steps), then commits or abstains."""

    def __init__(self, target: Structure, *, abstain: bool = False) -> None:
        """Initialize with the target structure and whether to abstain at the end."""
        self.target = target
        self.abstain = abstain

    def act(self, obs: Observation) -> Action:
        """Edit toward the target; terminate once reached."""
        if obs.structure == self.target:
            if self.abstain:
                return Abstain(proposed_design=_default_probe_design())
            return Commit()
        return _edit_toward(obs.structure, self.target)


class RandomEditPolicy:
    """Applies random valid edits, then commits (seeded; for rejection sampling)."""

    def __init__(self, seed: int = 0, *, max_edits: int = 3) -> None:
        """Initialize the seeded RNG and the edit budget."""
        self._rng = random.Random(seed)
        self.max_edits = max_edits
        self._n = 0

    def act(self, obs: Observation) -> Action:
        """Either commit or take a random valid structural edit."""
        if self._n >= self.max_edits or self._rng.random() < 0.4:
            return Commit()
        self._n += 1
        return self._rng.choice(valid_structural_edits(obs.structure))

    def reset(self) -> None:
        """Reset the edit counter (keeps RNG state advancing across episodes)."""
        self._n = 0


def _default_probe_design() -> Design:
    """A discriminating probe design used by abstaining policies."""
    return Design(
        doses=(Dose(compartment="A1", amount=100.0, unit=Unit(expr="mg")),),
        sample_times=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    )
