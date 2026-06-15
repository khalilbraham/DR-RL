"""Supervised warm-start on canonicalized targets.

Targets are recorded by canonical key so a policy that proposes a valid
*reparameterization* of the target structure is not penalized. The trainer is
expressed over the :class:`~drrl.policy.base.Policy` interface; the CPU stub
``ImitationPolicy`` memorizes a structure->action table so the env loop is
testable without an LLM. Abstention exemplars can be synthesized and mixed in.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from drrl.env.actions import Action, Commit
from drrl.env.environment import Observation
from drrl.env.state import ModelState, Structure, build_spec
from drrl.rl.reject import Trajectory


@dataclass(frozen=True)
class SFTExample:
    """One supervised example.

    Attributes:
        observation: The (partial) observation shown to the agent.
        target_action: The action to imitate.
        target_canonical_key: Canonical key of the resulting structure (so valid
            reparameterizations of the target are treated as equivalent).
    """

    observation: Observation
    target_action: Action
    target_canonical_key: str


def _canonical_key(structure: Structure) -> str:
    return build_spec(ModelState.initial(structure)).canonicalize().key


def build_sft_dataset(trajectories: list[Trajectory]) -> list[SFTExample]:
    """Flatten accepted trajectories into canonical-target SFT examples."""
    examples: list[SFTExample] = []
    for traj in trajectories:
        key = _canonical_key(traj.committed_structure)  # type: ignore[arg-type]
        for obs, action in traj.steps:
            examples.append(
                SFTExample(
                    observation=obs, target_action=action, target_canonical_key=key
                )
            )
    return examples


class ImitationPolicy:
    """A CPU stub policy: memorizes the majority target action per structure."""

    def __init__(self, table: dict[Structure, Action]) -> None:
        """Initialize from a structure -> action table."""
        self._table = dict(table)

    def act(self, obs: Observation) -> Action:
        """Return the learned action for the observed structure (else Commit)."""
        return self._table.get(obs.structure, Commit())


def fit_sft(dataset: list[SFTExample]) -> ImitationPolicy:
    """Fit the imitation stub: majority-vote action per observed structure.

    Args:
        dataset: Supervised examples (e.g. from rejection sampling).

    Returns:
        An :class:`ImitationPolicy`.
    """
    by_structure: dict[Structure, Counter[Action]] = {}
    for ex in dataset:
        by_structure.setdefault(ex.observation.structure, Counter())[
            ex.target_action
        ] += 1
    table = {
        structure: counts.most_common(1)[0][0]
        for structure, counts in by_structure.items()
    }
    return ImitationPolicy(table)
