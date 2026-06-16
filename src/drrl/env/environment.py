"""Multi-turn, partially observable model-edit environment (Gym-like).

The agent observes only diagnostics about *its own* candidate (Tier-A gates,
feedback, its identifiability, and fit on the design it was shown) — never the
reference structure, the hidden battery, or oracle labels. Episodes are
deterministic given the action sequence and terminate on commit or abstain.
Reward is computed only at termination, from the (internal) reference and hidden
battery via the distinguishability-relative reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drrl.env.actions import Abstain, Action, Commit
from drrl.env.state import ModelState, Structure, apply_edit, build_spec
from drrl.reward.breakdown import RewardBreakdown
from drrl.reward.compose import compose_reward
from drrl.reward.context import RewardContext, RewardWeights
from drrl.sim.backend import Backend
from drrl.spec.model import Design, ModelSpec
from drrl.verifier.competitors import CompetitorSet
from drrl.verifier.distinguish import distinguish
from drrl.verifier.identify import identifiability, observation_sigma
from drrl.verifier.pipeline import verify


@dataclass(frozen=True)
class Observation:
    """Partial observation given to the agent (no oracle labels).

    Attributes:
        turn: Current turn index (0-based).
        turns_left: Remaining turns before forced termination.
        structure: Current candidate structure label.
        tierA_gates: Tier-A gate booleans for the candidate.
        feedback: Actionable edit-guiding text.
        identifiable_fraction: Candidate identifiability on the shown design.
        observed_rmse: Standardized fit residual on the shown design.
        terminated: Whether the episode has ended.
    """

    turn: int
    turns_left: int
    structure: Structure
    tierA_gates: dict[str, bool]
    feedback: str
    identifiable_fraction: float
    observed_rmse: float
    terminated: bool = False


@dataclass
class _EpisodeRecord:
    """Internal trajectory record (for replay/debugging)."""

    actions: list[Action] = field(default_factory=list)
    structures: list[Structure] = field(default_factory=list)


class ModelEditEnv:
    """A deterministic multi-turn model-edit environment."""

    def __init__(
        self,
        reference: ModelSpec,
        observed_design: Design,
        hidden_battery: list[Design],
        backend: Backend,
        weights: RewardWeights,
        *,
        max_turns: int = 8,
        initial_structure: Structure = "one_compartment",
        sigma_floor: float = 1e-9,
    ) -> None:
        """Initialize the environment with the (hidden) reference and batteries."""
        self.reference = reference
        self.observed_design = observed_design
        self.hidden_battery = hidden_battery
        self.backend = backend
        self.weights = weights
        self.max_turns = max_turns
        self.initial_structure = initial_structure
        self.sigma_floor = sigma_floor
        # Observed data the agent fits to (deterministic; reference predictions).
        self._observed_data = backend.simulate(reference, observed_design).observed
        self._state = ModelState.initial(initial_structure)
        self._turn = 0
        self._terminated = False
        self.record = _EpisodeRecord()

    @property
    def state(self) -> ModelState:
        """The current (full) model state — for inspection, not the agent."""
        return self._state

    def reset(self) -> Observation:
        """Reset to the initial structure and return the first observation."""
        self._state = ModelState.initial(self.initial_structure)
        self._turn = 0
        self._terminated = False
        self.record = _EpisodeRecord()
        return self._observe()

    def _observed_rmse(self, candidate: ModelSpec) -> float:
        try:
            pred = self.backend.simulate(candidate, self.observed_design).observed
        except Exception:
            return float("inf")
        sigma = observation_sigma(
            self._observed_data,
            self.reference.observation.error,
            floor=self.sigma_floor,
        )
        resid = (pred - self._observed_data) / sigma
        return float(np.sqrt(np.mean(resid**2)))

    def observe_state(self, state: ModelState) -> Observation:
        """Return the (partial) observation for a hypothetical ``state``.

        Pure: does not mutate the episode. Used by inference-time edit search to
        preview the diagnostics of candidate edits without committing.
        """
        candidate = build_spec(state)
        report = verify(candidate, self.observed_design, self.backend)
        ident = 0.0
        if report.tierA_passed:
            ident = identifiability(
                candidate,
                [self.observed_design],
                self.backend,
                sigma_floor=self.sigma_floor,
            ).score
        return Observation(
            turn=self._turn,
            turns_left=self.max_turns - self._turn,
            structure=state.structure,
            tierA_gates=report.tierA_gates,
            feedback=report.feedback,
            identifiable_fraction=ident,
            observed_rmse=self._observed_rmse(candidate),
            terminated=self._terminated,
        )

    def _observe(self) -> Observation:
        return self.observe_state(self._state)

    def _admissible_keys(self) -> tuple[str, ...]:
        """Canonical keys of enumerated structures indistinguishable from ref."""
        competitors = CompetitorSet()
        competitors.add_enumeration()
        keys: list[str] = []
        for member in competitors.members():
            try:
                verdict = distinguish(
                    self.reference,
                    member,
                    self.hidden_battery,
                    self.backend,
                    sigma_floor=self.sigma_floor,
                ).verdict
            except Exception:
                continue
            if verdict == "indistinguishable":
                keys.append(member.canonicalize().key)
        return tuple(keys)

    def _terminal_reward(self, action: Action) -> RewardBreakdown:
        candidate = build_spec(self._state)
        report = verify(
            candidate,
            self.observed_design,
            self.backend,
            hidden_battery=self.hidden_battery,
        )
        ctx = RewardContext(
            candidate=candidate,
            reference=self.reference,
            hidden_battery=tuple(self.hidden_battery),
            observed_designs=(self.observed_design,),
            action="abstain" if isinstance(action, Abstain) else "commit",
            proposed_design=action.proposed_design
            if isinstance(action, Abstain)
            else None,
            admissible_keys=self._admissible_keys(),
        )
        return compose_reward(
            report, ctx, self.weights, self.backend, sigma_floor=self.sigma_floor
        )

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, dict[str, object]]:
        """Apply ``action`` and return ``(observation, reward, done, info)``."""
        if self._terminated:
            raise RuntimeError("step() called on a terminated episode; call reset()")

        self.record.actions.append(action)
        info: dict[str, object] = {}

        if isinstance(action, Commit | Abstain):
            breakdown = self._terminal_reward(action)
            self._terminated = True
            info["reward_breakdown"] = breakdown
            info["committed_structure"] = self._state.structure
            obs = self._observe()
            return obs, breakdown.total, True, info

        # Structural / parameter edit.
        self._state = apply_edit(self._state, action)
        self.record.structures.append(self._state.structure)
        self._turn += 1
        done = self._turn >= self.max_turns
        if done:
            # Out of turns without committing: force a commit-scored terminal.
            breakdown = self._terminal_reward(Commit())
            self._terminated = True
            info["reward_breakdown"] = breakdown
            info["forced_commit"] = True
            return self._observe(), breakdown.total, True, info
        return self._observe(), 0.0, False, info
