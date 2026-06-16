"""Process reward model: verifier-derived step labels (no human labels).

The PRM scores *observations* (the agent's own diagnostics — Tier-A gates,
identifiability, fit), so it stays inside the partial-observability contract: it
uses no oracle/reference labels. The per-step reward is the change in a
verifier-derived potential, which (per the brief) is used as dense shaping during
RL and to guide best-first edit search at inference.

We do **not** claim potential-based policy invariance for this PRM; instead the
experiment in ``experiments/prm_shift.py`` measures whether PRM-guided search
shifts the terminal-reward distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from drrl.env.environment import Observation


@dataclass(frozen=True)
class PRMWeights:
    """Weights for the verifier-derived potential.

    Attributes:
        w_gate: Reward for passing all Tier-A gates.
        w_identify: Reward for prediction-affecting identifiable fraction.
        w_fit: Reward for fit quality (``exp(-standardized rmse)``).
    """

    w_gate: float = 0.4
    w_identify: float = 0.3
    w_fit: float = 0.3


def observation_potential(obs: Observation, weights: PRMWeights | None = None) -> float:
    """Verifier-derived potential ``Phi(obs)`` in ``[0, 1]``.

    A model that fails Tier-A has potential 0 (the gate is multiplicative);
    otherwise potential rewards identifiability and fit. Uses only fields the
    agent observes — no oracle labels.
    """
    w = weights or PRMWeights()
    if not (obs.tierA_gates and all(obs.tierA_gates.values())):
        return 0.0
    fit = math.exp(-obs.observed_rmse) if math.isfinite(obs.observed_rmse) else 0.0
    raw = w.w_gate + w.w_identify * obs.identifiable_fraction + w.w_fit * fit
    total = w.w_gate + w.w_identify + w.w_fit
    return raw / total if total > 0 else 0.0


class ProcessRewardModel:
    """Verifier-derived process reward: dense step labels from potential changes."""

    def __init__(self, weights: PRMWeights | None = None) -> None:
        """Initialize with potential weights."""
        self.weights = weights or PRMWeights()

    def potential(self, obs: Observation) -> float:
        """Potential of an observation."""
        return observation_potential(obs, self.weights)

    def step_reward(self, before: Observation, after: Observation) -> float:
        """Dense step reward = ``Phi(after) - Phi(before)``.

        Positive when an edit improves the verifier-derived potential (fixes a
        Tier-A violation, raises identifiability, or improves fit).
        """
        return self.potential(after) - self.potential(before)
