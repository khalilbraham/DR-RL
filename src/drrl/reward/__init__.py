"""Compose a VerifierReport into a distinguishability-relative RewardBreakdown."""

from drrl.reward.breakdown import RewardBreakdown, RewardError, safe01
from drrl.reward.compose import compose_reward
from drrl.reward.context import RewardContext, RewardWeights

__all__ = [
    "RewardBreakdown",
    "RewardContext",
    "RewardError",
    "RewardWeights",
    "compose_reward",
    "safe01",
]
