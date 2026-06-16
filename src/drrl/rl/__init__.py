"""SFT, rejection sampling, GRPO, and the verifier-derived process reward model."""

from drrl.rl.decision import Decision, parse_completion
from drrl.rl.edit_search import SearchResult, best_first_repair, one_shot_commit
from drrl.rl.evaluate import EvalReport, flat_commit_rate, score_decisions
from drrl.rl.grpo import GRPOSettings, make_reward_funcs, weights_for_mode
from drrl.rl.prm import PRMWeights, ProcessRewardModel, observation_potential
from drrl.rl.reject import RejectionStats, Trajectory, rejection_sample, rollout
from drrl.rl.sft import ImitationPolicy, SFTExample, build_sft_dataset, fit_sft
from drrl.rl.tasks import TaskCase, TaskRegistry, build_registry, to_dataset_rows

__all__ = [
    "Decision",
    "EvalReport",
    "GRPOSettings",
    "ImitationPolicy",
    "PRMWeights",
    "ProcessRewardModel",
    "RejectionStats",
    "SFTExample",
    "SearchResult",
    "TaskCase",
    "TaskRegistry",
    "Trajectory",
    "best_first_repair",
    "build_registry",
    "build_sft_dataset",
    "fit_sft",
    "flat_commit_rate",
    "make_reward_funcs",
    "observation_potential",
    "one_shot_commit",
    "parse_completion",
    "rejection_sample",
    "rollout",
    "score_decisions",
    "to_dataset_rows",
    "weights_for_mode",
]
