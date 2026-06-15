"""SFT, rejection sampling, GRPO, and the verifier-derived process reward model."""

from drrl.rl.decision import Decision, parse_completion
from drrl.rl.evaluate import EvalReport, score_decisions
from drrl.rl.grpo import GRPOSettings, make_reward_funcs, weights_for_mode
from drrl.rl.reject import RejectionStats, Trajectory, rejection_sample, rollout
from drrl.rl.sft import ImitationPolicy, SFTExample, build_sft_dataset, fit_sft
from drrl.rl.tasks import TaskCase, TaskRegistry, build_registry, to_dataset_rows

__all__ = [
    "Decision",
    "EvalReport",
    "GRPOSettings",
    "ImitationPolicy",
    "RejectionStats",
    "SFTExample",
    "TaskCase",
    "TaskRegistry",
    "Trajectory",
    "build_registry",
    "build_sft_dataset",
    "fit_sft",
    "make_reward_funcs",
    "parse_completion",
    "rejection_sample",
    "rollout",
    "score_decisions",
    "to_dataset_rows",
    "weights_for_mode",
]
