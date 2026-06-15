"""Rejection-sampling bootstrap (STaR-style).

Roll out a policy in the environment, keep only trajectories whose terminal
reward clears a threshold, and return them as supervision for SFT. With a stub
policy this exercises the full env -> reward loop on CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from drrl.env.actions import Action
from drrl.env.environment import ModelEditEnv, Observation
from drrl.policy.base import Policy


@dataclass(frozen=True)
class Trajectory:
    """A single rolled-out episode.

    Attributes:
        steps: ``(observation, action)`` pairs in order.
        reward: Terminal reward.
        committed_structure: The structure committed/forced at termination.
    """

    steps: tuple[tuple[Observation, Action], ...]
    reward: float
    committed_structure: str


def rollout(env: ModelEditEnv, policy: Policy) -> Trajectory:
    """Run one episode to termination and return its trajectory."""
    obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    steps: list[tuple[Observation, Action]] = []
    reward = 0.0
    committed: str = env.state.structure
    done = False
    while not done:
        action = policy.act(obs)
        steps.append((obs, action))
        obs, reward, done, info = env.step(action)
        if done:
            committed = str(info.get("committed_structure", env.state.structure))
    return Trajectory(steps=tuple(steps), reward=reward, committed_structure=committed)


@dataclass
class RejectionStats:
    """Summary of a rejection-sampling run."""

    n_sampled: int = 0
    n_accepted: int = 0
    accepted: list[Trajectory] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of sampled trajectories accepted."""
        return self.n_accepted / self.n_sampled if self.n_sampled else 0.0


def rejection_sample(
    env_factory: Callable[[], ModelEditEnv],
    policy_factory: Callable[[int], Policy],
    *,
    n_samples: int,
    reward_threshold: float,
) -> RejectionStats:
    """Sample trajectories and keep those above ``reward_threshold``.

    Args:
        env_factory: Builds a fresh environment per sample.
        policy_factory: Builds a policy given the sample index (for seed variety).
        n_samples: Number of trajectories to sample.
        reward_threshold: Minimum terminal reward to accept.

    Returns:
        A :class:`RejectionStats` with the accepted trajectories.
    """
    stats = RejectionStats()
    for i in range(n_samples):
        traj = rollout(env_factory(), policy_factory(i))
        stats.n_sampled += 1
        if traj.reward >= reward_threshold:
            stats.n_accepted += 1
            stats.accepted.append(traj)
    return stats
