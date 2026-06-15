"""Inputs to reward composition: weights (from config) and the scoring context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from omegaconf import DictConfig

from drrl.spec.model import Design, ModelSpec

Action = Literal["commit", "abstain"]


@dataclass(frozen=True)
class RewardWeights:
    """Reward weights and gating thresholds (populated from ``configs/reward``).

    Weights are a convex combination (sum to 1). The abstention penalties are
    deliberately asymmetric: a confident wrong commit is punished harder than
    over-abstaining.
    """

    w_eq: float = 0.25
    w_fit: float = 0.20
    w_pk: float = 0.05
    w_pd: float = 0.05
    w_identify: float = 0.20
    w_parsimony: float = 0.05
    w_abstain: float = 0.15
    w_expl: float = 0.05

    # Abstention asymmetry.
    wrong_commit_penalty: float = 1.0
    over_abstain_penalty: float = 0.3

    # Gating / shaping thresholds.
    fit_adequate_tau: float = 0.5  # min r_fit for parsimony to count
    ambiguous_min_admissible: int = 2  # admissible-set size that makes abstention right
    parsimony_penalty_per_param: float = 0.1
    identify_rank_rtol: float = 1e-2  # practical-identifiability cutoff for r_identify

    @classmethod
    def from_config(cls, cfg: DictConfig) -> RewardWeights:
        """Build weights from a ``configs/reward`` config."""
        w = cfg.weights
        ab = cfg.abstain
        return cls(
            w_eq=float(w.r_eq),
            w_fit=float(w.r_fit),
            w_pk=float(w.r_pk),
            w_pd=float(w.r_pd),
            w_identify=float(w.r_identify),
            w_parsimony=float(w.r_parsimony),
            w_abstain=float(w.r_abstain),
            w_expl=float(w.r_expl),
            wrong_commit_penalty=float(ab.wrong_commit_penalty),
            over_abstain_penalty=float(ab.over_abstain_penalty),
            fit_adequate_tau=float(cfg.get("fit_adequate_tau", 0.5)),
            ambiguous_min_admissible=int(cfg.get("ambiguous_min_admissible", 2)),
            parsimony_penalty_per_param=float(
                cfg.get("parsimony_penalty_per_param", 0.1)
            ),
            identify_rank_rtol=float(cfg.get("identify_rank_rtol", 1e-2)),
        )


@dataclass(frozen=True)
class RewardContext:
    """Everything the reward needs beyond the verifier report.

    Attributes:
        candidate: The proposed model.
        reference: The ground-truth reference model (data-generating).
        hidden_battery: Designs used to score fit/identifiability — the agent
            never observed these (enforced by callers; see the fit test).
        observed_designs: Designs the agent *did* see (for the disjointness check).
        action: ``"commit"`` (commit to ``candidate``) or ``"abstain"``.
        proposed_design: For abstention, the experiment the agent proposes.
        admissible_keys: Canonical keys of the admissible set (its size drives
            whether abstention is the right call).
    """

    candidate: ModelSpec
    reference: ModelSpec
    hidden_battery: tuple[Design, ...]
    observed_designs: tuple[Design, ...] = ()
    action: Action = "commit"
    proposed_design: Design | None = None
    admissible_keys: tuple[str, ...] = field(default_factory=tuple)
