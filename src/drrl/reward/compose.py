"""Compose a VerifierReport (+ context) into a distinguishability-relative reward.

Principles enforced here (and tested adversarially):

- **Tier-A is a multiplicative gate**: any failed gate zeroes the total.
- **Distinguishability-relative**: a candidate observationally indistinguishable
  from the reference earns full equivalence credit, regardless of its structure.
- **Fit on the hidden battery**: ``r_fit`` is scored on designs the agent never
  observed, so it cannot be gamed by overfitting the shown design.
- **Parsimony is gated**: it only tie-breaks among fit-adequate models and so can
  never select an under-fitting / empty model.
- **Abstention is asymmetric**: a confident wrong commit is punished harder than
  over-abstaining.
- **NaN-safe**: every component is squashed to ``[0, 1]``; non-finite values map
  to 0 and the result is asserted finite before return.
"""

from __future__ import annotations

import math

from drrl.reward.breakdown import RewardBreakdown, safe01
from drrl.reward.context import RewardContext, RewardWeights
from drrl.sim.backend import Backend
from drrl.verifier.distinguish import distinguish, predictive_noncentrality
from drrl.verifier.identify import identifiability
from drrl.verifier.pkpd import pkpd_metrics
from drrl.verifier.report import VerifierReport


def _fit_score(lam: float, n_points: int) -> float:
    """Map noise-normalized misfit to ``[0, 1]`` (1 = perfect)."""
    if n_points == 0:
        return 0.0
    return safe01(math.exp(-0.5 * lam / n_points))


def _parsimony(
    candidate_params: int, reference_params: int, penalty_per_param: float
) -> float:
    """Parsimony: 1 if no more complex than the reference, decaying with excess."""
    excess = max(0, candidate_params - reference_params)
    return safe01(1.0 - penalty_per_param * excess)


def _abstain_score(
    ctx: RewardContext, indistinguishable: bool, weights: RewardWeights
) -> float:
    """Calibrated, asymmetric abstention reward."""
    ambiguous = len(ctx.admissible_keys) >= weights.ambiguous_min_admissible
    if ctx.action == "commit":
        if indistinguishable:
            return 1.0  # correct, confident commit
        return safe01(1.0 - weights.wrong_commit_penalty)  # wrong commit: hard penalty
    # abstain
    if ambiguous:
        return 1.0 if ctx.proposed_design is not None else 0.6
    return safe01(1.0 - weights.over_abstain_penalty)  # over-abstaining: mild penalty


def compose_reward(
    report: VerifierReport,
    ctx: RewardContext,
    weights: RewardWeights,
    backend: Backend,
    *,
    sigma_floor: float = 1e-9,
    alpha: float = 0.05,
) -> RewardBreakdown:
    """Compose the distinguishability-relative :class:`RewardBreakdown`.

    Args:
        report: The verifier report (its Tier-A gates drive the multiplicative gate).
        ctx: Scoring context (candidate, reference, hidden battery, action, ...).
        weights: Reward weights and thresholds.
        backend: Simulator.
        sigma_floor: Noise floor for the discrepancy metrics.
        alpha: Nominal FPR for the distinguishability verdict.

    Returns:
        A NaN-safe :class:`RewardBreakdown`.
    """
    gate = 1.0 if report.tierA_passed else 0.0
    if gate == 0.0:
        zero = RewardBreakdown(
            gate=0.0,
            r_eq=0.0,
            r_fit=0.0,
            r_pk=0.0,
            r_pd=0.0,
            r_identify=0.0,
            r_parsimony=0.0,
            r_abstain=0.0,
            r_expl=0.0,
            total=0.0,
        )
        zero.assert_no_nan()
        return zero

    battery = list(ctx.hidden_battery)

    # Discrepancy candidate-vs-reference on the HIDDEN battery.
    lam, n_points = predictive_noncentrality(
        ctx.reference, ctx.candidate, battery, backend, sigma_floor=sigma_floor
    )
    verdict = distinguish(
        ctx.reference,
        ctx.candidate,
        battery,
        backend,
        alpha=alpha,
        sigma_floor=sigma_floor,
    ).verdict
    indistinguishable = verdict == "indistinguishable"

    r_fit = _fit_score(lam, n_points)
    r_eq = 1.0 if indistinguishable else r_fit

    # PK-metric agreement on a representative design (AUC over the window).
    design = battery[0]
    m_ref = pkpd_metrics(ctx.reference, design, backend).metrics
    m_cand = pkpd_metrics(ctx.candidate, design, backend).metrics
    auc_ref = m_ref.get("auc", 0.0)
    auc_cand = m_cand.get("auc", 0.0)
    rel = abs(auc_cand - auc_ref) / (abs(auc_ref) + 1e-12)
    r_pk = safe01(math.exp(-rel))
    r_pd = r_fit  # no separate PD observable in the MVE library (documented)

    # Identifiability (prediction-based, on the hidden battery).
    if report.identifiability is not None:
        r_identify = safe01(report.identifiability.score)
    else:
        r_identify = safe01(
            identifiability(
                ctx.candidate, battery, backend, sigma_floor=sigma_floor
            ).score
        )

    # Parsimony — gated by fit adequacy so it can never rescue an under-fit model.
    parsimony_raw = _parsimony(
        len(ctx.candidate.parameters),
        len(ctx.reference.parameters),
        weights.parsimony_penalty_per_param,
    )
    r_parsimony = parsimony_raw if r_fit >= weights.fit_adequate_tau else 0.0

    r_abstain = _abstain_score(ctx, indistinguishable, weights)
    r_expl = 1.0  # neutral placeholder (feedback adherence; weighted low)

    total = gate * (
        weights.w_eq * r_eq
        + weights.w_fit * r_fit
        + weights.w_pk * r_pk
        + weights.w_pd * r_pd
        + weights.w_identify * r_identify
        + weights.w_parsimony * r_parsimony
        + weights.w_abstain * r_abstain
        + weights.w_expl * r_expl
    )

    breakdown = RewardBreakdown(
        gate=gate,
        r_eq=safe01(r_eq),
        r_fit=safe01(r_fit),
        r_pk=safe01(r_pk),
        r_pd=safe01(r_pd),
        r_identify=safe01(r_identify),
        r_parsimony=safe01(r_parsimony),
        r_abstain=safe01(r_abstain),
        r_expl=safe01(r_expl),
        total=safe01(total),
    )
    breakdown.assert_no_nan()
    return breakdown
