"""The reward value object: a transparent breakdown, NaN-safe by contract.

The total is a gated convex combination ``gate * sum_i w_i r_i`` with every
component squashed to ``[0, 1]``. The gate is the product of the Tier-A booleans
(a single failed gate zeroes the whole reward). :meth:`RewardBreakdown.assert_no_nan`
is called before any value reaches the optimizer.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict


class RewardError(ValueError):
    """Raised when a reward value is non-finite (NaN/inf reaching the optimizer)."""


class RewardBreakdown(BaseModel):
    """Per-component reward with the gated total.

    Attributes:
        gate: Product of Tier-A booleans (0.0 or 1.0).
        r_eq: Equivalence-class credit (indistinguishable from the reference).
        r_fit: Fit on the hidden design battery.
        r_pk: PK-metric agreement.
        r_pd: PD-metric agreement.
        r_identify: Parameterization-invariant identifiable fraction.
        r_parsimony: Parsimony tie-break (gated; only within fit-adequate models).
        r_abstain: Calibrated abstention reward (asymmetric).
        r_expl: Explanation/feedback adherence.
        total: ``gate * sum_i w_i r_i`` in ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True)

    gate: float
    r_eq: float
    r_fit: float
    r_pk: float
    r_pd: float
    r_identify: float
    r_parsimony: float
    r_abstain: float
    r_expl: float
    total: float

    def assert_no_nan(self) -> None:
        """Raise :class:`RewardError` if any field is NaN or infinite."""
        for name, value in self.__dict__.items():
            if not math.isfinite(value):
                raise RewardError(f"reward component {name!r} is non-finite: {value}")

    def components(self) -> dict[str, float]:
        """Return the weighted component scores (excluding gate/total)."""
        return {
            "r_eq": self.r_eq,
            "r_fit": self.r_fit,
            "r_pk": self.r_pk,
            "r_pd": self.r_pd,
            "r_identify": self.r_identify,
            "r_parsimony": self.r_parsimony,
            "r_abstain": self.r_abstain,
            "r_expl": self.r_expl,
        }


def safe01(x: float) -> float:
    """Clamp ``x`` to ``[0, 1]``; map NaN/inf to 0.0 (never poisons the optimizer)."""
    if not math.isfinite(x):
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
