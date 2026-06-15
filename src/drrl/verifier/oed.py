"""Optimal experiment design: utility of a proposed design.

Two uses:

- **Parameter precision** (``d_optimal``): ``log det`` of the Fisher information
  matrix — how well a design pins down parameters.
- **Model discrimination** (``eig`` proxy): the worst-case predictive
  noncentrality between the reference and the admissible competitors under the
  design. A design *resolves the ambiguity* when even the least-separated
  competitor becomes distinguishable. This is what verifies an abstention
  proposal: the agent should propose a design that would separate the admissible
  set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from drrl.sim.backend import Backend
from drrl.spec.model import Design, ModelSpec
from drrl.verifier.distinguish import predictive_noncentrality, predictive_threshold
from drrl.verifier.identify import sensitivity_matrix
from drrl.verifier.report import OEDReport

FloatArray = NDArray[np.float64]


def fisher_information(
    spec: ModelSpec, design: Design, backend: Backend, *, sigma_floor: float = 1e-9
) -> FloatArray:
    """Fisher information matrix ``S^T S`` (noise-normalized) for a design."""
    s_matrix = sensitivity_matrix(spec, [design], backend, sigma_floor=sigma_floor)
    return s_matrix.T @ s_matrix


def d_optimal_value(
    spec: ModelSpec, design: Design, backend: Backend, *, sigma_floor: float = 1e-9
) -> float:
    """``log det`` of the FIM (D-optimality). ``-inf`` if singular."""
    fim = fisher_information(spec, design, backend, sigma_floor=sigma_floor)
    sign, logdet = np.linalg.slogdet(fim)
    return float(logdet) if sign > 0 else float("-inf")


def discrimination_utility(
    ref: ModelSpec,
    competitors: list[ModelSpec],
    design: Design,
    backend: Backend,
    *,
    sigma_floor: float = 1e-9,
) -> float:
    """Worst-case predictive separation between ``ref`` and the competitors.

    Returns the *minimum* noncentrality over competitors (the hardest pair to
    tell apart). ``0.0`` if there are no competitors.
    """
    if not competitors:
        return 0.0
    lams = [
        predictive_noncentrality(ref, c, [design], backend, sigma_floor=sigma_floor)[0]
        for c in competitors
    ]
    return float(min(lams))


def score_design(
    ref: ModelSpec,
    design: Design,
    backend: Backend,
    *,
    competitors: list[ModelSpec] | None = None,
    criterion: str = "eig",
    alpha: float = 0.05,
    sigma_floor: float = 1e-9,
) -> OEDReport:
    """Score a proposed design.

    Args:
        ref: Reference model.
        design: The proposed design to score.
        backend: Simulator.
        competitors: Admissible competitors (required for the ``eig`` criterion).
        criterion: ``"eig"`` (model discrimination) or ``"d_optimal"`` (parameter
            precision).
        alpha: Nominal FPR used to decide ``resolves_ambiguity``.
        sigma_floor: Noise floor.

    Returns:
        An :class:`OEDReport`.
    """
    if criterion == "d_optimal":
        util = d_optimal_value(ref, design, backend, sigma_floor=sigma_floor)
        return OEDReport(criterion="d_optimal", utility=util, resolves_ambiguity=False)

    comps = competitors or []
    util = discrimination_utility(ref, comps, design, backend, sigma_floor=sigma_floor)
    n_points = len(design.sample_times)
    resolves = bool(util > predictive_threshold(n_points, alpha=alpha))
    return OEDReport(criterion="eig", utility=util, resolves_ambiguity=resolves)
