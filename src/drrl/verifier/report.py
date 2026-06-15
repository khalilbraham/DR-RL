"""Structured verifier reports — the contract every check produces.

Each verifier layer returns a small, serializable report. They aggregate into a
:class:`VerifierReport`. Tier-A checks are *gates* (booleans); the scientific
layers (distinguishability, identifiability) carry richer structure. No NumPy
arrays cross this boundary — everything is plain floats/lists so reports are
JSON-serializable and comparable across runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    """Base for frozen, serializable reports."""

    model_config = ConfigDict(frozen=True)


class SchemaReport(_Frozen):
    """Grammar/schema validity.

    Attributes:
        ok: Whether the candidate parsed into a valid ``ModelSpec``.
        errors: Human-readable validation errors (empty when ``ok``).
    """

    ok: bool
    errors: tuple[str, ...] = ()


class UnitsReport(_Frozen):
    """Dimensional-analysis result.

    Attributes:
        ok: Whether every ODE term is dimensionally coherent.
        violations: One message per offending term (empty when ``ok``).
    """

    ok: bool
    violations: tuple[str, ...] = ()


class ExecutionReport(_Frozen):
    """Codegen + integration outcome (failures captured as data, not raised).

    Attributes:
        ok: Whether the model compiled and integrated without error.
        integrator_ok: Whether the integrator reported success at every step.
        message: Diagnostic detail (empty when ``ok``).
    """

    ok: bool
    integrator_ok: bool
    message: str = ""


class PlausibilityReport(_Frozen):
    """Mass-balance and non-negativity probes.

    Attributes:
        mass_balance_ok: Closed-system mass accounting closes to tolerance.
        nonneg_ok: All states stayed >= -tolerance over the trajectory.
        details: Diagnostic messages.
    """

    mass_balance_ok: bool
    nonneg_ok: bool
    details: tuple[str, ...] = ()


class PkPdReport(_Frozen):
    """Derived PK/PD metrics.

    Attributes:
        metrics: Named scalar metrics (e.g. ``auc``, ``cmax``, ``tmax``,
            ``thalf``, ``cl``). Absent metrics are simply omitted.
    """

    metrics: dict[str, float] = {}


class IdentifyReport(_Frozen):
    """Parameterization-invariant, prediction-based identifiability.

    Attributes:
        param_names: Parameter names (column order of the analysis).
        n_params: Number of parameters.
        identifiable_rank: Rank of the noise-normalized prediction sensitivity
            over the held-out design battery (number of prediction-affecting
            independent directions).
        identifiable_fraction: ``identifiable_rank / n_params`` — the
            reparameterization-invariant identifiability score.
        score: Alias of ``identifiable_fraction`` (the reward-facing scalar).
        prediction_affecting: Per parameter, whether it individually affects
            predictions (nonzero sensitivity column).
        nonidentifiable_directions: Basis of the null space (each a unit vector
            in parameter space) — the sloppy/flat directions.
    """

    param_names: tuple[str, ...]
    n_params: int
    identifiable_rank: int
    identifiable_fraction: float
    score: float
    prediction_affecting: dict[str, bool]
    nonidentifiable_directions: tuple[tuple[float, ...], ...] = ()


class DistinguishReport(_Frozen):
    """Calibrated predictive distinguishability verdict.

    Attributes:
        verdict: ``"indistinguishable"`` or ``"distinguishable"`` vs the
            reference at the operative noise level / design.
        method: ``"transfer_function"`` (exact linear fast path) or
            ``"predictive"`` (general oracle).
        statistic: The test statistic (noncentrality for the predictive path;
            ``0.0`` for an exact transfer-function match).
        threshold: Decision threshold the statistic was compared against.
        admissible: Canonical keys of structures judged indistinguishable from
            the reference (the equivalence class / admissible set).
    """

    verdict: Literal["indistinguishable", "distinguishable"]
    method: Literal["transfer_function", "predictive"]
    statistic: float
    threshold: float
    admissible: tuple[str, ...] = ()


class OEDReport(_Frozen):
    """Optimal-experiment-design utility for a proposed design.

    Attributes:
        criterion: ``"d_optimal"`` or ``"eig"``.
        utility: Scalar design utility (higher is more informative).
        resolves_ambiguity: Whether the design is expected to separate the
            admissible set (used to verify abstention proposals).
    """

    criterion: Literal["d_optimal", "eig"]
    utility: float
    resolves_ambiguity: bool = False


class VerifierReport(_Frozen):
    """Aggregate report consumed by the reward layer and the agent.

    Attributes:
        tierA_gates: Boolean gates (``schema``, ``units``, ``execution``,
            ``mass_balance``, ``nonneg``). Their product is the reward gate.
        distinguishability: Distinguishability verdict, if computed.
        identifiability: Identifiability analysis, if computed.
        pkpd: PK/PD metrics, if computed.
        plausibility: Plausibility probes, if computed.
        oed: Experiment-design utility, if computed.
        feedback: Actionable, edit-guiding text for the agent.
    """

    tierA_gates: dict[str, bool]
    distinguishability: DistinguishReport | None = None
    identifiability: IdentifyReport | None = None
    pkpd: PkPdReport | None = None
    plausibility: PlausibilityReport | None = None
    oed: OEDReport | None = None
    feedback: str = ""

    @property
    def tierA_passed(self) -> bool:
        """Whether every Tier-A gate passed."""
        return bool(self.tierA_gates) and all(self.tierA_gates.values())
