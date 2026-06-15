"""Layered scientific verifier; each check returns a structured report."""

from drrl.verifier.competitors import (
    CompetitorSet,
    admissible_set,
    enumerate_structures,
)
from drrl.verifier.distinguish import (
    distinguish,
    is_linear,
    linear_indistinguishable,
    markov_parameters,
    predictive_distinguishable,
    predictive_noncentrality,
)
from drrl.verifier.execution import check_execution
from drrl.verifier.identify import (
    identifiability,
    prediction_change_along,
    sensitivity_matrix,
)
from drrl.verifier.oed import (
    d_optimal_value,
    discrimination_utility,
    fisher_information,
    score_design,
)
from drrl.verifier.pipeline import tier_a_gates, verify
from drrl.verifier.pkpd import compute_pk_metrics, hill, pkpd_metrics
from drrl.verifier.plausibility import check_plausibility
from drrl.verifier.report import (
    DistinguishReport,
    ExecutionReport,
    IdentifyReport,
    OEDReport,
    PkPdReport,
    PlausibilityReport,
    SchemaReport,
    UnitsReport,
    VerifierReport,
)
from drrl.verifier.schema import check_schema
from drrl.verifier.units import check_units

__all__ = [
    "CompetitorSet",
    "DistinguishReport",
    "ExecutionReport",
    "IdentifyReport",
    "OEDReport",
    "PkPdReport",
    "PlausibilityReport",
    "SchemaReport",
    "UnitsReport",
    "VerifierReport",
    "admissible_set",
    "check_execution",
    "check_plausibility",
    "check_schema",
    "check_units",
    "compute_pk_metrics",
    "d_optimal_value",
    "discrimination_utility",
    "distinguish",
    "enumerate_structures",
    "fisher_information",
    "hill",
    "identifiability",
    "is_linear",
    "linear_indistinguishable",
    "markov_parameters",
    "pkpd_metrics",
    "prediction_change_along",
    "predictive_distinguishable",
    "predictive_noncentrality",
    "score_design",
    "sensitivity_matrix",
    "tier_a_gates",
    "verify",
]
