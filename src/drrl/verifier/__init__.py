"""Layered scientific verifier; each check returns a structured report."""

from drrl.verifier.execution import check_execution
from drrl.verifier.identify import (
    identifiability,
    prediction_change_along,
    sensitivity_matrix,
)
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
    "DistinguishReport",
    "ExecutionReport",
    "IdentifyReport",
    "OEDReport",
    "PkPdReport",
    "PlausibilityReport",
    "SchemaReport",
    "UnitsReport",
    "VerifierReport",
    "check_execution",
    "check_plausibility",
    "check_schema",
    "check_units",
    "compute_pk_metrics",
    "hill",
    "identifiability",
    "pkpd_metrics",
    "prediction_change_along",
    "sensitivity_matrix",
]
