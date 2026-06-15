"""Aggregate the verifier layers into a single VerifierReport.

This is the contract the env/reward layers consume: run the Tier-A gates and
(when they pass) the scientific layers, and package them with actionable
feedback.
"""

from __future__ import annotations

from drrl.sim.backend import Backend
from drrl.spec.model import Design, ModelSpec
from drrl.verifier.execution import check_execution
from drrl.verifier.identify import identifiability
from drrl.verifier.plausibility import check_plausibility
from drrl.verifier.report import VerifierReport
from drrl.verifier.units import check_units


def tier_a_gates(spec: ModelSpec, design: Design, backend: Backend) -> dict[str, bool]:
    """Run the Tier-A gates and return their boolean verdicts.

    ``schema`` is True here because a valid :class:`ModelSpec` already parsed
    (schema validity is enforced upstream by :func:`drrl.verifier.check_schema`).
    """
    units_ok = check_units(spec).ok
    exec_report, _ = check_execution(spec, design, backend)
    plaus = check_plausibility(spec, design, backend)
    return {
        "schema": True,
        "units": units_ok,
        "execution": exec_report.ok,
        "mass_balance": plaus.mass_balance_ok,
        "nonneg": plaus.nonneg_ok,
    }


def _feedback(gates: dict[str, bool]) -> str:
    """Actionable, edit-guiding text from the failed gates."""
    failed = [name for name, ok in gates.items() if not ok]
    if not failed:
        return "All Tier-A gates passed."
    hints = {
        "units": "fix dimensional inconsistency in an ODE term",
        "execution": "model failed to integrate; check stiffness/divergence",
        "mass_balance": "net flux creates mass; balance internal transfers",
        "nonneg": "a state goes negative; check elimination/transfer signs",
        "schema": "model does not satisfy the schema",
    }
    return "; ".join(hints.get(f, f) for f in failed)


def verify(
    spec: ModelSpec,
    design: Design,
    backend: Backend,
    *,
    hidden_battery: list[Design] | None = None,
    sigma_floor: float = 1e-9,
) -> VerifierReport:
    """Run the full verifier and return an aggregated :class:`VerifierReport`.

    Args:
        spec: The model.
        design: The (observed) design for the Tier-A gates.
        backend: Simulator.
        hidden_battery: If given and Tier-A passes, identifiability is computed on
            this held-out battery.
        sigma_floor: Noise floor for identifiability.

    Returns:
        A :class:`VerifierReport`.
    """
    gates = tier_a_gates(spec, design, backend)
    passed = all(gates.values())
    ident = None
    if passed and hidden_battery:
        ident = identifiability(spec, hidden_battery, backend, sigma_floor=sigma_floor)
    return VerifierReport(
        tierA_gates=gates,
        identifiability=ident,
        feedback=_feedback(gates),
    )
