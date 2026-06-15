"""Tier-A gates: mass balance and non-negativity (invariant #5).

Two physical sanity checks:

- **Non-negativity** (numeric): states must stay >= 0 (within tolerance) over a
  fine trajectory. A model that drives an amount negative is unphysical.
- **Mass balance** (structural, reparameterization-invariant): the net flux
  ``sum_X dX/dt`` may only *remove* mass (elimination) or conserve it. Any term
  that *creates* mass — a state-proportional source, or a spontaneous constant
  source in the ODEs — is a modeling error (e.g. an internal transfer added to
  the receiver but not subtracted from the donor). Reading the sign off the
  symbolic net flux is independent of parameterization.
"""

from __future__ import annotations

import numpy as np
import sympy

from drrl.sim.backend import Backend
from drrl.spec.model import Design, ModelSpec
from drrl.verifier.report import PlausibilityReport


def _term_sign(term: sympy.Expr) -> int:
    """Sign of a monomial assuming all symbols positive (PK convention)."""
    val = sympy.simplify(term.subs({s: sympy.Integer(1) for s in term.free_symbols}))
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def _mass_balance(spec: ModelSpec) -> tuple[bool, list[str]]:
    """Structural check: net flux must not create mass."""
    rhs = spec.rhs_exprs()
    syms = spec.symbols
    state_syms = {syms[n] for n in spec.state_names}
    net = sympy.expand(sum(rhs.values(), start=sympy.Integer(0)))
    issues: list[str] = []
    for term in sympy.Add.make_args(net):
        states_in = term.free_symbols & state_syms
        sign = _term_sign(term)
        if states_in and sign > 0:
            issues.append(f"mass-creating term in net flux: +{term}")
        elif not states_in and sign > 0:
            # A negative constant is zero-order elimination (allowed); a
            # positive constant is a spontaneous mass source.
            issues.append(f"spontaneous source term in net flux: {term}")
    return (not issues), issues


def check_plausibility(
    spec: ModelSpec,
    design: Design,
    backend: Backend,
    *,
    atol: float = 1e-8,
    n_grid: int = 400,
) -> PlausibilityReport:
    """Check non-negativity and mass balance.

    Args:
        spec: The model.
        design: Dosing + sampling design (its doses define initial conditions).
        backend: Simulator used for the fine non-negativity trajectory.
        atol: Absolute tolerance for the non-negativity floor.
        n_grid: Number of fine time points for the non-negativity trajectory.

    Returns:
        A :class:`PlausibilityReport`.
    """
    mass_ok, issues = _mass_balance(spec)

    t_end = max(design.sample_times) if design.sample_times else 1.0
    fine_times = tuple(np.linspace(0.0, float(t_end), n_grid))
    fine_design = Design(doses=design.doses, sample_times=fine_times)

    details = list(issues)
    try:
        result = backend.simulate(spec, fine_design)
        min_state = float(np.min(result.states))
        nonneg_ok = bool(min_state >= -atol)
        if not nonneg_ok:
            details.append(f"state reached {min_state:.3e} (< -{atol:.0e})")
    except Exception as exc:
        nonneg_ok = False
        details.append(f"non-negativity sim failed: {type(exc).__name__}: {exc}")

    return PlausibilityReport(
        mass_balance_ok=mass_ok, nonneg_ok=nonneg_ok, details=tuple(details)
    )
