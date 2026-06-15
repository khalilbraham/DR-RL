"""Tier-A gate: dimensional analysis of every ODE term (invariant #4).

Each term of ``d(state)/dt`` must carry dimension ``[state] / [time]``. We walk
the sympy expression tree, assigning each symbol its declared pint dimension,
and check coherence — catching seeded unit errors (rate vs clearance, amount vs
concentration, mass vs amount-of-substance) with full recall.
"""

from __future__ import annotations

from typing import Any

import sympy

from drrl.spec.model import ModelSpec
from drrl.spec.units import UREG
from drrl.verifier.report import UnitsReport

_DIMENSIONLESS = UREG.Quantity(1.0, "dimensionless").dimensionality
_TIME = UREG.Quantity(1.0, "second").dimensionality


class DimensionalError(ValueError):
    """Raised internally when a term is dimensionally incoherent."""


def _quantity(unit_expr: str) -> Any:
    """A unit-bearing quantity of magnitude 1 for ``unit_expr``."""
    return UREG.Quantity(1.0, unit_expr)


def _dim(expr: sympy.Expr, env: dict[str, Any]) -> Any:
    """Return the pint quantity (magnitude 1) representing ``expr``'s dimension."""
    if expr.is_Number or expr.is_NumberSymbol:
        return _quantity("dimensionless")
    if expr.is_Symbol:
        q = env.get(str(expr))
        if q is None:
            raise DimensionalError(f"unknown symbol {expr}")
        return q
    if expr.is_Add:
        parts = [_dim(a, env) for a in expr.args]
        base = parts[0]
        for q in parts[1:]:
            if q.dimensionality != base.dimensionality:
                raise DimensionalError(
                    f"incompatible sum: {base.dimensionality} + {q.dimensionality}"
                )
        return base
    if expr.is_Mul:
        prod = _quantity("dimensionless")
        for a in expr.args:
            prod = prod * _dim(a, env)
        return prod
    if expr.is_Pow:
        base, exponent = expr.args
        if not exponent.is_number:
            raise DimensionalError("non-numeric exponent")
        return _dim(base, env) ** float(exponent)
    if expr.is_Function:
        for a in expr.args:
            if _dim(a, env).dimensionality != _DIMENSIONLESS:
                raise DimensionalError(f"argument of {expr.func} must be dimensionless")
        return _quantity("dimensionless")
    raise DimensionalError(f"unsupported expression node: {expr}")


def check_units(spec: ModelSpec) -> UnitsReport:
    """Dimensionally check every ODE term.

    Args:
        spec: The model.

    Returns:
        A :class:`UnitsReport`; ``ok`` iff every term has dimension
        ``[target_state] / [time]``.
    """
    env: dict[str, Any] = {c.name: _quantity(c.unit.expr) for c in spec.compartments}
    env.update({p.name: _quantity(p.unit.expr) for p in spec.parameters})
    unit_of = {c.name: c.unit for c in spec.compartments}
    syms = spec.symbols

    violations: list[str] = []
    for ode in spec.odes:
        required = (
            _quantity(unit_of[ode.target].expr) / _quantity("second")
        ).dimensionality
        expr = sympy.sympify(ode.expr, locals=syms)
        try:
            dim = _dim(expr, env)
        except DimensionalError as exc:
            violations.append(f"{ode.target}: {exc}")
            continue
        if dim.dimensionality != required:
            violations.append(
                f"{ode.target}: dimension {dim.dimensionality}, expected {required}"
            )
    return UnitsReport(ok=not violations, violations=tuple(violations))
