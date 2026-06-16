"""Typed mechanistic-model schema (the core ``ModelSpec`` contract).

Design choices that make the science testable:

- ODE right-hand sides are stored as **sympy-parseable strings** (``ODETerm.expr``),
  not opaque callables. This keeps a :class:`ModelSpec` losslessly JSON
  round-trippable and enables the linear-system transfer-function fast path and
  symbolic dimensional analysis downstream.
- Parameters carry a coordinate (``linear``/``log``). The *natural-space* value
  is what enters the ODE (``exp(value)`` when ``coord == "log"``), so a model
  and its log-reparameterization produce identical predictions.
- The observation maps a state to the measured quantity, optionally dividing by
  a volume parameter (``divide_by``) so an amount compartment yields a
  concentration — this is what makes ``AUC = Dose/CL`` hold for the closed-form
  invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import sympy
from pydantic import BaseModel, ConfigDict, Field, model_validator

from drrl.spec.units import Unit

if TYPE_CHECKING:
    from drrl.spec.canonical import CanonicalForm

Coord = Literal["linear", "log"]
Route = Literal["iv_bolus", "iv_infusion", "oral"]
ErrorKind = Literal["additive", "proportional", "combined"]
Transform = Literal["none", "log"]


class _Frozen(BaseModel):
    """Base for frozen, hashable value objects."""

    model_config = ConfigDict(frozen=True)


class Compartment(_Frozen):
    """A model compartment (state variable).

    Attributes:
        name: Identifier used as the state symbol in ODE expressions.
        unit: Physical unit of the state (amount, e.g. ``mg``, or concentration,
            e.g. ``mg/L``).
        initial: Initial value of the state at ``t=0`` before any dose (e.g. a
            target baseline ``R0`` for TMDD, or a PD turnover baseline). Doses are
            added on top of this.
    """

    name: str
    unit: Unit
    initial: float = 0.0


class Parameter(_Frozen):
    """A model parameter.

    Attributes:
        name: Identifier used as a symbol in ODE expressions.
        value: Stored value in the parameter's ``coord`` coordinate.
        unit: Physical unit of the *natural-space* parameter.
        coord: ``"linear"`` (value is natural-space) or ``"log"`` (natural-space
            value is ``exp(value)``).
    """

    name: str
    value: float
    unit: Unit
    coord: Coord = "linear"

    @property
    def natural_value(self) -> float:
        """The parameter value in natural (model) space."""
        if self.coord == "log":
            return float(sympy.exp(self.value))
        return self.value


class Dose(_Frozen):
    """A dosing event.

    Attributes:
        compartment: Target compartment name.
        amount: Dose amount in ``unit``.
        unit: Physical unit of the dose amount.
        time: Event time (same time unit as the design).
        route: Administration route.
        duration: Infusion duration for ``iv_infusion`` (else ``None``).
    """

    compartment: str
    amount: float
    unit: Unit
    time: float = 0.0
    route: Route = "iv_bolus"
    duration: float | None = None


class ErrorModel(_Frozen):
    """Residual error model for the observation.

    Attributes:
        kind: ``additive`` (``y + a``), ``proportional`` (``y(1 + b)``), or
            ``combined``.
        sigma_add: Additive SD (used by ``additive``/``combined``).
        sigma_prop: Proportional SD (used by ``proportional``/``combined``).
    """

    kind: ErrorKind = "additive"
    sigma_add: float = 0.0
    sigma_prop: float = 0.0


class ObservationModel(_Frozen):
    """Maps a state to the measured quantity.

    Attributes:
        state: Observed compartment name.
        divide_by: Optional parameter name to divide the state by (e.g. a volume
            ``V`` mapping amount -> concentration). ``None`` observes the state.
        transform: Output transform applied to predictions/data.
        error: Residual error model.
    """

    state: str
    divide_by: str | None = None
    transform: Transform = "none"
    error: ErrorModel = Field(default_factory=ErrorModel)


class ODETerm(_Frozen):
    """One state's time derivative.

    Attributes:
        target: Compartment whose derivative this defines.
        expr: sympy-parseable RHS over compartment and parameter names.
    """

    target: str
    expr: str


class Design(_Frozen):
    """A dosing + sampling design.

    Attributes:
        doses: Dosing events.
        sample_times: Observation times (sorted, non-negative).
    """

    doses: tuple[Dose, ...]
    sample_times: tuple[float, ...]

    @model_validator(mode="after")
    def _check_times(self) -> Design:
        if any(t < 0 for t in self.sample_times):
            raise ValueError("sample_times must be non-negative")
        if list(self.sample_times) != sorted(self.sample_times):
            raise ValueError("sample_times must be sorted ascending")
        return self


class ModelSpec(_Frozen):
    """A complete mechanistic model: structure, parameters, and observation.

    Attributes:
        compartments: State variables.
        odes: One :class:`ODETerm` per compartment.
        parameters: Model parameters.
        observation: Observation model.
    """

    compartments: tuple[Compartment, ...]
    odes: tuple[ODETerm, ...]
    parameters: tuple[Parameter, ...]
    observation: ObservationModel

    @property
    def state_names(self) -> tuple[str, ...]:
        """Compartment names in declaration order."""
        return tuple(c.name for c in self.compartments)

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names in declaration order."""
        return tuple(p.name for p in self.parameters)

    @property
    def symbols(self) -> dict[str, sympy.Symbol]:
        """Map every state/parameter name to a positive sympy symbol.

        Parameters and states are declared positive (the PK convention), which
        lets structural sign analysis treat coefficients as positive.
        """
        names = list(self.state_names) + list(self.param_names)
        return {n: sympy.Symbol(n, positive=True) for n in names}

    @model_validator(mode="after")
    def _validate_consistency(self) -> ModelSpec:
        states = set(self.state_names)
        if len(states) != len(self.compartments):
            raise ValueError("duplicate compartment names")
        params = set(self.param_names)
        if len(params) != len(self.parameters):
            raise ValueError("duplicate parameter names")
        targets = [o.target for o in self.odes]
        if set(targets) != states or len(targets) != len(states):
            raise ValueError("each compartment must have exactly one ODE")
        if self.observation.state not in states:
            raise ValueError(
                f"observed state {self.observation.state!r} is not a compartment"
            )
        if (
            self.observation.divide_by is not None
            and self.observation.divide_by not in params
        ):
            raise ValueError(
                f"divide_by {self.observation.divide_by!r} is not a parameter"
            )
        # Validate ODE expressions parse against the declared symbols.
        allowed = set(self.symbols)
        for ode in self.odes:
            expr = sympy.sympify(ode.expr, locals=self.symbols)
            unknown = {s.name for s in expr.free_symbols} - allowed
            if unknown:
                raise ValueError(
                    f"ODE {ode.target!r} has unknown symbols {sorted(unknown)}"
                )
        return self

    def rhs_exprs(self) -> dict[str, sympy.Expr]:
        """Return ``{compartment -> sympy RHS expression}``."""
        syms = self.symbols
        return {o.target: sympy.sympify(o.expr, locals=syms) for o in self.odes}

    def canonicalize(self) -> CanonicalForm:
        """Return the reparameterization-invariant canonical form.

        See :mod:`drrl.spec.canonical`.
        """
        from drrl.spec.canonical import canonicalize

        return canonicalize(self)
