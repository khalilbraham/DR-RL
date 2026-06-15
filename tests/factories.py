"""Model builders shared across Phase 1 tests (canonical PK structures)."""

from __future__ import annotations

import math

from drrl.spec import (
    Compartment,
    Design,
    Dose,
    ErrorModel,
    ModelSpec,
    ObservationModel,
    ODETerm,
    Parameter,
    Unit,
)

# A realistic proportional error model so noise-normalized analyses (e.g.
# identifiability) use a meaningful SD rather than a degenerate floor.
PROP_ERR = ErrorModel(kind="proportional", sigma_prop=0.1)


def with_prop_error(spec: ModelSpec) -> ModelSpec:
    """Return ``spec`` with a proportional (10%) observation error model."""
    obs = spec.observation
    return spec.model_copy(
        update={
            "observation": ObservationModel(
                state=obs.state,
                divide_by=obs.divide_by,
                transform=obs.transform,
                error=PROP_ERR,
            )
        }
    )


def add_unused_param(spec: ModelSpec, name: str = "dummy") -> ModelSpec:
    """Return ``spec`` with one extra parameter that appears in no ODE.

    Predictions are unchanged (so the model is fit-equivalent) but it is strictly
    richer — used to test that parsimony tie-breaks toward the simpler model.
    """
    extra = Parameter(name=name, value=1.0, unit=Unit(expr="dimensionless"))
    return spec.model_copy(update={"parameters": (*spec.parameters, extra)})


MG = Unit(expr="mg")
L = Unit(expr="L")
LPH = Unit(expr="L/h")
PERH = Unit(expr="1/h")


def one_comp_macro(cl: float = 2.0, v: float = 10.0) -> ModelSpec:
    """1-compartment IV model in macro constants (CL, V)."""
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=MG),),
        odes=(ODETerm(target="A1", expr="-(CL/V)*A1"),),
        parameters=(
            Parameter(name="CL", value=cl, unit=LPH),
            Parameter(name="V", value=v, unit=L),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )


def one_comp_micro(ke: float = 0.2, v: float = 10.0) -> ModelSpec:
    """1-compartment IV model in micro rate (k10) — reparam of the macro form."""
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=MG),),
        odes=(ODETerm(target="A1", expr="-k10*A1"),),
        parameters=(
            Parameter(name="k10", value=ke, unit=PERH),
            Parameter(name="V", value=v, unit=L),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )


def one_comp_log(ke: float = 0.2, v: float = 10.0) -> ModelSpec:
    """1-compartment IV model with log-coordinate parameters."""
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=MG),),
        odes=(ODETerm(target="A1", expr="-k10*A1"),),
        parameters=(
            Parameter(name="k10", value=math.log(ke), unit=PERH, coord="log"),
            Parameter(name="V", value=math.log(v), unit=L, coord="log"),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )


def two_comp_macro() -> ModelSpec:
    """2-compartment IV model in macro constants (CL, Q, V1, V2)."""
    return ModelSpec(
        compartments=(
            Compartment(name="A1", unit=MG),
            Compartment(name="A2", unit=MG),
        ),
        odes=(
            ODETerm(target="A1", expr="-(CL/V1 + Q/V1)*A1 + (Q/V2)*A2"),
            ODETerm(target="A2", expr="(Q/V1)*A1 - (Q/V2)*A2"),
        ),
        parameters=(
            Parameter(name="CL", value=2.0, unit=LPH),
            Parameter(name="Q", value=1.0, unit=LPH),
            Parameter(name="V1", value=10.0, unit=L),
            Parameter(name="V2", value=20.0, unit=L),
        ),
        observation=ObservationModel(state="A1", divide_by="V1"),
    )


def two_comp_micro() -> ModelSpec:
    """2-compartment IV model in micro rates (k10, k12, k21) — differently named."""
    return ModelSpec(
        compartments=(
            Compartment(name="C", unit=MG),
            Compartment(name="P", unit=MG),
        ),
        odes=(
            ODETerm(target="C", expr="-(k10+k12)*C + k21*P"),
            ODETerm(target="P", expr="k12*C - k21*P"),
        ),
        parameters=(
            Parameter(name="k10", value=0.3, unit=PERH),
            Parameter(name="k12", value=0.2, unit=PERH),
            Parameter(name="k21", value=0.1, unit=PERH),
            Parameter(name="V1", value=8.0, unit=L),
        ),
        observation=ObservationModel(state="C", divide_by="V1"),
    )


def two_comp_micro_matched() -> ModelSpec:
    """Micro-rate 2-compartment model *numerically equivalent* to two_comp_macro.

    Derived rates k10=CL/V1, k12=Q/V1, k21=Q/V2 with central volume V1 reproduce
    the macro model's predictions exactly — an indistinguishable reparameterization.
    Doses/observes ``A1`` so the same design applies as the macro form.
    """
    return ModelSpec(
        compartments=(
            Compartment(name="A1", unit=MG),
            Compartment(name="A2", unit=MG),
        ),
        odes=(
            ODETerm(target="A1", expr="-(k10+k12)*A1 + k21*A2"),
            ODETerm(target="A2", expr="k12*A1 - k21*A2"),
        ),
        parameters=(
            Parameter(name="k10", value=0.2, unit=PERH),  # CL/V1 = 2/10
            Parameter(name="k12", value=0.1, unit=PERH),  # Q/V1  = 1/10
            Parameter(name="k21", value=0.05, unit=PERH),  # Q/V2  = 1/20
            Parameter(name="V1", value=10.0, unit=L),
        ),
        observation=ObservationModel(state="A1", divide_by="V1"),
    )


def two_comp_mm_elim() -> ModelSpec:
    """1-compartment with Michaelis-Menten (saturable) elimination."""
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=MG),),
        odes=(ODETerm(target="A1", expr="-Vmax*A1/(Km + A1)"),),
        parameters=(
            Parameter(name="Vmax", value=5.0, unit=Unit(expr="mg/h")),
            Parameter(name="Km", value=2.0, unit=MG),
            Parameter(name="V", value=10.0, unit=L),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )


def iv_bolus_design(
    times: tuple[float, ...], *, amount: float = 100.0, into: str = "A1"
) -> Design:
    """An IV-bolus-at-0 design sampling at ``times``."""
    return Design(
        doses=(Dose(compartment=into, amount=amount, unit=MG, time=0.0),),
        sample_times=times,
    )


def held_out_battery(into: str = "A1") -> list[Design]:
    """A small held-out design battery (varied doses + sample grids)."""
    grids = ((0.25, 0.75, 1.5, 3.0, 6.0), (0.5, 1.0, 2.0, 5.0, 9.0, 14.0))
    amounts = (50.0, 150.0)
    return [iv_bolus_design(g, amount=a, into=into) for g in grids for a in amounts]


def confound_product(p: float = 2.0, q: float = 3.0, v: float = 10.0) -> ModelSpec:
    """1-comp whose elimination is ``-(p*q/V)*A1`` — p and q are confounded.

    Only the product ``p*q`` and ``V`` affect predictions, so identifiable rank
    is 2 of 3 parameters; the flat direction lies in the (p, q) plane.
    """
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=MG),),
        odes=(ODETerm(target="A1", expr="-(p*q/V)*A1"),),
        parameters=(
            Parameter(name="p", value=p, unit=Unit(expr="dimensionless")),
            Parameter(name="q", value=q, unit=PERH),
            Parameter(name="V", value=v, unit=L),
        ),
        observation=ObservationModel(state="A1", divide_by="V", error=PROP_ERR),
    )


def confound_product_reparam(r: float = 6.0, v: float = 10.0) -> ModelSpec:
    """Reparameterization of :func:`confound_product`: elimination ``-(r/V)*A1``.

    Parameters are ``(r, q, V)`` with ``r = p*q``; here ``q`` does not appear in
    the dynamics, so it is the unidentifiable direction. Same predictions, same
    identifiable rank (2 of 3) — a non-trivial deficient-rank reparameterization.
    """
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=MG),),
        odes=(ODETerm(target="A1", expr="-(r/V)*A1"),),
        parameters=(
            Parameter(name="r", value=r, unit=PERH),
            Parameter(name="q", value=3.0, unit=PERH),
            Parameter(name="V", value=v, unit=L),
        ),
        observation=ObservationModel(state="A1", divide_by="V", error=PROP_ERR),
    )
