"""Model builders shared across Phase 1 tests (canonical PK structures)."""

from __future__ import annotations

import math

from drrl.spec import (
    Compartment,
    Design,
    Dose,
    ModelSpec,
    ObservationModel,
    ODETerm,
    Parameter,
    Unit,
)

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
