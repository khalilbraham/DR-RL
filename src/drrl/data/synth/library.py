"""Programmatic synthetic PK library with ground-truth labels.

The generator *knows the construction*, so it emits the reparameterization-
invariant equivalence class (canonical key) and an identifiability label for
free. It also produces deliberately **indistinguishable model pairs** under a
given design — the only reliable source of abstention ground truth.

Scope (the MVE-relevant subset, per the build order): 1- and 2-compartment
first-order models, a 1-compartment Michaelis-Menten model, and a deliberately
non-identifiable confounded model. TMDD / PBPK / indirect-response models arrive
with the Phase-7 curriculum.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

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

_MG = Unit(expr="mg")
_L = Unit(expr="L")
_LPH = Unit(expr="L/h")
_PERH = Unit(expr="1/h")
_MGPH = Unit(expr="mg/h")
_PROP = ErrorModel(kind="proportional", sigma_prop=0.1)


@dataclass(frozen=True)
class SyntheticCase:
    """A synthetic problem with ground-truth labels known by construction.

    Attributes:
        name: Mechanism label.
        spec: The data-generating model.
        design: A representative dosing/sampling design.
        equivalence_key: Canonical (reparameterization-invariant) key — the
            ground-truth equivalence class.
        fully_identifiable: Whether all parameters are identifiable under a rich
            design (ground truth from construction).
        n_params: Number of parameters.
    """

    name: str
    spec: ModelSpec
    design: Design
    equivalence_key: str
    fully_identifiable: bool
    n_params: int


def _design(amount: float = 100.0, into: str = "A1") -> Design:
    times = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 18.0, 24.0)
    return Design(
        doses=(Dose(compartment=into, amount=amount, unit=_MG),), sample_times=times
    )


def one_compartment(cl: float = 2.0, v: float = 10.0) -> ModelSpec:
    """1-compartment first-order IV model (fully identifiable)."""
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=_MG),),
        odes=(ODETerm(target="A1", expr="-(CL/V)*A1"),),
        parameters=(
            Parameter(name="CL", value=cl, unit=_LPH),
            Parameter(name="V", value=v, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V", error=_PROP),
    )


def two_compartment() -> ModelSpec:
    """2-compartment first-order IV model (fully identifiable)."""
    return ModelSpec(
        compartments=(
            Compartment(name="A1", unit=_MG),
            Compartment(name="A2", unit=_MG),
        ),
        odes=(
            ODETerm(target="A1", expr="-(k10+k12)*A1 + k21*A2"),
            ODETerm(target="A2", expr="k12*A1 - k21*A2"),
        ),
        parameters=(
            Parameter(name="k10", value=0.2, unit=_PERH),
            Parameter(name="k12", value=0.1, unit=_PERH),
            Parameter(name="k21", value=0.05, unit=_PERH),
            Parameter(name="V1", value=10.0, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V1", error=_PROP),
    )


def michaelis_menten(vmax: float = 8.0, km: float = 3.0, v: float = 10.0) -> ModelSpec:
    """1-compartment Michaelis-Menten elimination (saturable; nonlinear)."""
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=_MG),),
        odes=(ODETerm(target="A1", expr="-Vmax*A1/(Km + A1)"),),
        parameters=(
            Parameter(name="Vmax", value=vmax, unit=_MGPH),
            Parameter(name="Km", value=km, unit=_MG),
            Parameter(name="V", value=v, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V", error=_PROP),
    )


def confounded() -> ModelSpec:
    """1-compartment with a product-confounded rate (NOT fully identifiable)."""
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=_MG),),
        odes=(ODETerm(target="A1", expr="-(p*q/V)*A1"),),
        parameters=(
            Parameter(name="p", value=2.0, unit=Unit(expr="dimensionless")),
            Parameter(name="q", value=0.1, unit=_PERH),
            Parameter(name="V", value=10.0, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V", error=_PROP),
    )


def two_compartment_macro() -> ModelSpec:
    """2-compartment in macro constants — a reparam of :func:`two_compartment`."""
    return ModelSpec(
        compartments=(
            Compartment(name="A1", unit=_MG),
            Compartment(name="A2", unit=_MG),
        ),
        odes=(
            ODETerm(target="A1", expr="-(CL/V1 + Q/V1)*A1 + (Q/V2)*A2"),
            ODETerm(target="A2", expr="(Q/V1)*A1 - (Q/V2)*A2"),
        ),
        parameters=(
            Parameter(name="CL", value=2.0, unit=_LPH),  # k10*V1
            Parameter(name="Q", value=1.0, unit=_LPH),  # k12*V1
            Parameter(name="V1", value=10.0, unit=_L),
            Parameter(name="V2", value=20.0, unit=_L),  # Q/k21
        ),
        observation=ObservationModel(state="A1", divide_by="V1", error=_PROP),
    )


def generate_cases() -> list[SyntheticCase]:
    """Return the labeled synthetic case library."""
    builders: list[tuple[str, ModelSpec, bool]] = [
        ("one_compartment", one_compartment(), True),
        ("two_compartment", two_compartment(), True),
        ("michaelis_menten", michaelis_menten(), True),
        ("confounded", confounded(), False),
    ]
    cases: list[SyntheticCase] = []
    for name, spec, identifiable in builders:
        cases.append(
            SyntheticCase(
                name=name,
                spec=spec,
                design=_design(),
                equivalence_key=spec.canonicalize().key,
                fully_identifiable=identifiable,
                n_params=len(spec.parameters),
            )
        )
    return cases


@dataclass(frozen=True)
class ModelPair:
    """A labeled pair for distinguishability ground truth.

    Attributes:
        name: Pair label.
        reference: Reference model.
        candidate: Candidate model.
        design: Operative design.
        indistinguishable: Ground-truth label (from construction).
    """

    name: str
    reference: ModelSpec
    candidate: ModelSpec
    design: Design
    indistinguishable: bool


def indistinguishable_pairs() -> Iterator[ModelPair]:
    """Yield labeled (in)distinguishable model pairs under a shared design.

    Indistinguishable pairs are exact reparameterizations (identical predictions);
    distinguishable pairs differ in structure or in a prediction-affecting rate.
    """
    design = _design()
    yield ModelPair(
        name="1c macro vs micro (reparam)",
        reference=one_compartment(cl=2.0, v=10.0),
        candidate=_one_compartment_micro(ke=0.2, v=10.0),
        design=design,
        indistinguishable=True,
    )
    yield ModelPair(
        name="2c macro vs micro (reparam)",
        reference=two_compartment_macro(),
        candidate=two_compartment(),
        design=design,
        indistinguishable=True,
    )
    yield ModelPair(
        name="1c different elimination rate",
        reference=one_compartment(cl=2.0, v=10.0),
        candidate=one_compartment(cl=5.0, v=10.0),
        design=design,
        indistinguishable=False,
    )
    yield ModelPair(
        name="1c vs 2c (different order)",
        reference=one_compartment(),
        candidate=two_compartment(),
        design=design,
        indistinguishable=False,
    )


def _one_compartment_micro(ke: float = 0.2, v: float = 10.0) -> ModelSpec:
    """1-compartment in micro rate (k10) — reparam of :func:`one_compartment`."""
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=_MG),),
        odes=(ODETerm(target="A1", expr="-k10*A1"),),
        parameters=(
            Parameter(name="k10", value=ke, unit=_PERH),
            Parameter(name="V", value=v, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V", error=_PROP),
    )
