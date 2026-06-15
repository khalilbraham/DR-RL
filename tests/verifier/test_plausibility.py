"""Invariant #5: mass balance and non-negativity gates."""

from drrl.sim import SimConfig, get_backend
from drrl.spec import (
    Compartment,
    ModelSpec,
    ObservationModel,
    ODETerm,
    Parameter,
    Unit,
)
from drrl.verifier import check_plausibility
from tests.factories import MG, iv_bolus_design, one_comp_macro, two_comp_micro

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-10, atol=1e-12))
_PERH = Unit(expr="1/h")
_MGPH = Unit(expr="mg/h")
_L = Unit(expr="L")


def test_one_and_two_compartment_are_plausible():
    for spec, into in ((one_comp_macro(), "A1"), (two_comp_micro(), "C")):
        design = iv_bolus_design(
            tuple(float(t) for t in (1.0, 2.0, 4.0, 8.0)), into=into
        )
        report = check_plausibility(spec, design, _BACKEND)
        assert report.mass_balance_ok
        assert report.nonneg_ok


def test_mass_creating_model_is_flagged():
    # A2 gains k12*A1 but A1 never loses it -> net flux creates mass.
    spec = ModelSpec(
        compartments=(Compartment(name="A1", unit=MG), Compartment(name="A2", unit=MG)),
        odes=(
            ODETerm(target="A1", expr="-(CL/V)*A1"),
            ODETerm(target="A2", expr="k12*A1"),
        ),
        parameters=(
            Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
            Parameter(name="V", value=10.0, unit=_L),
            Parameter(name="k12", value=0.5, unit=_PERH),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )
    report = check_plausibility(spec, iv_bolus_design((1.0, 2.0, 4.0)), _BACKEND)
    assert not report.mass_balance_ok


def test_negative_going_state_is_flagged():
    # Zero-order elimination with no floor drives the amount negative.
    spec = ModelSpec(
        compartments=(Compartment(name="A1", unit=MG),),
        odes=(ODETerm(target="A1", expr="-k0"),),
        parameters=(
            Parameter(name="k0", value=50.0, unit=_MGPH),
            Parameter(name="V", value=10.0, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )
    # Dose 100, k0 50 mg/h -> empty at t=2; sample out to t=5.
    design = iv_bolus_design((1.0, 2.0, 3.0, 4.0, 5.0), amount=100.0)
    report = check_plausibility(spec, design, _BACKEND)
    assert not report.nonneg_ok
    # zero-order elimination is a valid (negative constant) net flux, not a source
    assert report.mass_balance_ok
