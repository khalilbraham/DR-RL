"""Tests for the ModelSpec schema: round-trip, validation, coordinates."""

import math

import pytest

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
from tests.factories import MG, one_comp_macro, two_comp_micro


def test_json_roundtrip_lossless():
    m = two_comp_micro()
    restored = ModelSpec.model_validate_json(m.model_dump_json())
    assert restored == m


def test_invalid_unit_rejected():
    with pytest.raises(ValueError, match="invalid unit"):
        Unit(expr="not_a_unit_xyz")


def test_log_coord_natural_value():
    p = Parameter(name="k", value=math.log(0.25), unit=Unit(expr="1/h"), coord="log")
    assert p.natural_value == pytest.approx(0.25)
    lin = Parameter(name="k", value=0.25, unit=Unit(expr="1/h"))
    assert lin.natural_value == pytest.approx(0.25)


def test_missing_ode_rejected():
    with pytest.raises(ValueError, match="exactly one ODE"):
        ModelSpec(
            compartments=(
                Compartment(name="A1", unit=MG),
                Compartment(name="A2", unit=MG),
            ),
            odes=(ODETerm(target="A1", expr="-k*A1"),),
            parameters=(Parameter(name="k", value=1.0, unit=Unit(expr="1/h")),),
            observation=ObservationModel(state="A1"),
        )


def test_unknown_symbol_in_ode_rejected():
    with pytest.raises(ValueError, match="unknown symbols"):
        ModelSpec(
            compartments=(Compartment(name="A1", unit=MG),),
            odes=(ODETerm(target="A1", expr="-kbad*A1"),),
            parameters=(Parameter(name="k", value=1.0, unit=Unit(expr="1/h")),),
            observation=ObservationModel(state="A1"),
        )


def test_observed_state_must_exist():
    with pytest.raises(ValueError, match="observed state"):
        ModelSpec(
            compartments=(Compartment(name="A1", unit=MG),),
            odes=(ODETerm(target="A1", expr="-k*A1"),),
            parameters=(Parameter(name="k", value=1.0, unit=Unit(expr="1/h")),),
            observation=ObservationModel(state="GHOST"),
        )


def test_divide_by_must_be_parameter():
    with pytest.raises(ValueError, match="divide_by"):
        ModelSpec(
            compartments=(Compartment(name="A1", unit=MG),),
            odes=(ODETerm(target="A1", expr="-k*A1"),),
            parameters=(Parameter(name="k", value=1.0, unit=Unit(expr="1/h")),),
            observation=ObservationModel(state="A1", divide_by="V"),
        )


def test_sample_times_must_be_sorted():
    with pytest.raises(ValueError, match="sorted"):
        Design(
            doses=(Dose(compartment="A1", amount=1.0, unit=MG),),
            sample_times=(2.0, 1.0),
        )


def test_rhs_exprs_keyed_by_compartment():
    m = one_comp_macro()
    rhs = m.rhs_exprs()
    assert set(rhs) == {"A1"}
