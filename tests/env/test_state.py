"""Tests for edit-action transitions and state -> spec materialization."""

import pytest

from drrl.env import (
    AddCompartment,
    EditError,
    ModelState,
    RemoveCompartment,
    SwapKinetics,
    TuneParam,
    apply_edit,
    build_spec,
    valid_structural_edits,
)
from drrl.verifier import check_units


def test_add_compartment_one_to_two():
    s = apply_edit(ModelState.initial("one_compartment"), AddCompartment())
    assert s.structure == "two_compartment"


def test_remove_compartment_two_to_one():
    s = apply_edit(ModelState.initial("two_compartment"), RemoveCompartment())
    assert s.structure == "one_compartment"


def test_swap_kinetics_round_trip():
    mm = apply_edit(ModelState.initial("one_compartment"), SwapKinetics())
    assert mm.structure == "michaelis_menten"
    back = apply_edit(mm, SwapKinetics())
    assert back.structure == "one_compartment"


def test_invalid_remove_from_one_raises():
    with pytest.raises(EditError):
        apply_edit(ModelState.initial("one_compartment"), RemoveCompartment())


def test_invalid_add_from_two_raises():
    with pytest.raises(EditError):
        apply_edit(ModelState.initial("two_compartment"), AddCompartment())


def test_swap_kinetics_invalid_on_two():
    with pytest.raises(EditError):
        apply_edit(ModelState.initial("two_compartment"), SwapKinetics())


def test_tune_param_scales():
    s0 = ModelState.initial("one_compartment")
    s1 = apply_edit(s0, TuneParam("CL", 2.0))
    assert s1.params["CL"] == pytest.approx(2 * s0.params["CL"])
    # original state unchanged (frozen / copied)
    assert s0.params["CL"] != s1.params["CL"]


def test_tune_unknown_param_raises():
    with pytest.raises(EditError):
        apply_edit(ModelState.initial("one_compartment"), TuneParam("ghost", 2.0))


@pytest.mark.parametrize(
    "structure", ["one_compartment", "two_compartment", "michaelis_menten"]
)
def test_build_spec_is_unit_coherent(structure):
    spec = build_spec(ModelState.initial(structure))
    assert check_units(spec).ok


def test_valid_structural_edits_nonempty():
    for s in ("one_compartment", "two_compartment", "michaelis_menten"):
        assert valid_structural_edits(s)
