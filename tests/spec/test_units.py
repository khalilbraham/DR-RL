"""Tests for the Unit value object and the shared pint registry."""

from drrl.spec import Unit


def test_unit_str_is_expr():
    assert str(Unit(expr="mg/L")) == "mg/L"


def test_same_dimension_true_for_equivalent_units():
    # mg/L and ug/mL are both [mass]/[volume].
    assert Unit(expr="mg/L").same_dimension(Unit(expr="ug/mL"))


def test_same_dimension_false_for_different_dimensions():
    assert not Unit(expr="mg/L").same_dimension(Unit(expr="1/h"))


def test_dimensionality_exposed():
    conc = Unit(expr="mg/L")
    assert conc.dimensionality == conc.pint.dimensionality
