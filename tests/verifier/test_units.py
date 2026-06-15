"""Invariant #4: dimensional analysis catches seeded unit errors (100% recall)."""

import pytest

from drrl.spec import (
    Compartment,
    ModelSpec,
    ObservationModel,
    ODETerm,
    Parameter,
    Unit,
)
from drrl.verifier import check_units
from tests.factories import MG, one_comp_macro, two_comp_micro, two_comp_mm_elim


def _model(
    odes: tuple[ODETerm, ...], params: tuple[Parameter, ...], comps=None
) -> ModelSpec:
    comps = comps or (Compartment(name="A1", unit=MG),)
    return ModelSpec(
        compartments=comps,
        odes=odes,
        parameters=params,
        observation=ObservationModel(state="A1", divide_by="V"),
    )


# --- clean models pass -------------------------------------------------------

CLEAN = [one_comp_macro(), two_comp_micro(), two_comp_mm_elim()]


@pytest.mark.parametrize("spec", CLEAN, ids=["1c_macro", "2c_micro", "mm_elim"])
def test_clean_models_have_coherent_units(spec):
    assert check_units(spec).ok


# --- seeded errors are all caught (100% recall) ------------------------------

# Error 1: clearance used as a first-order rate (L/h * mg = mg*L/h, not mg/h).
_ERR_RATE_VS_CLEARANCE = _model(
    odes=(ODETerm(target="A1", expr="-CL*A1"),),
    params=(
        Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
        Parameter(name="V", value=10.0, unit=Unit(expr="L")),
    ),
)

# Error 2: amount-of-substance vs mass mixed in a sum (mg + mol/L).
_ERR_MASS_VS_SUBSTANCE = _model(
    odes=(ODETerm(target="A1", expr="-Vmax*A1/(Km + A1)"),),
    params=(
        Parameter(name="Vmax", value=5.0, unit=Unit(expr="mg/h")),
        Parameter(name="Km", value=2.0, unit=Unit(expr="mol/L")),  # should be mg
        Parameter(name="V", value=10.0, unit=Unit(expr="L")),
    ),
)

# Error 3: a quadratic term gives mg^2/h instead of mg/h.
_ERR_QUADRATIC = _model(
    odes=(ODETerm(target="A1", expr="-k*A1*A1"),),
    params=(
        Parameter(name="k", value=0.1, unit=Unit(expr="1/h")),
        Parameter(name="V", value=10.0, unit=Unit(expr="L")),
    ),
)

SEEDED_ERRORS = [_ERR_RATE_VS_CLEARANCE, _ERR_MASS_VS_SUBSTANCE, _ERR_QUADRATIC]


@pytest.mark.parametrize(
    "spec", SEEDED_ERRORS, ids=["rate_vs_clearance", "mass_vs_substance", "quadratic"]
)
def test_seeded_unit_errors_are_detected(spec):
    report = check_units(spec)
    assert not report.ok
    assert report.violations


def test_full_recall_on_seeded_suite():
    # Every seeded error must be flagged: recall == 1.0.
    detected = sum(0 if check_units(s).ok else 1 for s in SEEDED_ERRORS)
    assert detected == len(SEEDED_ERRORS)
