"""Tests for the schema and execution Tier-A gates."""

from drrl.sim import SimConfig, get_backend
from drrl.verifier import check_execution, check_schema
from tests.factories import iv_bolus_design, one_comp_macro

_TIGHT = SimConfig(rtol=1e-10, atol=1e-12)


def test_schema_accepts_valid_model():
    spec = one_comp_macro()
    report, parsed = check_schema(spec.model_dump_json())
    assert report.ok
    assert parsed is not None
    assert parsed == spec


def test_schema_captures_errors_without_raising():
    report, parsed = check_schema({"compartments": [], "odes": []})
    assert not report.ok
    assert parsed is None
    assert report.errors


def test_schema_rejects_unknown_ode_symbol():
    bad = {
        "compartments": [{"name": "A1", "unit": {"expr": "mg"}}],
        "odes": [{"target": "A1", "expr": "-kbad*A1"}],
        "parameters": [
            {"name": "k", "value": 1.0, "unit": {"expr": "1/h"}, "coord": "linear"}
        ],
        "observation": {"state": "A1"},
    }
    report, parsed = check_schema(bad)
    assert not report.ok
    assert parsed is None


def test_execution_ok_for_valid_model():
    spec = one_comp_macro()
    design = iv_bolus_design(tuple(float(t) for t in (0.5, 1.0, 2.0, 4.0)))
    report, result = check_execution(spec, design, get_backend("scipy", _TIGHT))
    assert report.ok
    assert report.integrator_ok
    assert result is not None


def test_execution_captures_unsupported_route():
    from drrl.spec import Design, Dose, Unit

    spec = one_comp_macro()
    design = Design(
        doses=(Dose(compartment="A1", amount=1.0, unit=Unit(expr="mg"), route="oral"),),
        sample_times=(1.0, 2.0),
    )
    report, result = check_execution(spec, design, get_backend("scipy", _TIGHT))
    assert not report.ok
    assert result is None
    assert "unsupported" in report.message
