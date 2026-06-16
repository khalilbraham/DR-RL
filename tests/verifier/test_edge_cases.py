"""Edge-case coverage for verifier Tier-A layers."""

import pytest

from drrl.sim.result import SimulationResult
from drrl.spec import (
    Compartment,
    ModelSpec,
    ObservationModel,
    ODETerm,
    Parameter,
    Unit,
)
from drrl.verifier import check_execution, check_plausibility, check_units
from tests.factories import MG, iv_bolus_design

_L = Unit(expr="L")
_PERH = Unit(expr="1/h")


def _spec(expr: str, params: tuple[Parameter, ...]) -> ModelSpec:
    return ModelSpec(
        compartments=(Compartment(name="A1", unit=MG),),
        odes=(ODETerm(target="A1", expr=expr),),
        parameters=params,
        observation=ObservationModel(state="A1", divide_by="V"),
    )


def test_units_numeric_coefficient_and_dimensionless_exp_ok():
    # Numeric coefficient (2) and exp of a dimensionless ratio are coherent.
    spec = _spec(
        "-2*(CL/V)*A1*exp(-A1/Kd)",
        (
            Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
            Parameter(name="V", value=10.0, unit=_L),
            Parameter(name="Kd", value=5.0, unit=MG),
        ),
    )
    assert check_units(spec).ok


def test_units_exp_of_dimensioned_argument_flagged():
    spec = _spec(
        "-k*exp(A1)",
        (
            Parameter(name="k", value=0.1, unit=_PERH),
            Parameter(name="V", value=10.0, unit=_L),
        ),
    )
    report = check_units(spec)
    assert not report.ok
    assert any("dimensionless" in v for v in report.violations)


def test_units_symbolic_exponent_flagged():
    spec = _spec(
        "-A1**k",
        (
            Parameter(name="k", value=2.0, unit=Unit(expr="dimensionless")),
            Parameter(name="V", value=10.0, unit=_L),
        ),
    )
    report = check_units(spec)
    assert not report.ok


def test_execution_captures_arbitrary_backend_error():
    class _BoomBackend:
        def simulate(self, spec, design, *, with_sensitivities=False):
            raise RuntimeError("solver exploded")

    spec = _spec(
        "-(CL/V)*A1",
        (
            Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
            Parameter(name="V", value=10.0, unit=_L),
        ),
    )
    report, result = check_execution(spec, iv_bolus_design((1.0, 2.0)), _BoomBackend())
    assert not report.ok
    assert result is None
    assert "RuntimeError" in report.message


def test_execution_flags_nonfinite_predictions():
    import numpy as np

    class _NaNBackend:
        def simulate(self, spec, design, *, with_sensitivities=False):
            t = np.asarray(design.sample_times, dtype=np.float64)
            return SimulationResult(
                times=t,
                states=np.full((len(t), 1), np.nan),
                observed=np.full(len(t), np.nan),
                sensitivities=None,
                integrator_ok=True,
                diagnostics={},
            )

    spec = _spec(
        "-(CL/V)*A1",
        (
            Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
            Parameter(name="V", value=10.0, unit=_L),
        ),
    )
    report, _ = check_execution(spec, iv_bolus_design((1.0, 2.0)), _NaNBackend())
    assert not report.ok
    assert "non-finite" in report.message


def test_plausibility_allows_zero_order_input():
    # A constant positive source is a legitimate zero-order input / turnover
    # (e.g. infusion, TMDD target synthesis) -- not a mass-balance violation.
    from drrl.sim import SimConfig, get_backend

    spec = _spec(
        "k0 - (CL/V)*A1",
        (
            Parameter(name="k0", value=5.0, unit=Unit(expr="mg/h")),
            Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
            Parameter(name="V", value=10.0, unit=_L),
        ),
    )
    backend = get_backend("scipy", SimConfig(rtol=1e-9, atol=1e-12))
    report = check_plausibility(spec, iv_bolus_design((1.0, 2.0, 4.0)), backend)
    assert report.mass_balance_ok


@pytest.mark.parametrize("expr", ["-A1**k"])
def test_units_report_lists_violation_for_each_bad_term(expr: str):
    spec = _spec(
        expr,
        (
            Parameter(name="k", value=2.0, unit=Unit(expr="dimensionless")),
            Parameter(name="V", value=10.0, unit=_L),
        ),
    )
    assert len(check_units(spec).violations) >= 1
