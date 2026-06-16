"""TMDD mechanism: integration, units, mass balance, baseline, nonlinearity."""

import numpy as np

from drrl.data.synth import tmdd
from drrl.sim import SimConfig, get_backend
from drrl.spec import Design, Dose, Unit
from drrl.verifier import check_plausibility, check_units

_NM = Unit(expr="nmol/L")
_CFG = SimConfig(rtol=1e-8, atol=1e-10)


def _design(dose: float, times: tuple[float, ...]) -> Design:
    return Design(
        doses=(Dose(compartment="L", amount=dose, unit=_NM),), sample_times=times
    )


def test_tmdd_units_and_mass_balance():
    spec = tmdd()
    assert check_units(spec).ok
    backend = get_backend("scipy", _CFG)
    # Target synthesis (kdeg*r0) is a legitimate zero-order source, not creation.
    report = check_plausibility(spec, _design(10.0, (1.0, 4.0, 12.0, 24.0)), backend)
    assert report.mass_balance_ok
    assert report.nonneg_ok


def test_tmdd_target_holds_at_baseline_without_drug():
    spec = tmdd(r0=1.0)
    backend = get_backend("scipy", _CFG)
    # Dose 0 into L: target R should stay at its baseline r0 = 1.0.
    design = Design(
        doses=(Dose(compartment="L", amount=0.0, unit=_NM),),
        sample_times=(1.0, 12.0, 48.0),
    )
    result = backend.simulate(spec, design)
    r_index = spec.state_names.index("R")
    np.testing.assert_allclose(result.states[:, r_index], 1.0, rtol=1e-4)


def test_tmdd_backends_agree():
    spec = tmdd()
    design = _design(50.0, tuple(np.linspace(0.5, 48.0, 24)))
    rs = get_backend("scipy", _CFG).simulate(spec, design)
    rd = get_backend("diffrax", _CFG).simulate(spec, design)
    np.testing.assert_allclose(rs.observed, rd.observed, rtol=1e-4, atol=1e-6)


def test_tmdd_clearance_is_dose_dependent():
    # Target-mediated (saturable) clearance: dose-normalized exposure increases
    # with dose (low dose cleared relatively faster). This is the nonlinearity
    # that a linear 1-compartment model cannot reproduce.
    spec = tmdd()
    backend = get_backend("scipy", _CFG)
    times = tuple(np.linspace(0.25, 60.0, 60))
    low = backend.simulate(spec, _design(2.0, times)).observed
    high = backend.simulate(spec, _design(100.0, times)).observed
    auc_low = float(np.trapezoid(low, np.asarray(times))) / 2.0
    auc_high = float(np.trapezoid(high, np.asarray(times))) / 100.0
    assert auc_high > 1.2 * auc_low  # clearly nonlinear (not dose-proportional)
