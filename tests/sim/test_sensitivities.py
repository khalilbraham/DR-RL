"""Forward sensitivities (diffrax autodiff) validated against finite differences.

The diffrax backend computes ``d observed / d theta`` by autodiff; the scipy
backend computes it by central finite differences. Agreement validates the
sensitivity machinery that the identifiability metric depends on.
"""

import numpy as np
import pytest
from omegaconf import DictConfig

from drrl.sim import SimConfig, get_backend
from tests.factories import iv_bolus_design, one_comp_macro, two_comp_micro

_TIGHT = SimConfig(rtol=1e-11, atol=1e-13)


def test_one_comp_sensitivities_match_finite_difference(tol: DictConfig):
    spec = one_comp_macro()
    times = tuple(np.linspace(0.5, 8.0, 16))
    design = iv_bolus_design(times)

    ad = get_backend("diffrax", _TIGHT).simulate(spec, design, with_sensitivities=True)
    fd = get_backend("scipy", _TIGHT).simulate(spec, design, with_sensitivities=True)

    assert ad.sensitivities is not None
    assert fd.sensitivities is not None
    assert ad.sensitivities.shape == (len(times), len(spec.parameters))
    np.testing.assert_allclose(
        np.asarray(ad.sensitivities),
        np.asarray(fd.sensitivities),
        rtol=tol.sensitivity_rtol,
        atol=tol.sensitivity_atol,
    )


def test_two_comp_sensitivities_match_finite_difference(tol: DictConfig):
    spec = two_comp_micro()
    times = tuple(np.linspace(0.5, 10.0, 20))
    design = iv_bolus_design(times, into="C")

    ad = get_backend("diffrax", _TIGHT).simulate(spec, design, with_sensitivities=True)
    fd = get_backend("scipy", _TIGHT).simulate(spec, design, with_sensitivities=True)

    assert ad.sensitivities is not None
    assert fd.sensitivities is not None
    np.testing.assert_allclose(
        np.asarray(ad.sensitivities),
        np.asarray(fd.sensitivities),
        rtol=tol.sensitivity_rtol,
        atol=tol.sensitivity_atol,
    )


def test_sensitivities_absent_when_not_requested():
    spec = one_comp_macro()
    design = iv_bolus_design(tuple(np.linspace(0.5, 4.0, 8)))
    result = get_backend("diffrax", _TIGHT).simulate(spec, design)
    assert result.sensitivities is None


def test_unsupported_route_raises():
    from drrl.spec import Design, Dose, Unit

    spec = one_comp_macro()
    design = Design(
        doses=(Dose(compartment="A1", amount=1.0, unit=Unit(expr="mg"), route="oral"),),
        sample_times=(1.0, 2.0),
    )
    with pytest.raises(NotImplementedError, match="iv_bolus only"):
        get_backend("scipy", _TIGHT).simulate(spec, design)
