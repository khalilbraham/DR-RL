"""Tests for optimal experiment design (parameter precision + discrimination)."""

from drrl.sim import SimConfig, get_backend
from drrl.spec import ErrorModel, ModelSpec, ObservationModel
from drrl.verifier import d_optimal_value, score_design
from tests.factories import iv_bolus_design, one_comp_macro, two_comp_macro

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-11, atol=1e-13))
_PROP = ErrorModel(kind="proportional", sigma_prop=0.1)


def _with_err(spec: ModelSpec) -> ModelSpec:
    obs = spec.observation
    return spec.model_copy(
        update={
            "observation": ObservationModel(
                state=obs.state,
                divide_by=obs.divide_by,
                transform=obs.transform,
                error=_PROP,
            )
        }
    )


def test_richer_design_has_higher_d_optimality():
    spec = one_comp_macro()
    sparse = iv_bolus_design((4.0, 8.0))
    rich = iv_bolus_design(tuple(0.5 * (i + 1) for i in range(16)))
    assert d_optimal_value(spec, rich, _BACKEND) > d_optimal_value(
        spec, sparse, _BACKEND
    )


def test_discrimination_design_resolves_ambiguity():
    # A reference vs a structurally different competitor: an informative design
    # should make them separable (resolves the ambiguity); a single late sample
    # should not.
    ref = _with_err(one_comp_macro(cl=2.0, v=10.0))
    competitor = _with_err(two_comp_macro())  # different structure, central naming A1

    informative = iv_bolus_design(tuple(0.25 * (i + 1) for i in range(20)))
    # A single very early sample: both models are ~ Dose/V there, so it barely
    # discriminates them.
    weak = iv_bolus_design((0.1,))

    good = score_design(
        ref, informative, _BACKEND, competitors=[competitor], criterion="eig"
    )
    poor = score_design(ref, weak, _BACKEND, competitors=[competitor], criterion="eig")

    assert good.utility > poor.utility
    assert good.resolves_ambiguity
    assert not poor.resolves_ambiguity


def test_score_design_d_optimal_criterion():
    spec = _with_err(one_comp_macro())
    rich = iv_bolus_design(tuple(0.5 * (i + 1) for i in range(12)))
    report = score_design(spec, rich, _BACKEND, criterion="d_optimal")
    assert report.criterion == "d_optimal"
    assert report.utility > float("-inf")
    assert not report.resolves_ambiguity


def test_no_competitors_gives_zero_discrimination():
    ref = one_comp_macro()
    report = score_design(
        ref, iv_bolus_design((1.0, 2.0)), _BACKEND, competitors=[], criterion="eig"
    )
    assert report.utility == 0.0
    assert not report.resolves_ambiguity
