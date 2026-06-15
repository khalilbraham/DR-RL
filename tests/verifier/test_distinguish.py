"""Invariant #3: distinguishability calibration on a labeled pair battery.

Ground-truth labels come from construction (reparameterizations are
indistinguishable; different structures/parameters are distinguishable). We
check that (a) the exact linear Markov fast path matches the labels, (b) the
general predictive oracle agrees with the fast path, and (c) the predictive
oracle's FP/FN rates are within the configured bounds.
"""

from omegaconf import DictConfig

from drrl.sim import SimConfig, get_backend
from drrl.spec import ErrorModel, ModelSpec, ObservationModel
from drrl.verifier import linear_indistinguishable, predictive_distinguishable
from tests.factories import (
    iv_bolus_design,
    one_comp_macro,
    one_comp_micro,
    two_comp_macro,
    two_comp_micro_matched,
)

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-11, atol=1e-13))
_PROP = ErrorModel(kind="proportional", sigma_prop=0.1)


def _with_err(spec: ModelSpec) -> ModelSpec:
    obs = spec.observation
    new_obs = ObservationModel(
        state=obs.state, divide_by=obs.divide_by, transform=obs.transform, error=_PROP
    )
    return spec.model_copy(update={"observation": new_obs})


def _set_param(spec: ModelSpec, name: str, value: float) -> ModelSpec:
    params = tuple(
        p.model_copy(update={"value": value}) if p.name == name else p
        for p in spec.parameters
    )
    return spec.model_copy(update={"parameters": params})


# (reference, candidate, indistinguishable_label)
def _battery() -> list[tuple[ModelSpec, ModelSpec, bool]]:
    one_macro = _with_err(one_comp_macro(cl=2.0, v=10.0))
    one_micro = _with_err(one_comp_micro(ke=0.2, v=10.0))  # ke = CL/V -> matched
    one_fast = _with_err(one_comp_micro(ke=0.5, v=10.0))
    two_macro = _with_err(two_comp_macro())
    two_micro = _with_err(two_comp_micro_matched())  # numerically equal to two_macro
    two_other = _with_err(_set_param(two_comp_macro(), "Q", 3.0))
    return [
        (one_macro, one_micro, True),  # reparameterization -> indistinguishable
        (one_macro, one_macro, True),  # identical
        (two_macro, two_micro, True),  # matched macro<->micro reparam
        (one_macro, one_fast, False),  # different elimination rate
        (one_macro, two_macro, False),  # different order
        (two_macro, two_other, False),  # different inter-compartmental rate
    ]


_DESIGNS = [
    iv_bolus_design(tuple(0.25 * (i + 1) for i in range(20))),
    iv_bolus_design((0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0)),
]


def test_linear_fast_path_matches_labels():
    for ref, cand, indist in _battery():
        verdict = linear_indistinguishable(ref, cand, "A1", "A1", rtol=1e-6, atol=1e-9)
        assert verdict is indist


def test_predictive_oracle_agrees_with_fast_path():
    for ref, cand, _ in _battery():
        fast = linear_indistinguishable(ref, cand, "A1", "A1", rtol=1e-6, atol=1e-9)
        predictive = not predictive_distinguishable(
            ref, cand, _DESIGNS, _BACKEND, alpha=0.05
        )
        assert fast == predictive


def test_distinguishability_calibration(tol: DictConfig):
    battery = _battery()
    fp = fn = n_neg = n_pos = 0
    for ref, cand, indist in battery:
        distinguishable = predictive_distinguishable(
            ref, cand, _DESIGNS, _BACKEND, alpha=tol.distinguish_nominal_fpr
        )
        if indist:
            n_neg += 1
            fp += int(distinguishable)  # called distinguishable but truly indist
        else:
            n_pos += 1
            fn += int(not distinguishable)  # called indist but truly distinguishable
    fpr = fp / n_neg
    fnr = fn / n_pos
    assert fpr <= tol.distinguish_max_fpr
    assert fnr <= tol.distinguish_max_fnr


def test_distinguish_report_records_verdict_and_method():
    from drrl.verifier import distinguish

    ref = _with_err(one_comp_macro())
    cand = _with_err(one_comp_micro(ke=0.2))
    report = distinguish(ref, cand, _DESIGNS, _BACKEND, admissible=("k1", "k2"))
    assert report.method == "transfer_function"
    assert report.verdict == "indistinguishable"
    assert report.admissible == ("k1", "k2")


def test_distinguish_predictive_path_for_nonlinear():
    from drrl.verifier import distinguish
    from tests.factories import two_comp_mm_elim

    ref = _with_err(two_comp_mm_elim())
    cand = _with_err(two_comp_mm_elim())
    report = distinguish(ref, cand, [iv_bolus_design((1.0, 2.0, 4.0, 8.0))], _BACKEND)
    assert report.method == "predictive"
    assert report.verdict == "indistinguishable"
