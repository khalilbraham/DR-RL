"""Synthetic-data ground-truth labels cross-validated against the verifier.

The generator emits equivalence/identifiability labels and (in)distinguishability
labels by construction. These tests confirm those labels agree with what the
independent verifier computes — otherwise the labels would be untrustworthy.
"""

import pytest

from drrl.data.synth import generate_cases, indistinguishable_pairs
from drrl.sim import SimConfig, get_backend
from drrl.spec import Design, Dose, Unit
from drrl.verifier import distinguish, identifiability

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-10, atol=1e-12))
_MG = Unit(expr="mg")


def _battery(design: Design) -> list[Design]:
    # The case design plus a second dose level => a richer held-out battery.
    alt = Design(
        doses=(
            Dose(compartment=d.compartment, amount=d.amount * 2.0, unit=_MG)
            for d in design.doses
        ),
        sample_times=design.sample_times,
    )
    return [design, alt]


def test_equivalence_keys_are_canonical():
    for case in generate_cases():
        assert case.equivalence_key == case.spec.canonicalize().key


@pytest.mark.parametrize("case", generate_cases(), ids=lambda c: c.name)
def test_identifiability_labels_match_verifier(case):
    report = identifiability(case.spec, _battery(case.design), _BACKEND)
    if case.fully_identifiable:
        assert report.identifiable_fraction == pytest.approx(1.0)
    else:
        assert report.identifiable_fraction < 1.0


@pytest.mark.parametrize("pair", list(indistinguishable_pairs()), ids=lambda p: p.name)
def test_distinguishability_labels_match_oracle(pair):
    verdict = distinguish(
        pair.reference, pair.candidate, [pair.design], _BACKEND
    ).verdict
    expected = "indistinguishable" if pair.indistinguishable else "distinguishable"
    assert verdict == expected


def test_library_equivalence_classes():
    cases = {c.name: c for c in generate_cases()}
    assert len(cases) >= 4
    # `confounded` shares the 1-compartment first-order *structure*; it differs
    # only in identifiability, not structure -> same canonical key. This is the
    # equivalence-class concept made concrete.
    assert (
        cases["one_compartment"].equivalence_key == cases["confounded"].equivalence_key
    )
    # The saturable (MM) and 2-compartment structures are distinct classes.
    keys = {c.equivalence_key for c in cases.values()}
    assert len(keys) == 3
