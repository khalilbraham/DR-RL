"""Invariant #6: the competitor/admissible set is open (not a static literal)."""

from drrl.sim import SimConfig, get_backend
from drrl.verifier import CompetitorSet, distinguish, enumerate_structures
from tests.factories import iv_bolus_design, one_comp_macro, one_comp_micro

_BACKEND = get_backend("scipy", SimConfig(rtol=1e-10, atol=1e-12))


def test_competitor_set_is_open():
    from tests.factories import two_comp_micro_matched

    cs = CompetitorSet()
    # Source 1: a policy rollout (a 1-compartment structure).
    rollout_spec = one_comp_micro(ke=0.37, v=12.0)
    cs.add_rollout(rollout_spec)
    # Source 2: a replay-buffer structure (a 2-compartment structure).
    cs.add_replay(two_comp_micro_matched())
    # Source 3: the programmatic enumeration prior (adds the MM structure, which
    # neither of the above provided). Dedup by canonical key keeps first claims.
    cs.add_enumeration()

    sources = cs.sources()
    assert sources.get("rollout", 0) >= 1
    assert sources.get("enumeration", 0) >= 1
    assert sources.get("replay", 0) >= 1

    # The rollout-derived member is present (by canonical key).
    rollout_key = rollout_spec.canonicalize().key
    member_keys = cs.keys()
    assert rollout_key in member_keys
    assert cs.source_of(rollout_key) == "rollout"

    # Not a static literal: members reflect the runtime inputs.
    assert len(cs.members()) >= 3


def test_enumeration_is_generated_not_hardcoded_list():
    # enumerate_structures yields freshly-built specs (a generator), and the
    # set dedupes by canonical key.
    specs = list(enumerate_structures())
    assert len(specs) >= 3
    keys = {s.canonicalize().key for s in specs}
    assert len(keys) == len(specs)  # all structurally distinct


def test_admissible_set_via_distinguish():
    # The admissible set = competitors indistinguishable from the reference.
    ref = one_comp_macro(cl=2.0, v=10.0)
    cs = CompetitorSet()
    cs.add_rollout(one_comp_micro(ke=0.2, v=10.0))  # == ref (indistinguishable)
    cs.add_enumeration()  # includes a 1c with CL/V=0.2 (== ref) and others

    designs = [iv_bolus_design((0.5, 1.0, 2.0, 4.0, 8.0))]
    admissible = [
        m
        for m in cs.members()
        if distinguish(ref, m, designs, _BACKEND).verdict == "indistinguishable"
    ]
    # At least the matched 1-compartment competitor is admissible; the 2c/MM are not.
    assert len(admissible) >= 1
    assert len(admissible) < len(cs.members())
