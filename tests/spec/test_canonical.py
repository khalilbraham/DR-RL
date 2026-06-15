"""Invariant #2 (partial): reparameterization-invariance of the canonical form.

The canonical key must be identical under (a) macro<->micro reparameterization,
(b) linear<->log parameter coordinates, and (c) renaming of compartments and
parameters. Genuinely different structures must get different keys.
"""

import keyword

import sympy
from hypothesis import given
from hypothesis import strategies as st

from drrl.spec import (
    Compartment,
    ModelSpec,
    ObservationModel,
    ODETerm,
)
from tests.factories import (
    one_comp_log,
    one_comp_macro,
    one_comp_micro,
    two_comp_macro,
    two_comp_micro,
    two_comp_mm_elim,
)


def test_one_comp_macro_micro_log_all_equal():
    macro = one_comp_macro().canonicalize()
    micro = one_comp_micro().canonicalize()
    log = one_comp_log().canonicalize()
    assert macro == micro == log
    assert macro.key == micro.key == log.key


def test_two_comp_macro_micro_equal():
    assert two_comp_macro().canonicalize() == two_comp_micro().canonicalize()


def test_one_vs_two_comp_distinguished():
    assert one_comp_macro().canonicalize() != two_comp_macro().canonicalize()


def test_linear_vs_saturable_elimination_distinguished():
    # Same single compartment, but first-order vs Michaelis-Menten elimination
    # are structurally different -> different canonical keys.
    assert one_comp_macro().canonicalize() != two_comp_mm_elim().canonicalize()


def _rename(spec: ModelSpec, mapping: dict[str, str]) -> ModelSpec:
    """Rename compartments and parameters consistently (a pure reparameterization)."""
    old_syms = spec.symbols
    sub = {old_syms[k]: sympy.Symbol(v, positive=True) for k, v in mapping.items()}
    rhs = spec.rhs_exprs()
    new_odes = tuple(
        ODETerm(target=mapping[o.target], expr=str(rhs[o.target].subs(sub)))
        for o in spec.odes
    )
    new_comps = tuple(
        Compartment(name=mapping[c.name], unit=c.unit) for c in spec.compartments
    )
    new_params = tuple(
        p.model_copy(update={"name": mapping[p.name]}) for p in spec.parameters
    )
    obs = spec.observation
    new_obs = ObservationModel(
        state=mapping[obs.state],
        divide_by=mapping[obs.divide_by] if obs.divide_by else None,
        transform=obs.transform,
        error=obs.error,
    )
    return ModelSpec(
        compartments=new_comps,
        odes=new_odes,
        parameters=new_params,
        observation=new_obs,
    )


# Valid, non-keyword lowercase identifiers (sympy parses these as symbols).
_IDENT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=6).filter(
    lambda s: not keyword.iskeyword(s)
)


@given(names=st.lists(_IDENT, min_size=6, max_size=6, unique=True))
def test_renaming_is_canonical_invariant(names: list[str]):
    spec = two_comp_micro()
    originals = list(spec.state_names) + list(spec.param_names)
    mapping = dict(zip(originals, names, strict=True))
    renamed = _rename(spec, mapping)
    assert renamed.canonicalize() == spec.canonicalize()
