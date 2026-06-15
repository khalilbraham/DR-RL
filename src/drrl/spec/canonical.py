"""Reparameterization-invariant canonical form of a model's *structure*.

The canonical form is a labeled directed graph over compartments plus a SINK
node, derived from the ODE structure, **not** from parameter names, values, or
coordinates. Two models that describe the same kinetic structure — whether
written in macro constants ``(CL, V)`` or micro rates ``(k10, k12, k21)``, in
linear or log parameter coordinates — produce the *same* canonical key.

How it is built (all reparameterization-invariant):

1. **Typed transfer edges** from the Jacobian sparsity: an off-diagonal entry
   ``d(dX/dt)/dS`` that is nonzero is an edge ``S -> X``. Its *type* is
   ``"first_order"`` if that derivative is state-independent (linear) or
   ``"saturable"`` if it still depends on a state (e.g. Michaelis-Menten).
2. **Typed elimination/source edges** from the net flux ``sum_X dX/dt``. After
   cancellation, surviving terms are exactly the mass entering/leaving the
   system; a surviving loss term in ``S`` is an elimination ``S -> SINK``. This
   is invariant because net flux is independent of how internal transfer
   coefficients are split or named.
3. **Observation marker**: which compartment is observed, its transform, error
   kind, and whether a volume scaling is applied.
4. **Canonical labeling**: compartment node identities are anonymized by
   choosing, over all compartment permutations, the lexicographically minimal
   edge/observation serialization. The SINK node is fixed.

The result is hashed (SHA-256) into :attr:`CanonicalForm.key`. Equality of keys
is the equivalence-class primitive used by the distinguishability and
equivalence-reward machinery.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import TYPE_CHECKING

import sympy
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from drrl.spec.model import ModelSpec

_SINK = "@sink"


class CanonicalForm(BaseModel):
    """Reparameterization-invariant structural key for a model.

    Attributes:
        key: SHA-256 hex digest of the canonical serialization.
        n_compartments: Number of compartments.
        edges: Canonical edge list ``(src, dst, type)`` after anonymization,
            where node ids are canonical integers and ``dst`` may be ``-1`` for
            the SINK. Retained for debugging/inspection.
        observation: Canonical observation descriptor.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    n_compartments: int
    edges: tuple[tuple[int, int, str], ...]
    observation: tuple[int, str, str, bool]

    def __eq__(self, other: object) -> bool:
        """Two canonical forms are equal iff their keys match."""
        if not isinstance(other, CanonicalForm):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        """Hash by key."""
        return hash(self.key)


def _dependence_type(
    expr: sympy.Expr, wrt: sympy.Symbol, state_syms: set[sympy.Symbol]
) -> str | None:
    """Classify ``d(expr)/d(wrt)`` structurally.

    Returns ``None`` if the derivative is zero, ``"saturable"`` if it depends on
    any state (nonlinear), else ``"first_order"`` (state-independent / linear).
    """
    deriv = sympy.simplify(sympy.diff(expr, wrt))
    if deriv == 0:
        return None
    if deriv.free_symbols & state_syms:
        return "saturable"
    return "first_order"


def _term_sign(term: sympy.Expr) -> int:
    """Sign of a monomial assuming all symbols are positive (PK convention)."""
    subbed = term.subs({s: sympy.Integer(1) for s in term.free_symbols})
    val = sympy.simplify(subbed)
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def _elimination_edges(
    spec: ModelSpec,
    rhs: dict[str, sympy.Expr],
    state_syms: set[sympy.Symbol],
    name_of: dict[sympy.Symbol, str],
) -> list[tuple[str, str, str]]:
    """Edges (S -> SINK, type) from net-flux loss terms (mass leaving system)."""
    total = sympy.expand(sum(rhs.values(), start=sympy.Integer(0)))
    edges: list[tuple[str, str, str]] = []
    for term in sympy.Add.make_args(total):
        term_states = term.free_symbols & state_syms
        if len(term_states) != 1:
            # Sources (no state) or multi-state couplings are not simple
            # eliminations; Phase 1 models do not produce them.
            continue
        (state_sym,) = tuple(term_states)
        if _term_sign(term) >= 0:
            continue  # a gain, not an elimination
        etype = _dependence_type(term, state_sym, state_syms) or "first_order"
        edges.append((name_of[state_sym], _SINK, etype))
    return edges


def _structural_edges(spec: ModelSpec) -> tuple[list[tuple[str, str, str]], set[str]]:
    """Return (typed edges, compartment-name set) for a spec."""
    rhs = spec.rhs_exprs()
    syms = spec.symbols
    state_syms = {syms[n] for n in spec.state_names}
    name_of = {syms[n]: n for n in spec.state_names}

    edges: list[tuple[str, str, str]] = []
    # Transfer edges: off-diagonal Jacobian dependence S -> X.
    for x in spec.state_names:
        for s in spec.state_names:
            if x == s:
                continue
            etype = _dependence_type(rhs[x], syms[s], state_syms)
            if etype is not None:
                edges.append((s, x, etype))
    # Elimination edges from net flux.
    edges.extend(_elimination_edges(spec, rhs, state_syms, name_of))
    return edges, set(spec.state_names)


def canonicalize(spec: ModelSpec) -> CanonicalForm:
    """Compute the reparameterization-invariant :class:`CanonicalForm`."""
    edges, comp_names = _structural_edges(spec)
    comps = sorted(comp_names)
    n = len(comps)
    observed = spec.observation.state
    obs_desc_base = (
        spec.observation.transform,
        spec.observation.error.kind,
        spec.observation.divide_by is not None,
    )

    best_serial: str | None = None
    best_edges: tuple[tuple[int, int, str], ...] | None = None
    best_obs: tuple[int, str, str, bool] | None = None

    # Canonical labeling: minimize serialization over compartment permutations.
    for perm in itertools.permutations(range(n)):
        idx = {comps[i]: perm[i] for i in range(n)}

        def node_id(name: str, _idx: dict[str, int] = idx) -> int:
            return -1 if name == _SINK else _idx[name]

        mapped_edges = tuple(sorted((node_id(s), node_id(d), t) for (s, d, t) in edges))
        obs = (idx[observed], *obs_desc_base)
        serial = repr((n, mapped_edges, obs))
        if best_serial is None or serial < best_serial:
            best_serial = serial
            best_edges = mapped_edges
            best_obs = obs

    assert best_serial is not None and best_edges is not None and best_obs is not None
    key = hashlib.sha256(best_serial.encode("utf-8")).hexdigest()
    return CanonicalForm(
        key=key, n_compartments=n, edges=best_edges, observation=best_obs
    )
