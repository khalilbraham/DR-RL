"""Open competitor / admissible set assembly (invariant #6).

The structures used for distinguishability and abstention are **never a hardcoded
list**. They are assembled at runtime from three open sources:

1. the policy's own rollouts,
2. a programmatic structure-enumeration prior, and
3. a replay buffer of previously seen structures.

Members are de-duplicated by reparameterization-invariant canonical key, so the
same structure proposed in different coordinates counts once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from drrl.spec import (
    Compartment,
    ModelSpec,
    ObservationModel,
    ODETerm,
    Parameter,
    Unit,
)

_MG = Unit(expr="mg")
_L = Unit(expr="L")
_PERH = Unit(expr="1/h")
_MGPH = Unit(expr="mg/h")


def enumerate_structures() -> Iterator[ModelSpec]:
    """Yield a programmatic prior over candidate PK structures.

    Generated, not hardcoded per problem: a 1-compartment first-order model, a
    2-compartment model, and a 1-compartment Michaelis-Menten model, all dosing
    and observing a central compartment named ``A1`` so a single design applies.
    """
    yield ModelSpec(
        compartments=(Compartment(name="A1", unit=_MG),),
        odes=(ODETerm(target="A1", expr="-(CL/V)*A1"),),
        parameters=(
            Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
            Parameter(name="V", value=10.0, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )
    yield ModelSpec(
        compartments=(
            Compartment(name="A1", unit=_MG),
            Compartment(name="A2", unit=_MG),
        ),
        odes=(
            ODETerm(target="A1", expr="-(CL/V + Q/V)*A1 + (Q/V2)*A2"),
            ODETerm(target="A2", expr="(Q/V)*A1 - (Q/V2)*A2"),
        ),
        parameters=(
            Parameter(name="CL", value=2.0, unit=Unit(expr="L/h")),
            Parameter(name="Q", value=1.0, unit=Unit(expr="L/h")),
            Parameter(name="V", value=10.0, unit=_L),
            Parameter(name="V2", value=20.0, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )
    yield ModelSpec(
        compartments=(Compartment(name="A1", unit=_MG),),
        odes=(ODETerm(target="A1", expr="-Vmax*A1/(Km + A1)"),),
        parameters=(
            Parameter(name="Vmax", value=5.0, unit=_MGPH),
            Parameter(name="Km", value=2.0, unit=_MG),
            Parameter(name="V", value=10.0, unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V"),
    )


@dataclass
class CompetitorSet:
    """A runtime-assembled, de-duplicated set of candidate structures.

    Tracks provenance (which source contributed each canonical key) so the
    openness invariant (#6) is verifiable.
    """

    _by_key: dict[str, ModelSpec] = field(default_factory=dict)
    _source_of: dict[str, str] = field(default_factory=dict)

    def _add(self, spec: ModelSpec, source: str) -> None:
        key = spec.canonicalize().key
        if key not in self._by_key:
            self._by_key[key] = spec
            self._source_of[key] = source

    def add_rollout(self, spec: ModelSpec) -> None:
        """Add a structure proposed by a policy rollout."""
        self._add(spec, "rollout")

    def add_replay(self, spec: ModelSpec) -> None:
        """Add a structure from the replay buffer."""
        self._add(spec, "replay")

    def add_enumeration(self, specs: Iterator[ModelSpec] | None = None) -> None:
        """Add the enumeration prior (defaults to :func:`enumerate_structures`)."""
        for spec in enumerate_structures() if specs is None else specs:
            self._add(spec, "enumeration")

    def members(self) -> list[ModelSpec]:
        """All de-duplicated candidate structures."""
        return list(self._by_key.values())

    def keys(self) -> list[str]:
        """Canonical keys of all members."""
        return list(self._by_key)

    def sources(self) -> dict[str, int]:
        """Count of members contributed by each source."""
        counts: dict[str, int] = {}
        for src in self._source_of.values():
            counts[src] = counts.get(src, 0) + 1
        return counts

    def source_of(self, key: str) -> str | None:
        """The source that first contributed ``key``."""
        return self._source_of.get(key)
