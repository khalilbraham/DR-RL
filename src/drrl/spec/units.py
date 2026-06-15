"""Unit handling built on a single shared ``pint`` registry.

A model is dimensionally meaningful only against one registry; mixing
registries makes quantities incomparable. We expose exactly one ``UREG`` and a
serializable :class:`Unit` value object whose string is validated parseable at
construction, so a malformed unit fails fast at the schema boundary rather than
deep inside dimensional analysis (verifier, Phase 2).
"""

from __future__ import annotations

from typing import Any

import pint
from pydantic import BaseModel, ConfigDict, field_validator

# The one and only registry. Importers must use this instance. Typed as Any
# because pint's UnitRegistry generic parameters are an implementation detail.
UREG: Any = pint.UnitRegistry()
# µmol/L convenience: pint already knows micromole and liter.


def parse_unit(expr: str) -> Any:
    """Parse a unit string into a ``pint.Unit`` on the shared registry.

    Args:
        expr: A pint-parseable unit expression, e.g. ``"mg/L"``, ``"1/h"``,
            ``"dimensionless"``.

    Returns:
        The parsed unit.

    Raises:
        ValueError: If ``expr`` is not a valid unit on :data:`UREG`.
    """
    try:
        return UREG.Unit(expr)
    except (pint.errors.UndefinedUnitError, AssertionError, ValueError) as exc:
        raise ValueError(f"invalid unit {expr!r}: {exc}") from exc


class Unit(BaseModel):
    """A serializable, validated physical unit.

    Attributes:
        expr: The pint-parseable unit string (the canonical serialization).
    """

    model_config = ConfigDict(frozen=True)

    expr: str

    @field_validator("expr")
    @classmethod
    def _validate_parseable(cls, v: str) -> str:
        parse_unit(v)  # raises ValueError if invalid
        return v

    @property
    def pint(self) -> Any:
        """Return the parsed ``pint.Unit`` (computed on access)."""
        return parse_unit(self.expr)

    @property
    def dimensionality(self) -> Any:
        """Return the pint dimensionality (e.g. ``[mass] / [length] ** 3``)."""
        return self.pint.dimensionality

    def same_dimension(self, other: Unit) -> bool:
        """Whether ``self`` and ``other`` share a physical dimension."""
        return bool(self.dimensionality == other.dimensionality)

    def __str__(self) -> str:
        """Return the unit string."""
        return self.expr
