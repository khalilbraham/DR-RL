"""Tier-A gate: grammar/schema validity.

The policy emits candidate models as JSON/dicts. This layer turns a raw
candidate into a validated :class:`~drrl.spec.model.ModelSpec`, capturing any
validation failure as data (a :class:`SchemaReport`) rather than raising — a
malformed proposal is a learning signal, not a crash.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from drrl.spec.model import ModelSpec
from drrl.verifier.report import SchemaReport


def check_schema(
    candidate: dict[str, Any] | str,
) -> tuple[SchemaReport, ModelSpec | None]:
    """Validate a candidate model against the schema.

    Args:
        candidate: A dict or JSON string describing a model.

    Returns:
        ``(report, spec)`` where ``spec`` is the parsed :class:`ModelSpec` when
        valid, else ``None``.
    """
    try:
        spec = (
            ModelSpec.model_validate_json(candidate)
            if isinstance(candidate, str)
            else ModelSpec.model_validate(candidate)
        )
    except ValidationError as exc:
        msgs = tuple(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return SchemaReport(ok=False, errors=msgs or ("invalid model",)), None
    except (ValueError, TypeError) as exc:
        return SchemaReport(ok=False, errors=(str(exc),)), None
    return SchemaReport(ok=True), spec
