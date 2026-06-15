"""Parse an LLM completion into a typed commit/abstain decision.

The policy is asked to end its answer with ``ANSWER: <structure>`` or
``ANSWER: abstain``. Parsing is lenient (accepts common synonyms) but a missing
or unrecognized answer yields an *invalid* decision, which the reward treats as a
Tier-A failure (zero reward) — there is no partial credit for unparseable output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from drrl.env.state import Structure

_STRUCTURE_SYNONYMS: dict[str, Structure] = {
    "one_compartment": "one_compartment",
    "one-compartment": "one_compartment",
    "1-compartment": "one_compartment",
    "1c": "one_compartment",
    "monocompartment": "one_compartment",
    "two_compartment": "two_compartment",
    "two-compartment": "two_compartment",
    "2-compartment": "two_compartment",
    "2c": "two_compartment",
    "bicompartment": "two_compartment",
    "michaelis_menten": "michaelis_menten",
    "michaelis-menten": "michaelis_menten",
    "michaelis": "michaelis_menten",
    "mm": "michaelis_menten",
    "saturable": "michaelis_menten",
}

_ANSWER_RE = re.compile(r"answer\s*[:=]\s*([a-z0-9_\- ]+)", re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    """A parsed terminal decision.

    Attributes:
        structure: The committed structure, or ``None`` if abstaining/invalid.
        abstain: Whether the policy abstained.
        valid: Whether the completion contained a recognizable answer.
    """

    structure: Structure | None
    abstain: bool
    valid: bool

    @staticmethod
    def invalid() -> Decision:
        """An unparseable decision (no recognizable answer)."""
        return Decision(structure=None, abstain=False, valid=False)


def parse_completion(text: str) -> Decision:
    """Parse a completion into a :class:`Decision`.

    Args:
        text: The model completion.

    Returns:
        A :class:`Decision`; ``valid=False`` if no answer was recognized.
    """
    matches = _ANSWER_RE.findall(text)
    token = matches[-1].strip().lower() if matches else ""
    if not token:
        # Fall back to scanning the whole completion for a single clear keyword.
        token = text.strip().lower()
    if "abstain" in token:
        return Decision(structure=None, abstain=True, valid=True)
    # Longest synonyms first so "2-compartment" wins over a stray "2".
    for key in sorted(_STRUCTURE_SYNONYMS, key=len, reverse=True):
        if key in token:
            return Decision(
                structure=_STRUCTURE_SYNONYMS[key], abstain=False, valid=True
            )
    return Decision.invalid()
