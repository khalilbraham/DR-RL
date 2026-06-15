"""Typed edit actions for the model-edit environment.

Actions are a closed, typed union. Structural edits transform the candidate;
terminal actions end the episode by committing to the current structure or
abstaining and proposing an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from drrl.spec.model import Design


class EditError(ValueError):
    """Raised when an edit is invalid for the current model state."""


@dataclass(frozen=True)
class AddCompartment:
    """Add a peripheral compartment (1-compartment -> 2-compartment)."""


@dataclass(frozen=True)
class RemoveCompartment:
    """Remove the peripheral compartment (2-compartment -> 1-compartment)."""


@dataclass(frozen=True)
class SwapKinetics:
    """Swap central elimination kinetics (first-order <-> Michaelis-Menten)."""


@dataclass(frozen=True)
class TuneParam:
    """Scale a named parameter by ``factor`` (a continuous fit edit)."""

    name: str
    factor: float


@dataclass(frozen=True)
class Commit:
    """Terminal: commit to the current model structure."""


@dataclass(frozen=True)
class Abstain:
    """Terminal: decline to commit and propose a discriminating experiment."""

    proposed_design: Design


# Structural (non-terminal) edits.
EditAction = AddCompartment | RemoveCompartment | SwapKinetics | TuneParam
# Terminal actions.
TerminalAction = Commit | Abstain
# Any action.
Action = EditAction | TerminalAction


def is_terminal(action: Action) -> bool:
    """Whether ``action`` ends the episode."""
    return isinstance(action, Commit | Abstain)
