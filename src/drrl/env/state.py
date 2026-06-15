"""Model state for the edit environment and its transition rules.

State is a structure label plus the current parameter values. ``build_spec``
materializes a :class:`ModelSpec`; ``apply_edit`` implements the typed structural
transitions over the MVE structure space.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from drrl.env.actions import (
    AddCompartment,
    EditAction,
    EditError,
    RemoveCompartment,
    SwapKinetics,
    TuneParam,
)
from drrl.spec import (
    Compartment,
    ErrorModel,
    ModelSpec,
    ObservationModel,
    ODETerm,
    Parameter,
    Unit,
)

Structure = Literal["one_compartment", "two_compartment", "michaelis_menten"]

_MG = Unit(expr="mg")
_L = Unit(expr="L")
_LPH = Unit(expr="L/h")
_PERH = Unit(expr="1/h")
_MGPH = Unit(expr="mg/h")
_PROP = ErrorModel(kind="proportional", sigma_prop=0.1)

DEFAULT_PARAMS: dict[Structure, dict[str, float]] = {
    "one_compartment": {"CL": 2.0, "V": 10.0},
    "two_compartment": {"k10": 0.2, "k12": 0.1, "k21": 0.05, "V1": 10.0},
    "michaelis_menten": {"Vmax": 8.0, "Km": 3.0, "V": 10.0},
}


@dataclass(frozen=True)
class ModelState:
    """The editable state: a structure label and its parameter values.

    Attributes:
        structure: One of the MVE structure labels.
        params: Current parameter values for ``structure``.
    """

    structure: Structure
    params: dict[str, float]

    @staticmethod
    def initial(structure: Structure = "one_compartment") -> ModelState:
        """A fresh state at ``structure`` with default parameters."""
        return ModelState(structure=structure, params=dict(DEFAULT_PARAMS[structure]))


def build_spec(state: ModelState) -> ModelSpec:
    """Materialize a :class:`ModelSpec` from a :class:`ModelState`."""
    p = state.params
    if state.structure == "one_compartment":
        return ModelSpec(
            compartments=(Compartment(name="A1", unit=_MG),),
            odes=(ODETerm(target="A1", expr="-(CL/V)*A1"),),
            parameters=(
                Parameter(name="CL", value=p["CL"], unit=_LPH),
                Parameter(name="V", value=p["V"], unit=_L),
            ),
            observation=ObservationModel(state="A1", divide_by="V", error=_PROP),
        )
    if state.structure == "michaelis_menten":
        return ModelSpec(
            compartments=(Compartment(name="A1", unit=_MG),),
            odes=(ODETerm(target="A1", expr="-Vmax*A1/(Km + A1)"),),
            parameters=(
                Parameter(name="Vmax", value=p["Vmax"], unit=_MGPH),
                Parameter(name="Km", value=p["Km"], unit=_MG),
                Parameter(name="V", value=p["V"], unit=_L),
            ),
            observation=ObservationModel(state="A1", divide_by="V", error=_PROP),
        )
    return ModelSpec(
        compartments=(
            Compartment(name="A1", unit=_MG),
            Compartment(name="A2", unit=_MG),
        ),
        odes=(
            ODETerm(target="A1", expr="-(k10+k12)*A1 + k21*A2"),
            ODETerm(target="A2", expr="k12*A1 - k21*A2"),
        ),
        parameters=(
            Parameter(name="k10", value=p["k10"], unit=_PERH),
            Parameter(name="k12", value=p["k12"], unit=_PERH),
            Parameter(name="k21", value=p["k21"], unit=_PERH),
            Parameter(name="V1", value=p["V1"], unit=_L),
        ),
        observation=ObservationModel(state="A1", divide_by="V1", error=_PROP),
    )


def valid_structural_edits(structure: Structure) -> list[EditAction]:
    """Structural edits valid from ``structure`` (excludes parameter tuning)."""
    if structure == "one_compartment":
        return [AddCompartment(), SwapKinetics()]
    if structure == "two_compartment":
        return [RemoveCompartment()]
    return [SwapKinetics()]  # michaelis_menten -> one_compartment


def apply_edit(state: ModelState, action: EditAction) -> ModelState:
    """Apply a structural/parameter edit, returning the new state.

    Raises:
        EditError: If the edit is invalid for ``state.structure``.
    """
    if isinstance(action, AddCompartment):
        if state.structure != "one_compartment":
            raise EditError("AddCompartment requires a 1-compartment model")
        return ModelState.initial("two_compartment")
    if isinstance(action, RemoveCompartment):
        if state.structure != "two_compartment":
            raise EditError("RemoveCompartment requires a 2-compartment model")
        return ModelState.initial("one_compartment")
    if isinstance(action, SwapKinetics):
        if state.structure == "one_compartment":
            return ModelState.initial("michaelis_menten")
        if state.structure == "michaelis_menten":
            return ModelState.initial("one_compartment")
        raise EditError("SwapKinetics is only defined for 1-compartment models")
    if isinstance(action, TuneParam):
        if action.name not in state.params:
            raise EditError(f"unknown parameter {action.name!r} for {state.structure}")
        new_params = dict(state.params)
        new_params[action.name] = new_params[action.name] * action.factor
        return replace(state, params=new_params)
    raise EditError(f"unsupported edit action: {action!r}")
