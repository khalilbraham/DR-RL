"""The four MVE GRPO tasks and their verifier-grounded cases.

Each task reduces to a single-turn decision (commit a structure or abstain),
graded by the distinguishability-relative reward against a hidden reference and
battery. The four task *types* differ in their prompt framing:

1. ``desc_to_ode``      — pick the structure implied by a mechanism description.
2. ``sim_matching``     — pick the structure consistent with a data summary.
3. ``identifiability``  — commit only if the data determine the structure, else abstain.
4. ``abstention``       — ambiguous data: abstain and propose an experiment.

The deliberately ambiguous cases use a Michaelis-Menten reference dosed *below
saturation*, which is observationally indistinguishable from a linear model — so
the correct action is to abstain and request a saturating experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from drrl.data.synth.library import (
    michaelis_menten,
    one_compartment,
    two_compartment,
)
from drrl.env.state import Structure
from drrl.spec import Design, Dose, ModelSpec, Unit

TaskType = Literal["desc_to_ode", "sim_matching", "identifiability", "abstention"]
CorrectAction = Literal["commit", "abstain"]
_MG = Unit(expr="mg")


def _design(amount: float, times: tuple[float, ...]) -> Design:
    return Design(
        doses=(Dose(compartment="A1", amount=amount, unit=_MG),), sample_times=times
    )


@dataclass(frozen=True)
class TaskCase:
    """A single GRPO task instance with its (hidden) grading material.

    Attributes:
        case_id: Unique id (carried in the dataset; resolves to this object).
        task_type: One of the four task framings.
        prompt: The text shown to the policy.
        reference: The data-generating model (hidden from the agent).
        observed_design: Design the agent is told about.
        hidden_battery: Held-out designs used for fit/identifiability/abstention.
        correct_action: ``"commit"`` or ``"abstain"`` (for evaluation only).
        correct_structure: The structure to commit (``None`` for abstain cases).
    """

    case_id: str
    task_type: TaskType
    prompt: str
    reference: ModelSpec
    observed_design: Design
    hidden_battery: tuple[Design, ...]
    correct_action: CorrectAction
    correct_structure: Structure | None


_RICH_TIMES = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 18.0, 24.0)
_SPARSE_TIMES = (0.5, 2.0, 8.0)
_INSTRUCTION = (
    "You are a pharmacometrician. Choose the simplest compartmental model the "
    "data can justify. Reply on the last line with exactly "
    "'ANSWER: one_compartment', 'ANSWER: two_compartment', "
    "'ANSWER: michaelis_menten', or 'ANSWER: abstain' (abstain if the data "
    "cannot determine the model and a new experiment is needed)."
)


def _prompt(scenario: str) -> str:
    return f"{_INSTRUCTION}\n\nScenario: {scenario}\n"


def _commit_cases() -> list[TaskCase]:
    rich = _design(100.0, _RICH_TIMES)
    hidden = (_design(50.0, _RICH_TIMES), _design(150.0, _RICH_TIMES))
    return [
        TaskCase(
            case_id="commit_1c",
            task_type="desc_to_ode",
            prompt=_prompt(
                "After an IV bolus the log-concentration declines as a single "
                "straight line (mono-exponential), and AUC is dose-proportional."
            ),
            reference=one_compartment(),
            observed_design=rich,
            hidden_battery=hidden,
            correct_action="commit",
            correct_structure="one_compartment",
        ),
        TaskCase(
            case_id="commit_2c",
            task_type="sim_matching",
            prompt=_prompt(
                "After an IV bolus the concentration shows a rapid initial "
                "distribution phase followed by a slower terminal phase "
                "(clearly bi-exponential on a log scale)."
            ),
            reference=two_compartment(),
            observed_design=rich,
            hidden_battery=hidden,
            correct_action="commit",
            correct_structure="two_compartment",
        ),
        TaskCase(
            case_id="commit_mm_high",
            task_type="identifiability",
            prompt=_prompt(
                "At a high IV dose the drug is eliminated slowly at first and "
                "then much faster as concentrations fall: clearance increases as "
                "concentration drops (clear saturable/Michaelis-Menten kinetics "
                "are visible across the curve)."
            ),
            reference=michaelis_menten(),
            observed_design=_design(200.0, _RICH_TIMES),
            hidden_battery=(_design(150.0, _RICH_TIMES), _design(250.0, _RICH_TIMES)),
            correct_action="commit",
            correct_structure="michaelis_menten",
        ),
    ]


def _abstain_cases() -> list[TaskCase]:
    # Michaelis-Menten dosed far below Km: indistinguishable from a linear model,
    # so the structure cannot be determined without a saturating experiment.
    low_mm = michaelis_menten(vmax=8.0, km=50.0, v=10.0)
    low_battery = (_design(5.0, _RICH_TIMES), _design(8.0, _RICH_TIMES))
    return [
        TaskCase(
            case_id="abstain_mm_low",
            task_type="abstention",
            prompt=_prompt(
                "Only low IV doses were given. The decline looks mono-exponential, "
                "but the dose was far below any saturation level, so linear and "
                "saturable (Michaelis-Menten) models fit the data equally well."
            ),
            reference=low_mm,
            observed_design=_design(5.0, _SPARSE_TIMES),
            hidden_battery=low_battery,
            correct_action="abstain",
            correct_structure=None,
        ),
        TaskCase(
            case_id="abstain_sparse",
            task_type="identifiability",
            prompt=_prompt(
                "Only three samples were collected at a single low dose, and they "
                "are consistent with several models; the experiment is too sparse "
                "to determine the structure."
            ),
            reference=michaelis_menten(vmax=6.0, km=80.0, v=12.0),
            observed_design=_design(5.0, _SPARSE_TIMES),
            hidden_battery=(_design(4.0, _SPARSE_TIMES), _design(6.0, _SPARSE_TIMES)),
            correct_action="abstain",
            correct_structure=None,
        ),
    ]


@dataclass
class TaskRegistry:
    """In-memory registry mapping case ids to full :class:`TaskCase` objects.

    The HF dataset only carries ``case_id`` + ``prompt`` (serializable); the
    reward function resolves the rest here.
    """

    cases: dict[str, TaskCase] = field(default_factory=dict)

    def add(self, case: TaskCase) -> None:
        """Register a case."""
        self.cases[case.case_id] = case

    def get(self, case_id: str) -> TaskCase:
        """Resolve a case id."""
        return self.cases[case_id]

    def ids(self) -> list[str]:
        """All registered case ids."""
        return list(self.cases)


def build_registry() -> TaskRegistry:
    """Build the MVE task registry (commit + abstain cases across the four tasks)."""
    reg = TaskRegistry()
    for case in _commit_cases() + _abstain_cases():
        reg.add(case)
    return reg


def to_dataset_rows(
    registry: TaskRegistry, *, repeats: int = 1
) -> list[dict[str, str]]:
    """Flatten the registry into ``[{prompt, case_id}, ...]`` rows for training."""
    rows: list[dict[str, str]] = []
    for _ in range(repeats):
        for cid in registry.ids():
            case = registry.get(cid)
            rows.append({"prompt": case.prompt, "case_id": cid})
    return rows
