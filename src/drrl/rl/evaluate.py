"""Greedy evaluation of a policy and the manipulation-check metrics.

``score_decisions`` is torch-free (testable on CPU with hand-built decisions);
``generate_decisions`` lazily loads the model and samples greedily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drrl.rl.decision import Decision, parse_completion
from drrl.rl.tasks import TaskRegistry


@dataclass(frozen=True)
class EvalReport:
    """Aggregate evaluation metrics.

    Attributes:
        commit_accuracy: Fraction of commit cases where the correct structure was
            committed.
        abstain_rate_on_abstain_cases: Fraction of abstain cases where the policy
            abstained (higher is better).
        flat_commit_rate: Fraction of abstain cases where the policy *committed* a
            structure instead of abstaining (the "fit-but-flat" failure).
        correct_action_rate: Fraction of all cases with the correct commit/abstain
            action.
        n_cases: Number of cases evaluated.
    """

    commit_accuracy: float
    abstain_rate_on_abstain_cases: float
    flat_commit_rate: float
    correct_action_rate: float
    n_cases: int


def score_decisions(
    decisions: dict[str, Decision], registry: TaskRegistry
) -> EvalReport:
    """Compute evaluation metrics from per-case decisions."""
    commit_cases = [c for c in registry.cases.values() if c.correct_action == "commit"]
    abstain_cases = [
        c for c in registry.cases.values() if c.correct_action == "abstain"
    ]

    def _mean(xs: list[bool]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    commit_acc = _mean(
        [decisions[c.case_id].structure == c.correct_structure for c in commit_cases]
    )
    abstain_rate = _mean([decisions[c.case_id].abstain for c in abstain_cases])
    flat_commit = _mean(
        [
            decisions[c.case_id].valid and not decisions[c.case_id].abstain
            for c in abstain_cases
        ]
    )
    correct_action = _mean(
        [
            (
                decisions[c.case_id].abstain
                if c.correct_action == "abstain"
                else decisions[c.case_id].structure == c.correct_structure
            )
            for c in registry.cases.values()
        ]
    )
    return EvalReport(
        commit_accuracy=commit_acc,
        abstain_rate_on_abstain_cases=abstain_rate,
        flat_commit_rate=flat_commit,
        correct_action_rate=correct_action,
        n_cases=len(registry.cases),
    )


def flat_commit_rate(
    decisions: dict[str, Decision],
    registry: TaskRegistry,
    backend: object,
    *,
    rank_rtol: float = 1e-2,
    threshold: float = 0.95,
    sigma_floor: float = 1e-9,
) -> float:
    """Fraction of ambiguous cases where the policy commits a *practically flat* model.

    For each abstain/ambiguous case, if the policy committed a structure, we
    best-fit that structure to the case data and check whether its parameters are
    practically identifiable on the held-out battery. Committing a structure whose
    parameters are non-identifiable is the "fit-but-flat" failure.
    """
    from drrl.env.state import ModelState, build_spec
    from drrl.sim.fitting import fit_params
    from drrl.verifier.identify import identifiability

    cases = [c for c in registry.cases.values() if c.correct_action == "abstain"]
    if not cases:
        return 0.0
    flats = 0
    for case in cases:
        d = decisions[case.case_id]
        if d.abstain or not d.valid or d.structure is None:
            continue
        template = build_spec(ModelState.initial(d.structure))
        target = backend.simulate(case.reference, case.observed_design).observed
        fitted, _ = fit_params(template, target, case.observed_design, backend)
        frac = identifiability(
            fitted,
            list(case.hidden_battery),
            backend,
            rank_rtol=rank_rtol,
            sigma_floor=sigma_floor,
        ).identifiable_fraction
        if frac < threshold:
            flats += 1
    return flats / len(cases)


def generate_decisions(  # pragma: no cover
    registry: TaskRegistry,
    *,
    model_name: str,
    adapter_dir: str | None = None,
    max_new_tokens: int = 96,
) -> dict[str, Decision]:
    """Greedily generate one decision per case (lazy torch import).

    GPU integration code (exercised by the real run, not the unit suite).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map={"": 0}
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    decisions: dict[str, Decision] = {}
    for cid in registry.ids():
        case = registry.get(cid)
        messages = [{"role": "user", "content": case.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        decisions[cid] = parse_completion(completion)
    return decisions
