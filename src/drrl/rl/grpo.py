"""GRPO over single-turn model-selection tasks, behind a thin TRL interface.

The verifier-grounded reward and the diversity bonus are first-party and
torch-free (so they are unit-tested on CPU); only :func:`train_grpo` lazily
imports torch/TRL/peft. Reward modes implement the manipulation-check ablation:
``full`` vs ``no_identify`` (drop ``r_identify``) vs ``fit_only`` (pure fit).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from drrl.env.state import ModelState, build_spec
from drrl.reward.compose import compose_reward
from drrl.reward.context import RewardContext, RewardWeights
from drrl.rl.decision import Decision, parse_completion
from drrl.rl.tasks import TaskCase, TaskRegistry
from drrl.sim.backend import Backend
from drrl.spec import Design, Dose, Unit
from drrl.verifier.competitors import CompetitorSet, admissible_set
from drrl.verifier.pipeline import verify

RewardMode = Literal["full", "no_identify", "fit_only"]

_PROBE = Design(
    doses=(Dose(compartment="A1", amount=300.0, unit=Unit(expr="mg")),),
    sample_times=(0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0),
)


def weights_for_mode(mode: RewardMode, base: RewardWeights) -> RewardWeights:
    """Return the reward weights for an ablation mode.

    ``full`` keeps everything; ``no_identify`` zeroes ``w_identify`` (the
    manipulation check); ``fit_only`` keeps only fit/PK (the weak baseline).
    """
    if mode == "full":
        return base
    if mode == "no_identify":
        return replace(base, w_identify=0.0)
    if mode == "fit_only":
        return replace(
            base,
            w_eq=0.0,
            w_identify=0.0,
            w_abstain=0.0,
            w_parsimony=0.0,
            w_expl=0.0,
        )
    raise ValueError(f"unknown reward mode {mode!r}")


def _decision_reward(
    decision: Decision,
    case: TaskCase,
    weights: RewardWeights,
    backend: Backend,
    *,
    sigma_floor: float,
) -> float:
    """Verifier-grounded reward for a parsed decision on a task case."""
    if not decision.valid:
        return 0.0
    battery = list(case.hidden_battery)
    competitors = CompetitorSet()
    competitors.add_enumeration()
    admissible = admissible_set(
        case.reference, competitors, battery, backend, sigma_floor=sigma_floor
    )

    if decision.abstain:
        candidate = build_spec(ModelState.initial("one_compartment"))
        action_proposed: Design | None = _PROBE
        action = "abstain"
    else:
        assert decision.structure is not None
        candidate = build_spec(ModelState.initial(decision.structure))
        action_proposed = None
        action = "commit"

    report = verify(candidate, case.observed_design, backend, hidden_battery=battery)
    ctx = RewardContext(
        candidate=candidate,
        reference=case.reference,
        hidden_battery=tuple(battery),
        observed_designs=(case.observed_design,),
        action=action,
        proposed_design=action_proposed,
        admissible_keys=admissible,
    )
    return compose_reward(report, ctx, weights, backend, sigma_floor=sigma_floor).total


def _structure_key(decision: Decision) -> str:
    """A canonical bucket for the diversity bonus."""
    if not decision.valid:
        return "@invalid"
    if decision.abstain:
        return "@abstain"
    assert decision.structure is not None
    return build_spec(ModelState.initial(decision.structure)).canonicalize().key


def make_reward_funcs(
    registry: TaskRegistry,
    *,
    mode: RewardMode = "full",
    base_weights: RewardWeights | None = None,
    backend: Backend | None = None,
    diversity_weight: float = 0.1,
    sigma_floor: float = 1e-9,
) -> list[Callable[..., list[float]]]:
    """Build the TRL ``reward_funcs`` list (verifier reward + diversity bonus).

    Both are plain callables ``(prompts, completions, **kwargs) -> list[float]``;
    TRL sums them. ``case_id`` arrives as a dataset-column kwarg.
    """
    weights = weights_for_mode(mode, base_weights or RewardWeights())
    sim_backend = backend
    if sim_backend is None:
        from drrl.sim import SimConfig, get_backend

        sim_backend = get_backend("scipy", SimConfig(rtol=1e-8, atol=1e-10))

    def verifier_reward(
        prompts: list[str], completions: list[str], **kwargs: Any
    ) -> list[float]:
        case_ids = kwargs["case_id"]
        out: list[float] = []
        for completion, cid in zip(completions, case_ids, strict=True):
            decision = parse_completion(_as_text(completion))
            out.append(
                _decision_reward(
                    decision,
                    registry.get(cid),
                    weights,
                    sim_backend,
                    sigma_floor=sigma_floor,
                )
            )
        return out

    def diversity_reward(
        prompts: list[str], completions: list[str], **kwargs: Any
    ) -> list[float]:
        return _group_diversity(
            prompts,
            [_structure_key(parse_completion(_as_text(c))) for c in completions],
            weight=diversity_weight,
        )

    return [verifier_reward, diversity_reward]


def _as_text(completion: Any) -> str:
    """Normalize a TRL completion (str or chat message list) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def _group_diversity(
    prompts: list[str], keys: list[str], *, weight: float
) -> list[float]:
    """Per-group entropy bonus over decision buckets (encourages enumeration)."""
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(prompts):
        groups.setdefault(p, []).append(i)
    bonus = [0.0] * len(prompts)
    for idxs in groups.values():
        counts: dict[str, int] = {}
        for i in idxs:
            counts[keys[i]] = counts.get(keys[i], 0) + 1
        n = len(idxs)
        entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
        norm = entropy / math.log(n) if n > 1 else 0.0
        for i in idxs:
            bonus[i] = weight * norm
    return bonus


@dataclass(frozen=True)
class GRPOSettings:
    """GRPO training settings (populated from ``configs/grpo``)."""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "outputs/grpo"
    reward_mode: RewardMode = "full"
    num_generations: int = 6
    beta: float = 0.04
    learning_rate: float = 1e-5
    max_steps: int = 100
    per_device_train_batch_size: int = 6
    gradient_accumulation_steps: int = 2
    max_completion_length: int = 96
    temperature: float = 0.9
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    diversity_weight: float = 0.1
    dataset_repeats: int = 40
    seed: int = 0
    sigma_floor: float = 1e-9


def train_grpo(
    settings: GRPOSettings, registry: TaskRegistry
) -> str:  # pragma: no cover
    """Train a QLoRA GRPO policy. Lazily imports torch/TRL/peft.

    GPU integration code (exercised by the real run, not the unit suite).

    Returns:
        The output directory containing the saved LoRA adapter.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    from drrl.utils.seed import set_seed

    set_seed(settings.seed)

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Turing: fp16, not bf16
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name, quantization_config=quant, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    rows = _dataset_rows(registry, settings.dataset_repeats)
    dataset = Dataset.from_list(rows)
    reward_funcs = make_reward_funcs(
        registry,
        mode=settings.reward_mode,
        diversity_weight=settings.diversity_weight,
        sigma_floor=settings.sigma_floor,
    )

    cfg = GRPOConfig(
        output_dir=settings.output_dir,
        num_generations=settings.num_generations,
        beta=settings.beta,
        learning_rate=settings.learning_rate,
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        max_completion_length=settings.max_completion_length,
        temperature=settings.temperature,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        seed=settings.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(settings.output_dir)
    tokenizer.save_pretrained(settings.output_dir)
    return settings.output_dir


def _dataset_rows(registry: TaskRegistry, repeats: int) -> list[dict[str, str]]:
    from drrl.rl.tasks import to_dataset_rows

    return to_dataset_rows(registry, repeats=repeats)
