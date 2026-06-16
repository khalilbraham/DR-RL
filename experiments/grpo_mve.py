"""Phase 5 MVE: GRPO training + the manipulation-check ablation.

Trains QLoRA GRPO policies under several reward modes and evaluates each (and the
untrained base model) on the task registry. The headline comparison is the
manipulation check: dropping ``r_identify`` ("no_identify") should let
"fit-but-flat" behaviour return — lower identifiability-aware reward and less
abstention on ambiguous cases than the full reward.

Run:
    uv run python -m experiments.grpo_mve --modes full no_identify --max-steps 60
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import replace
from pathlib import Path

from drrl.rl.decision import Decision
from drrl.rl.evaluate import (
    EvalReport,
    flat_commit_rate,
    generate_decisions,
    score_decisions,
)
from drrl.rl.grpo import GRPOSettings, make_reward_funcs, train_grpo
from drrl.rl.tasks import TaskRegistry, build_registry
from drrl.sim import SimConfig, get_backend
from drrl.utils.config import config_hash, load_config
from drrl.utils.logging import configure_logging, get_logger
from drrl.utils.manifest import build_manifest
from drrl.utils.seed import set_seed

_REPO = Path(__file__).resolve().parents[1]
log = get_logger("grpo_mve")


def _completion(decision: Decision) -> str:
    if not decision.valid:
        return "ANSWER: (none)"
    if decision.abstain:
        return "ANSWER: abstain"
    return f"ANSWER: {decision.structure}"


def _mean_full_reward(decisions: dict[str, Decision], registry: TaskRegistry) -> float:
    """Identifiability-aware score: mean FULL reward of the policy's decisions."""
    backend = get_backend("scipy", SimConfig(rtol=1e-8, atol=1e-10))
    reward = make_reward_funcs(registry, mode="full", backend=backend)[0]
    totals = []
    for cid in registry.ids():
        case = registry.get(cid)
        totals.append(
            reward([case.prompt], [_completion(decisions[cid])], case_id=[cid])[0]
        )
    return sum(totals) / len(totals) if totals else 0.0


def _evaluate(
    registry: TaskRegistry, model_name: str, adapter_dir: str | None
) -> dict[str, object]:
    decisions = generate_decisions(
        registry, model_name=model_name, adapter_dir=adapter_dir
    )
    report: EvalReport = score_decisions(decisions, registry)
    backend = get_backend("scipy", SimConfig(rtol=1e-8, atol=1e-10))
    return {
        **dataclasses.asdict(report),
        "mean_full_reward": _mean_full_reward(decisions, registry),
        # Identifiability-aware: committed a practically non-identifiable model.
        "flat_commit_identify": flat_commit_rate(decisions, registry, backend),
        "decisions": {cid: _completion(d) for cid, d in decisions.items()},
    }


def main() -> Path:
    """Run the GRPO ablation and write a results JSON + manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", nargs="+", default=["full", "no_identify", "fit_only"]
    )
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    configure_logging(level="INFO")
    cfg = load_config(_REPO / "configs" / "grpo" / "default.yaml")
    set_seed(int(cfg.seed))

    base = GRPOSettings(
        model_name=str(cfg.model_name),
        num_generations=int(cfg.num_generations),
        beta=float(cfg.beta),
        learning_rate=float(cfg.learning_rate),
        max_steps=int(args.max_steps if args.max_steps is not None else cfg.max_steps),
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        max_completion_length=int(cfg.max_completion_length),
        temperature=float(cfg.temperature),
        lora_r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        diversity_weight=float(cfg.diversity_weight),
        dataset_repeats=int(cfg.dataset_repeats),
        seed=int(cfg.seed),
    )
    registry = build_registry()
    out_dir = _REPO / "outputs" / "grpo_mve"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}
    log.info("evaluating untrained base model %s", base.model_name)
    results["base"] = _evaluate(registry, base.model_name, None)

    for mode in args.modes:
        adapter_dir = str(out_dir / mode)
        settings = replace(base, reward_mode=mode, output_dir=adapter_dir)
        log.info("training mode=%s steps=%d", mode, settings.max_steps)
        train_grpo(settings, registry)
        log.info("evaluating mode=%s", mode)
        results[mode] = _evaluate(registry, base.model_name, adapter_dir)

    manifest = build_manifest(
        seed=int(cfg.seed),
        config_hash=config_hash(cfg),
        repo_root=_REPO,
        extra={"experiment": "grpo_mve", "modes": ",".join(args.modes)},
    )
    payload = {"manifest": json.loads(manifest.to_json()), "results": results}
    path = out_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote results -> %s", path)
    _log_manipulation_check(results)
    return path


def _log_manipulation_check(results: dict[str, object]) -> None:
    """Log the headline comparison (full vs no_identify)."""

    def metric(mode: str, key: str) -> object:
        r = results.get(mode)
        return r.get(key) if isinstance(r, dict) else None

    log.info("=== manipulation check (E-IRL vs ablations) ===")
    for mode in ("base", "full", "no_identify", "fit_only"):
        if mode in results:
            log.info(
                "%-12s full_reward=%s flat_commit_identify=%s abstain=%s commit_acc=%s",
                mode,
                metric(mode, "mean_full_reward"),
                metric(mode, "flat_commit_identify"),
                metric(mode, "abstain_rate_on_abstain_cases"),
                metric(mode, "commit_accuracy"),
            )


if __name__ == "__main__":
    main()
