"""Phase 6 PRM-shift experiment: does PRM-guided edit search shift the
terminal-reward distribution vs a one-shot commit?

We do NOT claim potential-based policy invariance for the (verifier-derived)
PRM. Instead we measure its effect: across references, starting from a wrong
initial structure, compare the terminal reward of (a) a one-shot commit and
(b) best-first edit search guided by the PRM. A rightward shift demonstrates the
PRM's value for inference-time multi-turn repair.

Run: uv run python -m experiments.prm_shift
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from omegaconf import OmegaConf

from drrl.data.synth.library import (
    michaelis_menten,
    one_compartment,
    two_compartment,
)
from drrl.env import ModelEditEnv
from drrl.reward import RewardWeights
from drrl.rl import ProcessRewardModel, best_first_repair, one_shot_commit
from drrl.sim import SimConfig, get_backend
from drrl.spec import Design, Dose, ModelSpec, Unit
from drrl.utils.config import config_hash
from drrl.utils.logging import configure_logging, get_logger
from drrl.utils.manifest import build_manifest
from drrl.utils.seed import set_seed

_REPO = Path(__file__).resolve().parents[1]
_MG = Unit(expr="mg")
log = get_logger("prm_shift")


def _env(reference: ModelSpec, initial: str) -> ModelEditEnv:
    obs = Design(
        doses=(Dose(compartment="A1", amount=100.0, unit=_MG),),
        sample_times=(0.5, 1.0, 2.0, 4.0, 8.0),
    )
    hidden = [
        Design(
            doses=(Dose(compartment="A1", amount=a, unit=_MG),),
            sample_times=(0.25, 0.75, 1.5, 3.0, 6.0),
        )
        for a in (50.0, 150.0)
    ]
    return ModelEditEnv(
        reference,
        obs,
        hidden,
        get_backend("scipy", SimConfig(rtol=1e-9, atol=1e-12)),
        RewardWeights(),
        max_turns=6,
        initial_structure=initial,  # type: ignore[arg-type]
    )


def main() -> Path:
    """Run the PRM-shift experiment and write a results JSON + manifest."""
    configure_logging(level="INFO")
    set_seed(0)
    prm = ProcessRewardModel()

    # References whose correct structure differs from the (wrong) start = 1c.
    references = {
        "two_compartment": two_compartment(),
        "michaelis_menten": michaelis_menten(vmax=8.0, km=3.0),
        "one_compartment": one_compartment(),  # already correct (control)
    }

    rows = []
    for name, ref in references.items():
        env = _env(ref, initial="one_compartment")
        one_shot = one_shot_commit(env)
        res = best_first_repair(env, prm)
        rows.append(
            {
                "reference": name,
                "one_shot_reward": one_shot,
                "repair_reward": res.terminal_reward,
                "repair_committed": res.committed_structure,
            }
        )
        log.info(
            "%-16s one_shot=%.3f  repair=%.3f (-> %s)",
            name,
            one_shot,
            res.terminal_reward,
            res.committed_structure,
        )

    one_shot_vals = [r["one_shot_reward"] for r in rows]
    repair_vals = [r["repair_reward"] for r in rows]
    summary = {
        "mean_one_shot": statistics.mean(one_shot_vals),
        "mean_repair": statistics.mean(repair_vals),
        "n_improved": sum(
            r["repair_reward"] > r["one_shot_reward"] + 1e-9 for r in rows
        ),
        "n_cases": len(rows),
    }
    log.info(
        "PRM-shift: mean one_shot=%.3f -> mean repair=%.3f (improved %d/%d)",
        summary["mean_one_shot"],
        summary["mean_repair"],
        summary["n_improved"],
        summary["n_cases"],
    )

    manifest = build_manifest(
        seed=0,
        config_hash=config_hash(OmegaConf.create({"experiment": "prm_shift"})),
        repo_root=_REPO,
        extra={"experiment": "prm_shift"},
    )
    out = _REPO / "outputs" / "prm_shift"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(
        json.dumps(
            {
                "manifest": json.loads(manifest.to_json()),
                "rows": rows,
                "summary": summary,
            },
            indent=2,
            default=str,
        )
    )
    log.info("wrote results -> %s", path)
    return path


if __name__ == "__main__":
    main()
