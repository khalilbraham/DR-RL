# DR-RL — Distinguishability-Relative Reinforcement Learning

Post-train an LLM to generate pharmacokinetic/pharmacodynamic (PK/PD) and
compartmental mechanistic models, graded not against a single reference
structure but against **what the data can determine** under a given dosing
design and noise model.

A single calibrated **distinguishability oracle** yields three rewards as
special cases:

1. **Equivalence-class reward** — credit for any structure observationally
   indistinguishable from the reference.
2. **Parameterization-invariant, prediction-based identifiability reward.**
3. **Calibrated abstention reward** — when the admissible structure set is
   large, the agent is rewarded for declining to commit and proposing an
   experiment (optimal experimental design) that would resolve the ambiguity.

Training is GRPO over multi-turn model-edit trajectories with verifier-grounded
rewards. The headline test runs the trained policy **with the verifier and
reward switched off**, on **real data labeled independently by
profile-likelihood analysis**, to test whether identifiability/abstention
reasoning was *internalized* rather than merely satisfied.

## Status

Phase 0 (foundations) complete. See `docs/ARCHITECTURE.md` for the layering and
the frozen contracts, and the Build Order below for what comes next.

## Quickstart

```bash
# 1. Install uv (https://docs.astral.sh/uv/) if needed:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync the locked dev environment:
make install            # == uv sync --group dev

# 3. The Phase-0 gate (lint -> type -> test + coverage):
make check

# 4. Sanity entrypoint (compose config, seed, write a run manifest):
make smoke
```

Optional heavy stacks are pulled in per phase via extras:

```bash
uv sync --extra sim      # Phase 1: JAX + diffrax (ODE + sensitivities)
uv sync --extra rl       # Phase 5: torch + transformers + trl + peft
```

## Reproducibility

Every run writes a manifest (git SHA, config hash, seed, library versions,
platform) via `drrl.utils.manifest`. All tolerances, weights, thresholds, and
seeds live in `configs/` — there are no magic numbers in code.

## Build order (phased; each phase ends green or we stop and report)

| Phase | Scope | DoD |
|------:|-------|-----|
| 0 | Foundations: uv, ruff, mypy(strict), pytest+cov, CI, utils, Hydra | `make check` green in CI |
| 1 | `spec/` + `sim/` | closed-form sim tests; reparam canonicalization; sensitivities vs FD |
| 2 | Verifier core (`units`, `execution`, `pkpd`, `plausibility`, **`distinguish`**, **`identify`**) | FP/FN on pair battery; identifiability invariant green |
| 3 | `reward/` + `data/synth/` | NaN-safe gated reward; reward-hacking adversarial tests fail to hack |
| 4 | `env/` + `rl/sft.py` + `rl/reject.py` | SFT+RS produces Tier-A-valid specs; deterministic episodes |
| 5 | `rl/grpo.py` (**MVE**) | manipulation-check ablation reproduces; E-IRL beats fit-only GRPO |
| 6 | `rl/prm.py` + edit search | PRM-shift experiment; multi-turn repair beats one-shot |
| 7 | Scale (TMDD/PBPK) + `data/real/` + full `bench/` | curriculum + profile-likelihood labels + 14 tasks |
| 8 | Validity gate + headline | reward-validity PASS/FAIL; reward-off internalization eval |

## Repository layout

See `docs/ARCHITECTURE.md`. The dependency direction is strictly downward:
the pure-science layer (`spec`, `sim`, `verifier`, `reward`) never imports the
LLM/RL layer.

## License

MIT.
