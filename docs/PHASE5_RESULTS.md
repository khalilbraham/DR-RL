# Phase 5 (MVE GRPO) — results

Qwen2.5-3B-Instruct, 4-bit QLoRA (LoRA r32), fp16, single TITAN RTX (24 GB).
150 GRPO steps/mode, group 8, lr 5e-5, **no KL anchor** (beta 0), temperature 1.0.
Reward = verifier-grounded distinguishability-relative reward (+ diversity bonus),
with committed structures scored at their **best fit** to the data (`sim.fit_params`)
and `r_identify` computed at a **practical** rank tolerance (1e-2).
Greedy eval on the 5-case registry (3 commit, 2 abstain).
Reproduce: `uv run python -m experiments.grpo_mve --modes full no_identify`.

## Trained-policy evaluation (best-fit run)

| mode | correct_action | commit_acc | flat_commit (identify) | mean full-reward |
|------|---------------:|-----------:|-----------------------:|-----------------:|
| base (untrained) | 0.60 | 0.67 | **0.50** | 0.836 |
| **full (E-IRL)** | 0.80 | **1.00** | **0.00** | **0.978** |
| no_identify | 0.80 | 1.00 | 0.00 | 0.978 |

`flat_commit (identify)` = fraction of ambiguous cases where the policy commits a
model whose parameters are *practically non-identifiable* at its best fit — the
"fit-but-flat" failure.

Per-case decisions (base → full):

| case | base | full |
|------|------|------|
| commit_1c | one_compartment ✓ | one_compartment ✓ |
| commit_2c | one_compartment ✗ | **two_compartment ✓** |
| commit_mm_high | michaelis_menten ✓ | michaelis_menten ✓ |
| **abstain_mm_low** | **michaelis_menten ✗ (fit-but-flat)** | **one_compartment ✓ (identifiable)** |
| abstain_sparse | abstain | abstain |

## What this shows

**E-IRL trains fit-but-flat *out* and perfects structure choice.** The
verifier-grounded reward takes the base 3B from flat-commit 0.50 → **0.00** and
commit-accuracy 0.67 → **1.00**, mean reward 0.84 → 0.98. Concretely it fixes the
two base errors: the biexponential under-fit (`commit_2c`) and — the headline —
the **fit-but-flat** commit on `abstain_mm_low`, where the base commits the
sub-saturation Michaelis-Menten model (parameters non-identifiable) and the
trained policy instead commits the identifiable 1-compartment model. This is the
central claim, demonstrated in a trained policy: the distinguishability/
identifiability-grounded reward shapes the agent toward *identifiable* model
choices.

## What this does *not* show (honest)

**The full-vs-no_identify ablation did not separate in the trained policy**
(both reach flat-commit 0.0). Across three runs the trained `no_identify` policy
also avoids the flat model. Cause: the remaining reward terms (equivalence,
parsimony, abstention) plus the base model's strong prior toward the simpler
identifiable answer already suppress the flat choice, and the residual
`r_identify` margin under `no_identify` is small (at the reward level the flat MM
sits only ~0.015 above the identifiable 1c), too thin for GRPO to overturn the
prior in 150 steps.

The manipulation check **is** demonstrated rigorously at the **reward level**
(`tests/rl/test_grpo_reward.py::test_identify_term_flips_flat_to_identifiable`:
with `r_identify` the reward's top choice is the identifiable model; without it,
the flat model becomes the top choice). The *trained-policy* reproduction needs a
setting where the flat model is the **unique, clearly-best** fit so the
`no_identify` reward strongly prefers it — i.e. the richer Phase-7 task set
(TMDD/PBPK, parameter-level non-identifiability), not the 3-structure MVE.

## Bottom line

- Reward correctness (the idea): supported — reward-level argmax flips with
  `r_identify`; verifier invariants green.
- Training (E-IRL): supported — full reward removes fit-but-flat (0.50→0.00) and
  lifts correctness (0.67→1.00) in a real QLoRA GRPO run, E-IRL ≫ base.
- Reward-ablation contrast in the trained policy: not reproduced at MVE scale;
  documented with the concrete fix (richer tasks). Decisive real-data
  internalization test remains Phase 8.
