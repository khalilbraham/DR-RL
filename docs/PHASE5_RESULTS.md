# Phase 5 (MVE GRPO) — results

Qwen2.5-3B-Instruct, 4-bit QLoRA (LoRA r32), fp16, single TITAN RTX (24 GB).
150 GRPO steps/mode, group 8, lr 5e-5, **no KL anchor** (beta 0), temperature 1.0.
Reward = verifier-grounded distinguishability-relative reward + diversity bonus.
Evaluation is greedy decoding on the 5-case task registry (3 commit, 2 abstain).
Reproduce: `uv run python -m experiments.grpo_mve --modes full fit_only`.

## Trained-policy evaluation

| mode | correct_action | commit_acc | abstain_rate (ambiguous) | flat_commit_rate | mean full-reward |
|------|---------------:|-----------:|-------------------------:|-----------------:|-----------------:|
| base (untrained) | 0.60 | 0.67 | 0.50 | 0.50 | 0.578 |
| **full (E-IRL)** | **1.00** | **1.00** | **1.00** | **0.00** | **0.746** |
| fit_only | 1.00 | 1.00 | 1.00 | 0.00 | 0.746 |

Per-case decisions (base → full):

| case | correct | base | full |
|------|---------|------|------|
| commit_1c | one_compartment | one_compartment ✓ | one_compartment ✓ |
| commit_2c | two_compartment | one_compartment ✗ | **two_compartment ✓** |
| commit_mm_high | michaelis_menten | michaelis_menten ✓ | michaelis_menten ✓ |
| abstain_mm_low | abstain | michaelis_menten ✗ (fit-but-flat) | **abstain ✓** |
| abstain_sparse | abstain | abstain ✓ | abstain ✓ |

## What this shows (and doesn't)

**Positive — the training loop and reward work.** GRPO with the verifier-grounded
distinguishability-relative reward lifts the untrained 3B from 0.60 → **1.00**
correct-action, fixing exactly the two errors the reward targets: the
biexponential under-fit (`commit_2c`) and the **fit-but-flat** Michaelis-Menten
commit on sub-saturation data (`abstain_mm_low` → abstain). Mean reward rises
0.58 → 0.75. So the end-to-end E-IRL pipeline trains correct
identifiability/abstention behaviour, and **E-IRL ≫ base**.

**Inconclusive — the full-vs-fit_only ablation did not separate at this scale.**
`fit_only` reached the same 1.0 policy. Cause: on the abstain cases the data are
sub-saturation, so a *simpler* identifiable model (1-compartment) and *abstaining*
both fit about as well as the flat Michaelis-Menten model — the pure-fit reward is
nearly *indifferent* among them, so committing a flat model is never uniquely
optimal, and the policy converges to the same (correct) answer under both rewards.
The principled manipulation signal is still visible at the **reward level**
(`tests/rl/test_grpo_reward.py::test_manipulation_check_at_reward_level`: the
abstain-vs-flat margin shrinks when `r_identify` is removed), but the
*trained-policy* reproduction needs a task where a flat (non-identifiable) model is
the **unique** best fit.

## To make the ablation bite (next)

Add abstain/identifiability cases where the flat model is the only structure that
fits (so under `fit_only`/`no_identify` the policy is pushed to commit it, while
`full` penalizes its non-identifiability and abstains). Run `full` vs
`no_identify` (isolating `r_identify`) on that harder set. Larger/parametric task
registry and a format/EOS fix (completions currently never emit EOS) would also
sharpen the signal.
