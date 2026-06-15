# ADR-0002: RL library for GRPO

- Status: Proposed (decided in Phase 5)
- Date: 2026-06-15

## Context

The brief mandates GRPO over multi-turn model-edit trajectories with a
group-relative advantage, a KL anchor to the SFT/RS reference, and a diversity
bonus over canonical forms. It also says: prefer a maintained library (TRL's
GRPO trainer or verl) behind a thin interface; do not hand-roll the PPO core
unless a test shows the library is unsuitable.

## Decision (provisional)

Build on **TRL's `GRPOTrainer`** behind a thin `drrl.rl` interface, with LoRA via
**peft** for the MVE. Keep group sampling, the diversity bonus (entropy over
canonical forms), and the verifier-grounded reward in first-party code so the
science is ours; delegate the PPO/GRPO optimizer core to TRL.

## Status note

Final acceptance is deferred to Phase 5, where we will verify TRL's installed
API supports custom group rewards and a KL reference, and record a fallback to
**verl** if a test shows TRL unsuitable for multi-turn edit trajectories.

## Consequences

- torch/transformers/trl/peft are an optional extra (`rl`).
- The env/reward/GRPO loop is unit-testable on CPU with a stubbed policy.
