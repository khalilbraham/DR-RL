# Assumptions & decisions made under ambiguity

Per the brief's Operating Rules: where the brief is silent, we decide, document
here with one sentence of rationale, and proceed. Only choices that would change
the *scientific meaning* of a result are escalated to the user.

## Phase 0 — Foundations

- **Project location.** Independent project at
  `/datadisks/datadisk3/khalil/dr-rl` (per user instruction), not a fork of the
  v2/v3 AMI line.
- **Packaging.** `uv` + `hatchling`, `src/` layout, package `drrl`. Lockfile
  (`uv.lock`) is committed for reproducible CI.
- **Heavy deps are extras.** JAX/diffrax (`sim`), torch/transformers/trl/peft
  (`rl`), and wandb (`track`) are optional extras pulled in by the phase that
  needs them, keeping the base lockfile and CI fast. Rationale: Phase 0–3 are
  pure-science/CPU and must not wait on CUDA wheels.
- **Python 3.11** as the floor (brief: ≥3.11) even though 3.13 is installed, so
  CI and contributors share the minimum supported runtime.
- **Coverage gate.** Repo-wide ≥85% (brief). `verifier/` and `reward/` will be
  raised to ≥95% as those modules land (Phases 2–3) via per-package thresholds.
- **Logging.** Stdlib `logging` with a fixed line format; experiment tracking is
  a separate thin adapter with a `NullTracker` fallback so code never
  hard-depends on a tracker.
- **Seeding.** One `set_seed` seeds Python+NumPy always, Torch+JAX when present.
  JAX's functional RNG means we only set legacy global state and validate the
  key; functional keys are derived per-call from the root seed.

## Known nondeterminism sources (cannot be fully eliminated)

- **JAX/XLA**: reduction order on GPU/TPU is not bit-reproducible; results are
  reproducible on CPU and stable to documented tolerances on GPU.
- **CUDA atomics / cuDNN**: non-associative float accumulation. `set_seed`
  requests deterministic algorithms (`torch.use_deterministic_algorithms`,
  `CUBLAS_WORKSPACE_CONFIG`) but some ops have no deterministic kernel.

## Pending decisions (to be recorded as they are made)

- Integrator tolerances per system class (Phase 1) — placeholders in
  `configs/tolerances/default.yaml`.
- Distinguishability test statistic (Phase 2) — see `docs/adr/0003-*` (draft).
- GRPO group size, LoRA rank, KL coefficient, diversity-bonus weight (Phase 5).
- How **non-identifiability is defined** operationally (prediction-affecting on
  a held-out design battery). This is scientific and is fixed in the brief
  (§verifier/identify) — recorded, not re-decided.
