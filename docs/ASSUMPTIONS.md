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

## Phase 1 — Spec + Simulator

- **ODE RHS as sympy strings** (`ODETerm.expr`). Keeps `ModelSpec` JSON
  round-trippable and enables symbolic structural analysis (canonicalization,
  the linear transfer-function fast path) and dimensional analysis.
- **Observation `divide_by`.** The observation may divide a state by a volume
  parameter (amount -> concentration). This is what makes the closed-form
  invariant `AUC_0->inf = Dose/CL` exact for an amount compartment.
- **CanonicalForm = structural graph key.** Reparameterization-invariance is
  achieved by deriving a typed directed graph (transfers from Jacobian
  sparsity, eliminations from net flux, observation marker) and hashing its
  canonical labeling. Parameter names/values/coordinates never enter the key.
  Verified: macro<->micro, linear<->log, and renaming all hash identically;
  first-order vs Michaelis-Menten elimination and 1c vs 2c differ. `kappa(J)`
  is deliberately not used (it fails reparam-invariance, per the brief).
- **diffrax requires float64 + `ForwardMode` adjoint.** PK needs x64 (enabled on
  import). The default diffrax adjoint is reverse-mode only and cannot be
  driven by `jax.jacfwd`; forward sensitivities use `diffrax.ForwardMode`.
  Validated against scipy finite differences (rtol 1e-3).
- **Single `SaveAt(ts=...)` solve; times as JAX arrays.** Passing sample times
  as Python floats made each a compile-time constant, triggering one XLA
  compilation per time — slow, and it reliably SIGABRT-ed the XLA CPU compiler
  on this host (kernel 5.4) once compilations accumulated across the test
  session. Integrating the whole grid in one solve with array-valued times
  fixed both. Recorded as the reason behind the diffrax-backend structure.
- **Phase 1 dosing = single IV bolus at t=0.** The diffrax fast path raises
  `NotImplementedError` for t>0 boluses; the scipy reference backend handles
  timed boluses correctly. Infusion/oral/transit absorption arrive with the
  curriculum (Phase 7).
- **Solver default `Kvaerno5`** (implicit, stiff-capable) for TMDD/PBPK later;
  scipy cross-check uses `LSODA`. See ADR-0001.

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
