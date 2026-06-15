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

## Phase 2 — Verifier core

- **Tier-A checks are total** (never raise): schema/execution/units/plausibility
  return structured reports so a bad proposal is a learning signal, not a crash.
- **Mass balance is structural** (sign analysis of the symbolic net flux), which
  is reparameterization-invariant and can fail (detects spurious mass creation),
  unlike a numeric-accumulator check that conserves by construction.
- **Identifiability metric = rank(S)/n_params** of the noise-normalized
  prediction-sensitivity matrix over a held-out battery. Rank is invariant under
  reparameterization; `kappa(S)` is not used (fails invariance, per the brief).
  Verdicts are grounded by `prediction_change_along` (move along a direction,
  measure RMS standardized prediction change).
- **Identifiability tests use the scipy (FD) backend**: clean rank separation,
  ~400x faster than diffrax autodiff (validated separately), and avoids
  XLA-compiler pressure. Identifiability is backend-agnostic.
- **Linear distinguishability fast path = Markov-parameter equality** `{c A^k b}`
  (realization theory), equivalent to transfer-function identity but pure NumPy
  and numerically robust. It is the ground truth the predictive oracle is
  calibrated against; FP/FN measured on a constructed labeled pair battery.
- **Distinguishability/identifiability need a meaningful noise model.** With a
  degenerate (zero) error model the `sigma` floor (1e-9) amplifies numerical
  residuals; analyses assume a realistic error model (e.g. proportional 0.1).
- **Competitor set is open**: assembled at runtime from rollouts + a programmatic
  enumeration prior + a replay buffer, de-duplicated by canonical key. Never a
  hardcoded list (invariant #6).
- **OED**: D-optimality (`log det` FIM) for parameter precision; a model-
  discrimination utility (worst-case predictive noncentrality over competitors)
  for abstention designs, with `resolves_ambiguity` gated by the chi-square
  quantile.

## Phase 3 — Reward + synthetic data

- **Reward total = `gate * sum_i w_i r_i`** with weights summing to 1 and every
  component squashed to `[0,1]` (`safe01` maps NaN/inf -> 0). `assert_no_nan` is
  called before returning, so no NaN reaches the optimizer.
- **Parsimony is gated by fit adequacy** (`r_fit >= tau`): an under-fitting model
  scores 0 parsimony and so can never win on simplicity (test:
  `test_parsimony_cannot_collapse_model`). Among fit-adequate models it tie-breaks
  toward fewer parameters.
- **`r_fit` is scored only on the hidden battery** the agent never observed
  (test records every simulated grid and asserts the observed grid is never
  simulated, while hidden grids are).
- **Abstention asymmetry**: wrong confident commit -> `1 - wrong_commit_penalty`
  (0.0); over-abstaining -> `1 - over_abstain_penalty` (0.7); correct abstention
  when ambiguous + a proposed design -> 1.0.
- **`r_pd` reuses `r_fit`** in the MVE library (no separate PD observable yet);
  it gains a real PD observable with effect/indirect-response models in Phase 7.
- **Synthetic labels are cross-validated against the verifier**: identifiability
  labels match `identifiability()` fractions and (in)distinguishability labels
  match the oracle — the generator's ground truth is not taken on faith.
- **`one_compartment` and `confounded` share a canonical key**: same *structure*,
  different identifiability — a concrete instance of the equivalence-class idea.
- **TMDD/PBPK/indirect-response deferred to Phase 7** (build order: not before
  the MVE loop is green).

## Phase 4 — Env + warm-start

- **Edit environment operates over a structure space** (one_compartment /
  two_compartment / michaelis_menten) with typed transitions — `AddCompartment`
  / `RemoveCompartment` (+/- compartment), `SwapKinetics` (first-order <-> MM),
  `TuneParam`, `Commit`, `Abstain(design)`. Free-form ODE-string surgery is
  deferred; this keeps Phase-4 tractable and fully tested while preserving the
  multi-turn edit + commit/abstain dynamics.
- **Partial observability**: the `Observation` exposes only the candidate's own
  Tier-A gates, feedback, identifiability, and fit on the *shown* design — never
  the reference, the hidden battery, or admissible labels (asserted by test).
- **Determinism**: the environment is deterministic given the action sequence
  (observed data are fixed reference predictions); episodes are replayable.
  Stochastic stub policies are seeded.
- **Layering note**: `policy` depends on `env` (it targets the env's
  action/observation space); `env` does not import `policy` (no cycle). This
  refines the sketch in the brief (which listed `env -> policy`).
- **Reward computed only at termination** via the distinguishability-relative
  reward, using the env-internal reference + hidden battery. Running out of turns
  forces a commit-scored terminal.
- **Warm-start without an LLM**: rejection sampling rolls out a `Policy` stub and
  keeps trajectories above a reward threshold; `fit_sft` returns an
  `ImitationPolicy` (structure->action table) on canonicalized targets. DoD test
  confirms SFT+RS yields a policy that commits Tier-A-valid specs.

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
