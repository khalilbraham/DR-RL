# Architecture

A single Python package, layered so the **pure science never imports the
LLM/RL layer**. Dependency direction is strictly downward.

```
drrl/
  spec/          # typed mechanistic-model schema; serialization; codegen targets
  sim/           # ODE integration + forward sensitivities (backend-swappable)
  verifier/      # layered checks; each returns a structured report
    schema.py        # grammar/schema validity
    units.py         # dimensional analysis (pint)
    execution.py     # codegen + integrate; capture failures as data
    distinguish.py   # calibrated predictive distinguishability oracle (CORE)
    identify.py      # parameterization-invariant, prediction-based identifiability (CORE)
    pkpd.py          # AUC/Cmax/t1/2/CL; Emax/EC50/Hill; turnover; indirect response
    plausibility.py  # mass balance, non-negativity, missing-compartment probes
    oed.py           # experiment-design utility (EIG / D-optimal) for abstention
    report.py        # VerifierReport dataclass; aggregation contract
  reward/          # composes VerifierReport -> RewardBreakdown
  data/{synth,real}/ + splits.py
  env/             # multi-turn model-edit environment (Gym-like); partial observability
  policy/          # LLM wrapper; proposer; edit-action interface
  rl/{sft,reject,grpo,prm}.py
  bench/           # PKPD-ReasonBench: tasks, metrics, reward_validity, internalization
  utils/           # seeding, logging, config, run-manifest
```

## Layering contract

The allowed import direction (top imports from below, never the reverse):

```
bench ─┐
rl ────┼─► env ─► policy ─► reward ─► verifier ─► sim ─► spec
data ──┘                                         └────► utils (leaf)
```

- **Science layer** = `spec`, `sim`, `verifier`, `reward`. Pure and
  side-effect-free: all I/O, model calls, and randomness are injected
  explicitly. This is what makes the science testable.
- `policy`, `env`, `rl`, `bench` may import the science layer; the science
  layer must not import them. (`bench/internalization.py` and
  `data/real` carry import-firewall tests — see §6 of the brief.)
- `utils` is a leaf: it imports nothing from `drrl` except other `utils`.

## Frozen contracts

These value objects are the seams between layers. Everything downstream
consumes these and nothing else. Changing a reward weight is a **config edit**,
never a code edit. (Interfaces are being landed first; bodies follow per phase.)

- `spec/model.py`: `Compartment`, `ODETerm`, `Parameter`, `ObservationModel`,
  `Design`, `Dose`, `ModelSpec` with `canonicalize() -> CanonicalForm`
  (the reparameterization-invariant key).
- `sim/result.py`: `SimulationResult` (times, states, observed, sensitivities,
  `integrator_ok`, diagnostics).
- `verifier/report.py`: `VerifierReport` (Tier-A gates, `DistinguishReport`,
  `IdentifyReport`, `PkPdReport`, `PlausibilityReport`, `OEDReport | None`,
  actionable `feedback`).
- `reward/breakdown.py`: `RewardBreakdown` (gate, component rewards, total,
  `assert_no_nan()`).

## Determinism

A single `drrl.utils.set_seed` seeds Python/NumPy and (when present) JAX/Torch.
Every run writes a `RunManifest`. Residual nondeterminism sources (JAX/XLA,
CUDA) are catalogued in `ASSUMPTIONS.md`.

## Decisions

Real architectural decisions are recorded as ADRs in `docs/adr/`.
