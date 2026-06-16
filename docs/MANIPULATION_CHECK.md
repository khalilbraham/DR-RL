# The manipulation check: where it reproduces, and where it cannot (yet)

The brief's manipulation check is: **remove `r_identify` → "fit-but-flat" models
return.** This note records, honestly, exactly where that reproduces in the
current system and where it does not, with the evidence.

## Reproduces rigorously at the reward level (CPU, no GPU)

With best-fit scoring (each committed structure scored at its least-squares best
fit), the reward's top choice on the ambiguous sub-saturation cases flips when
`r_identify` is removed:

| case | full argmax | no_identify argmax |
|------|-------------|--------------------|
| abstain_mm_low | one_compartment (identifiable) | michaelis_menten (flat) |
| abstain_sparse | one_compartment (identifiable) | michaelis_menten (flat) |

Test: `tests/rl/test_grpo_reward.py::test_identify_term_flips_flat_to_identifiable`.
This is the principled demonstration that `r_identify` is the term that
discourages committing a practically non-identifiable model.

## Reproduces in the trained policy as base → E-IRL (GPU)

Full-reward GRPO trains the fit-but-flat behaviour *out*: the base 3B commits the
flat Michaelis-Menten model on `abstain_mm_low`; the trained `full` policy commits
the identifiable 1-compartment model instead (flat-commit 0.50 → 0.00,
commit-accuracy 0.67 → 1.00). See `docs/PHASE5_RESULTS.md`.

## Does NOT reproduce as a trained full-vs-no_identify *separation* in the MVE

Across three GRPO runs, the trained `full` and `no_identify` policies came out
identical (both avoid the flat model). This is a **structural property of the
minimal 1c/2c/MM space**, confirmed by analysis and two parameter scans:

- **1c vs MM** (`scan over Km, dose`): a Michaelis-Menten model is *practically
  non-identifiable* only at sub-saturation, where a 1-compartment model fits it
  equally well (not unique). When the dose is high enough that MM is the *unique*
  best fit, MM is also *identifiable*. So flatness and uniqueness are mutually
  exclusive — at every (Km, dose) both `full` and `no_identify` pick MM (when
  unique+identifiable) or 1c (when sub-saturation).
- **2c** (`scan over distribution rates, sampling`): a 2-compartment model can be
  the unique best fit (1c fits clearly worse) but is then only *mildly* flat
  (identifiable fraction ≈ 0.75 at the practical 1e-2 cutoff) — too mild for the
  `full` reward's identifiability penalty to overturn its genuine fit advantage,
  so `full` and `no_identify` both (correctly) commit 2c.

The general obstruction: for **nested** PK structures, making a richer structure
practically non-identifiable (flat) means the data don't support its extra
complexity, so a simpler nested structure fits as well — which removes the fit
advantage the `no_identify` policy would need to prefer the flat model. A clean
trained-policy separation therefore requires a setting where a parameter is
**strongly non-identifiable while its structure is strictly required**.

## Update (Phase 7): the obstruction is general — confirmed for TMDD too

We added a full TMDD mechanism (`drrl.data.synth.tmdd`) and tested whether it
provides a flat-but-unique regime. It does not, for the same reason:

- **Fast binding** (large `kon`/`koff`): the individual binding rates are
  non-identifiable (only `KD = koff/kon` matters; identifiable fraction ≈ 0.50),
  **but** rapid-equilibrium TMDD reduces to ~linear/Michaelis-Menten, so a
  1-compartment or MM model fits the curve well (rmse ≈ 0.4) — *not unique*.
- **Slow binding**: TMDD is uniquely required (1c/MM fit worse), **but** then the
  binding rates shape the curve and become *identifiable* — *not flat*.

This now holds across **three** model families (1c/MM, 2c, TMDD), establishing a
general principle:

> For nested mechanistic models, a parameter being *practically non-identifiable*
> means its effect is below the data's resolution, which means a reduced model
> without it fits the data. Hence a model cannot be simultaneously **flat** and
> the **unique** best fit. `r_identify` is therefore a *tie-breaker in the
> ambiguous regime* (where a flat and an identifiable model fit equally), not a
> unique-fit discriminator.

**Consequence.** The literal trained-policy ablation ("remove `r_identify` →
fit-but-flat returns") is *structurally precluded* for nested PK structures — it
is not a matter of finding the right mechanism. The genuine validations of the
identifiability principle are therefore:

1. the **reward-level** tie-break (shown: full → identifiable, no_identify → flat
   in the ambiguous regime),
2. the **base → E-IRL** improvement (fit-but-flat trained out 0.50 → 0.00), and
3. the decisive **reward-off, real-data internalization** test (Phase 8), which
   does not depend on the ablation separating.

## (Superseded) What a trained ablation would have needed

Real mechanisms where parameter-level non-identifiability persists in a
*required* structure:

- **TMDD** (target-mediated drug disposition): the full model is required to fit
  the curve, but under common designs the QSS/MM-reducible parameters are
  non-identifiable — a textbook fit-but-flat regime.
- **PBPK** with an unmeasured tissue compartment that is needed for the
  mass balance but whose volume/partition is unidentifiable from plasma alone.
- **Effect-compartment / indirect-response** models where `ke0` or a turnover
  rate is flat under single-dose designs.

These live in the Phase-7 curriculum. Until then, the manipulation check stands
**rigorously at the reward level** and as the **base → E-IRL** improvement; the
full-vs-ablation *trained-policy* contrast is deferred to the richer task set,
and we deliberately do not manufacture it by over-tuning weights on the MVE.
