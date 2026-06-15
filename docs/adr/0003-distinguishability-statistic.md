# ADR-0003: Distinguishability test statistic

- Status: Proposed (decided in Phase 2)
- Date: 2026-06-15

## Context

The CORE `verifier/distinguish.py` must decide whether two models are
**observationally indistinguishable** at the operative noise level under a given
design — calibrated to a configurable nominal false-positive rate and validated
on a labeled battery of constructed pairs (invariant #3). For linear
compartmental systems there is an exact fast path: transfer-function identity.

## Decision (provisional)

- **Fast path (linear systems):** exact equality of the input-output transfer
  function (Laplace-domain), computed symbolically with `sympy`. This is the
  ground-truth oracle the general test is calibrated against.
- **General path:** a **predictive likelihood-ratio / two-sample test** between
  `Sim(M)` and `Sim(M*)` over the design's sampling grid at the operative noise
  model — statistic and threshold calibrated so the measured FPR on the pair
  battery matches `tolerances.distinguish_nominal_fpr`.

Candidate general statistics to benchmark in Phase 2: (a) noise-normalized
predictive RMSE gap with a permutation null, (b) GLRT on the residuals under the
declared error model. Selection is by measured FP/FN on the battery, recorded
here on acceptance.

## Consequences

- The fast path requires linearity detection in `spec/`/`sim/`.
- The general statistic's calibration is a *measured* artifact reported by the
  test suite, not a hardcoded number.
