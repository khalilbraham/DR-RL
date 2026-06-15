# ADR-0001: ODE backend

- Status: Accepted (Phase 0; revisit in Phase 1 if a battery test shows unfit)
- Date: 2026-06-15

## Context

`sim/` must integrate stiff compartmental + PK/PD ODEs and provide **forward
sensitivities** (`d observed / d theta`) for the identifiability metric. We also
need a second, independent backend to cross-check correctness (invariant #1).

## Decision

Primary backend: **diffrax** (JAX) with an implicit stiff solver (Kvaerno5),
exposed behind a `Protocol` in `sim/`. Sensitivities via JAX autodiff.
Cross-check backend: **scipy** (`solve_ivp`, LSODA/Radau) with finite-difference
sensitivities, used only in tests.

## Rationale

- diffrax gives autodiff sensitivities for free, avoiding hand-coded sensitivity
  ODEs and matching the brief's "default diffrax+JAX for autodiff sensitivities".
- A scipy adapter is dependency-light and gives a genuinely independent oracle
  for the closed-form and FD-sensitivity tests.
- Backend behind a `Protocol` keeps the choice swappable (rxode2/Julia adapters
  can be added later without touching callers).

## Consequences

- JAX is an optional extra (`sim`); the pure-science schema layer stays JAX-free.
- GPU reductions are not bit-reproducible (see `ASSUMPTIONS.md`); CPU is the
  reference for tests.
