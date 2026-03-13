# Plan: Remaining Work for Hybrid `linstab2d` + Operator Architecture

Date: 2026-03-12

The operator foundation already exists under `src/linstab2d/linear_operators/`.
This file now tracks only the work that remains open.

## 1. Integrate the operator path into the current resolvent workflow

1. Use the operator-composed route alongside the current closure-based resolvent implementation.
2. Compare gains, modes, and residuals on small deterministic cases.
3. Keep the existing route as the reference path until parity and regression checks pass.
4. Only switch defaults after the operator route is validated on current cases.

### Acceptance

- Operator-composed standard resolvent matches the current implementation on controlled tests.
- Migration rule remains explicit: current route = reference, operator route = promoted only after parity passes.

## 2. Complete harmonic validation beyond unit tests

1. Keep `HarmonicLayout`, weight lifting, block assembly, BC lifting, and projected harmonic resolvent on tiny deterministic cases first.
2. Add manufactured or dense-reference checks for `harmonic_resolvent_analysis(...)` and `projected_harmonic_resolvent_analysis(...)`.
3. Connect harmonic operators to real-case inputs only after the toy/manufactured ladder is stable.
4. Do not promote projector-heavy harmonic workflows before the baseline harmonic operator is validated end-to-end.

### Acceptance

- Tiny dense-reference harmonic tests pass for gains, modes, and block structure.
- Projected harmonic route is only enabled after its neutral-direction handling is verified.

## 3. Finish production validation and parity

1. Keep Martini example validation intact while the operator layer evolves.
2. Add operator-path parity checks for Case 1 and other representative workflows before any default switch.
3. Keep Song 2023 spectrum/normalization work on an explicit validation path rather than folding it into architecture claims too early.
4. Treat harmonic validation as a separate ladder from the standard resolvent parity path.

### Acceptance

- Existing validation cases remain reproducible.
- New operator/harmonic paths have dedicated parity or manufactured-reference evidence.

## 4. Add coverage and test-run structure

1. Add explicit coverage reporting for the operator and harmonic modules.
2. Split tests into fast unit/operator tests, parity tests, and slow integration/MPI tests.
3. Add fail-under thresholds only after the test partition is stable.

### Acceptance

- Operator and harmonic helpers have explicit coverage targets and a stable test split.

## 5. Keep the publication mapping aligned with what is actually implemented

1. Limit paper claims to the operator and harmonic capabilities that are already validated.
2. Map equations and terminology directly to the shipped modules and tests.
3. Keep forward-looking harmonic extensions clearly separated from validated current functionality.
