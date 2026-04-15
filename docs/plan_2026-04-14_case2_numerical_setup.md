# Case 2 EVP-First Numerical-Setup Plan

Date: 2026-04-15

## Purpose

Build a publication-quality Case 2 workflow around the temporal EVP before
touching resolvent.

The core requirement is stronger than "the solver runs":

- for a fixed `stride`, the accepted EVP spectrum must be insensitive to the
  main SLEPc/PETSc parameters

Only after that is shown do we study:

- filter sensitivity
- Ducros or shock-cleanup sensitivity
- higher resolution through smaller `stride`

## Publication Standard

For each `stride`, the workflow must answer this sequence in order:

1. Can we identify the relevant spectral pocket reliably?
2. Is the accepted spectrum insensitive to the main solver knobs?
3. Once the solver is converged, how sensitive is the spectrum to filter and
   shock-capturing choices?
4. Only then: what changes when we move to a finer mesh?

This means:

- first `stride = 4`
- then `stride = 2`
- then `stride = 1`

We do not promote the resolution until the lower-resolution case is already
understood.

## Scope

Active scope now:

- temporal EVP only
- distributed PETSc operator assembly
- SLEPc EPS shift-invert eigensolves

Frozen for later:

- resolvent runs
- mixed stability+resolvent campaigns
- shock-cleanup experiments as a default workflow

## Fixed Baseline Per Stride

Until a stride is declared solver-converged, keep these choices fixed:

- Case: `results/case2/baseflow`
- `run_stability = true`
- `run_resolvent = false`
- `fd_order = 4`
- `beta = 0.0`
- `alpha_filter = "none"` for EVP-only studies
- shock cleanup disabled
- one physical socket, 10 MPI ranks
- `petsc_matrix_mode = distributed`
- direct shift-invert with MUMPS
- current BC recipe from `results/case2/parity_config.jsonc`

Important note:

- historical configs often store `alpha_filter = 0.0`
- the runtime now overrides that to `"none"` when `run_resolvent = false`
- for new EVP-only studies, write `"none"` explicitly to avoid ambiguity

## Open Interpretation Risk

`beta` is not yet fully documented in the Python-facing Case 2 workflow.

Current code path:

- Case 2 configs expose `numerics.beta`
- the runtime passes that value into `build_lnse_system(..., kx=beta, ...)`
- the continuous equations and literature notation suggest `beta` is the
  spanwise wavenumber

So the action item is:

- verify the exact meaning of `beta` in the current implementation before any
  publication-facing `beta` sweep

Until then:

- keep `beta = 0.0` in the baseline

## Acceptance Metrics

Every EVP run must record:

- wall time
- peak RSS
- stage timings
- `stride`
- `fd_order`
- `beta`
- requested `nev`
- used `ncv`
- `eps_tol`
- `conv_test`
- `true_residual`
- actual `nconv`
- explicit postcheck residuals
- target-distance diagnostics
- accepted-mode counts
- saved eigenvalue spectrum artifact

Do not use `nconv` as the acceptance criterion by itself.

## What "Solver-Converged" Means

For one fixed `stride`, we declare the EVP setup solver-converged only when:

- the same spectral pocket is found across the useful solver settings
- the leading accepted eigenvalues stop moving in a material way when `ncv`
  and `eps_tol` are tightened
- accepted-mode counts stop changing materially across the useful solver range
- explicit postcheck residuals are below the agreed quality gate
- runtime may still change, but the accepted spectrum should not

The exact numeric thresholds should be recorded in the campaign summary for
each stride, but the qualitative rule is:

- solver knobs may change cost
- they must not change the published physics conclusion

## Phase Order

### Phase 0: Inventory and freeze

Goal:

- simplify the campaign before launching more jobs

Actions:

- mark EVP as the only active path
- freeze resolvent and shock-cleanup studies
- keep legacy files for evidence, but do not schedule new runs from them
- document the active path in `results/case2/README.md`

Exit condition:

- everyone can identify which files are active, frozen, or legacy

### Phase 1: Stride-4 branch localization

Goal:

- find a trustworthy spectral pocket cheaply before large sweeps

Rules:

- keep `fd_order = 4`, `beta = 0.0`, filter off, shock cleanup off
- use small `nev`
- use a broad but explicit acceptance window

Questions answered here:

- which target pocket should the solver chase at stride 4
- are there multiple candidate pockets worth following

Exit condition:

- one target region is selected for the stride-4 convergence study

### Phase 2: Stride-4 PETSc/SLEPc insensitivity study

Goal:

- show that the stride-4 EVP is not sensitive to the main solver knobs

Subphase 2A: `ncv` sweep

- fix target pocket
- fix small `nev`
- vary `ncv`

Decision:

- identify the smallest `ncv` range that stabilizes the accepted spectrum

Subphase 2B: `eps_tol` sweep

- hold target and `nev` fixed
- sweep `eps_tol` over a practical range

Decision:

- identify the cheapest tolerance that does not move the accepted spectrum

Subphase 2C: `nev` growth

- increase `nev` only after `ncv` and `eps_tol` are stable

Decision:

- determine how far the eigensolve can grow before accepted quality degrades or
  cost becomes unreasonable

Subphase 2D: ambiguity breakers

Only if the spectrum still changes materially:

- compare `conv_test = rel|abs|norm`
- compare `true_residual = true|false`
- check `eps_max_it` only if there is evidence of iteration truncation

Exit condition:

- stride 4 has a solver setup that is publication-safe

### Phase 3: Stride-4 physical and numerical sensitivity

Goal:

- once solver uncertainty is removed, test the non-solver knobs

Order:

1. filter sensitivity
2. shock-cleanup or Ducros sensitivity
3. `beta` sensitivity

Rules:

- vary one family at a time
- keep the solver setup fixed from Phase 2
- compare spectra against the stride-4 converged baseline

Questions answered here:

- does the filter materially change the accepted spectrum
- does Ducros or shock cleanup materially change the accepted spectrum
- how sensitive is the spectrum to `beta`

Exit condition:

- stride 4 has both a converged solver setup and a documented sensitivity map

### Phase 4: Promote to stride 2

Goal:

- repeat the same logic on a finer grid

Rules:

- start from the stride-4 winner, not from scratch
- keep filter off and shock cleanup off initially
- rerun branch localization only if the target pocket moved

Required order:

1. target confirmation
2. solver-insensitivity check
3. filter sensitivity
4. Ducros sensitivity

Exit condition:

- stride 2 reaches the same level of confidence already achieved at stride 4

### Phase 5: Promote to stride 1

Goal:

- confirm the final high-resolution spectrum

Rules:

- only carry the narrowed settings from stride 2
- do not reopen broad solver sweeps unless the spectrum changes materially

Required order:

1. target confirmation
2. solver-insensitivity check
3. filter sensitivity
4. Ducros sensitivity

Exit condition:

- stride 1 is good enough to support publication claims

### Phase 6: Resolvent

Goal:

- return to resolvent only after the EVP path is understood

At that point:

- revisit mesh filtering in resolvent mode
- revisit shock cleanup if it still matters
- compare resolvent conclusions against the already-converged EVP baseline

## Recommended Campaign Matrix

For each stride:

1. localization run
2. `ncv` sweep at fixed low `nev`
3. `eps_tol` sweep at the useful `ncv`
4. `nev` growth at the chosen solver point
5. filter sensitivity
6. Ducros sensitivity
7. `beta` sensitivity

That matrix is the minimum needed to defend non-sensitivity claims.

## Delete Policy

Do not delete the legacy files yet.

First:

- complete the EVP-first reset
- identify the active config set
- get one stride-4 converged story

After that:

- archive or delete the legacy mixed-study files with much less risk

## Deliverables

The Case 2 campaign is only complete when we have:

- one clear README for the case
- one documented active EVP path
- one converged stride-4 story
- one converged stride-2 story
- one converged stride-1 story
- a justified statement about filter sensitivity
- a justified statement about Ducros or shock-cleanup sensitivity
- a justified statement about solver non-sensitivity

Only then should the results be treated as publication-ready.
