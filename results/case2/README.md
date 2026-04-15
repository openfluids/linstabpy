# Case 2 EVP-First Campaign

## Objective

Reset `results/case2` around one clear goal:

- obtain a stable and explainable temporal EVP spectrum for Case 2
- understand how that spectrum changes with `stride`
- understand how the EVP changes with SLEPc/PETSc solver settings
- postpone resolvent until the EVP path is under control

The final target is still to support both EVP and resolvent. For now, the
active campaign is EVP only.

## Working Decision

From now on, new Case 2 runs should follow these rules:

- `run_stability = true`
- `run_resolvent = false`
- start with `stride = 4`
- then promote to `stride = 2`
- then promote to `stride = 1`
- keep shock cleanup disabled until the baseline EVP is trusted
- keep resolvent studies frozen until the EVP setup is defensible

Do not schedule new mixed stability+resolvent runs while the EVP baseline is
still unclear.

## Active Questions

We do not yet have firm answers for:

- which spectral pocket should be targeted first at each `stride`
- how large `ncv` must be for a stable accepted spectrum
- how much tighter `eps_tol` actually improves explicit postchecked residuals
- how `nev` should grow once the solver setup is stable
- whether `conv_test` and `true_residual` materially change the accepted modes
- what `beta` means operationally in the current Python interface
- whether EVP should ever use the mesh filter, or always use `"none"`
- whether shock cleanup changes physics or only repairs numerical pathologies

These are study questions, not fixed truths.

## Baseline

The baseline to hold fixed while calibrating the EVP is:

- Case 2 baseflow from `results/case2/baseflow`
- `fd_order = 4`
- `beta = 0.0`
- `run_stability = true`
- `run_resolvent = false`
- 10 MPI ranks pinned to one physical socket
- `petsc_matrix_mode = distributed`
- shift-invert EVP with direct LU through MUMPS
- no shock cleanup
- explicit stage heartbeat enabled for long runs

For future EVP-only configs, prefer setting:

- `numerics.alpha_filter = "none"`

instead of relying on the runtime override from `0.0 -> "none"`.

## What Is Active Now

These files are part of the active EVP path:

- `results/case2/parity_config.jsonc`
  - canonical full-resolution Case 2 config
- `results/case2/case2.py`
  - direct Case 2 runtime entrypoint
- `results/case2/perf_memory.py`
  - Case 2 timing and memory campaign harness
- `results/case2/run_perf_queue_once.sh`
  - queue wrapper for Case 2 campaign matrices
- `results/case2/evp_stride4_localize.jsonc`
  - active stride-4 branch-localization matrix

Historical configs, scripts, and plots live under:

- `results/case2/legacy/`

They are kept for evidence, but they are intentionally out of the active path.

## First Execution Order

This order is strict. Do not advance to the next block before the current one
is defensible.

### Phase 1: Stride 4 branch localization

Goal:

- get one trustworthy spectral pocket on the cheap mesh first

Use:

- `evp_stride4_localize.jsonc`

Expected output:

- at least one accepted mode with explicit postcheck and target distance that
  make physical sense

### Phase 2: Stride 4 PETSc/SLEPc convergence

Goal:

- hold physics fixed and show that the accepted EVP spectrum is not sensitive
  to the main solver knobs

Study order:

1. `ncv` sweep at fixed `nev`
2. `eps_tol` sweep at the useful `ncv` values
3. `nev` growth after the low-`nev` setup is stable
4. optional `conv_test` and `true_residual` checks only if residual quality is
   still ambiguous

Publication rule for this phase:

- stride 4 is only considered converged when the leading accepted spectrum is
  stable under these solver changes

### Phase 3: Stride 4 physical and numerical sensitivity

Only after the stride-4 EVP is solver-insensitive:

- interpret and sweep `beta`
- compare `alpha_filter = "none"` against filtered alternatives
- test shock cleanup or Ducros-style cleanup as a controlled perturbation, not
  as a default

### Phase 4: Promote to stride 2

Repeat the stride-4 workflow with the smallest matrix that answers:

- does the spectral pocket stay in the same place
- do the accepted modes stay consistent
- what is the runtime and memory increase
- is the stride-2 spectrum again insensitive to the solver knobs

### Phase 5: Promote to stride 1

Repeat only the calibrated subset:

- chosen target pocket
- chosen `ncv`
- chosen `eps_tol`
- chosen `nev`

Stride 1 should be a confirmation run, not the place where solver choices are
still drifting.

### Phase 6: Resolvent

Only after the EVP path is stable:

- restore `run_resolvent = true`
- revisit filter choices
- revisit shock cleanup if it still seems useful

## Metrics That Matter

Every EVP run should be judged with:

- wall time
- peak RSS
- stage timings
- requested `nev`
- actual `nconv`
- explicit postcheck residuals
- target-distance diagnostics
- accepted-mode counts
- a saved spectrum artifact

Do not trust `nconv` alone.

## Commands

Direct single-case run:

```bash
LINSTAB2D_CASE2_CONFIG=results/case2/parity_config.jsonc \
LINSTAB2D_CASE2_OUTPUT_NPZ=results/case2/output/runs/case2_manual.npz \
LINSTAB2D_STAGE_HEARTBEAT_SECONDS=30 \
./.venv/bin/python results/case2/case2.py
```

Queued matrix run through the shared campaign harness:

```bash
LINSTAB2D_CASE2_PERF_CONFIG=results/case2/evp_stride4_localize.jsonc \
./.venv/bin/python results/case2/perf_memory.py
```

## Delete Policy

The top level of `results/case2` should stay small.

If a file is not part of the current EVP-first campaign, move it to
`results/case2/legacy/` instead of leaving it in the active path.
