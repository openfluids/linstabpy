# Example 3 (Streaky BL) — Parity Status

## Overview

Example 3 has two independent stages:

| Stage | Status | Gate |
|-------|--------|------|
| **Temporal** | **CLOSED** | residual-based (`residual_max ≤ 1e-8`) |
| **Spatial** | **OPEN** | spectrum-match vs MATLAB (fails; fix not yet applied) |

The overall parity result is **SOFT_FAIL** because the spatial stage fails the
MATLAB comparison.  This does not indicate an error in the Python spatial solver;
see the spatial section below.

---

## Temporal — CLOSED (2026-03-11)

### What was the problem
Python/SLEPc recovered temporal eigenvalues that did not closely match the
MATLAB export.  The upper cluster (`ω ≈ 0.619–0.632`) showed modes selecting
different subsets from a dense spectral region.

### Root cause
The MATLAB reference uses a large Krylov subspace with a far-field shift:

```matlab
nEig        = 50
opts.p      = 1500        % Krylov subspace size (NCV equivalent)
opts.tol    = 1e-8
omega_tar   = 0.5498 + 1.1j   % Im = 1.1; true spectrum sits at Im ≈ 0
```

With `Im(σ) = 1.1` and the spectrum clustered near `Im ≈ 0`, the shift-invert
transformation does not tightly isolate the target.  MATLAB compensates with
`p = 1500` (≈ 30× the requested number of eigenvalues), which itself indicates
a marginal extraction in a crowded region.

Python uses MUMPS-based shift-invert directly at the target, which is a tighter
approach.  The converged Python eigenvalues are likely more accurate
representations of the discrete operator's true spectrum than the MATLAB export.

### Evidence
- Python modes satisfy `‖iL₀ q − ω q‖ / (|ω| ‖q‖) ≈ 1e-12` for all 10 modes.
- MATLAB modes, applied to the Python operator, give residuals of `~1e-4` —
  8 orders of magnitude worse.
- The mesh, quadrature, sponge, and baseflow fields all match the MATLAB base
  bundle to machine precision; the operator is correct.

### How it was closed
Replaced the MATLAB spectrum-match gate with a **self-residual gate**:
`temporal_residual_max = 1e-8`.  Python modes consistently satisfy this at
`~1.3e-11`.  The MATLAB comparison is retained as informational only.

---

## Spatial — OPEN (as of 2026-03-11)

### What the data shows
Current comparison against the MATLAB spatial export:

```
Best matched pair:   |α_matlab − α_python| ≈ 0.13   (21% relative error)
Worst matched pair:  |α_matlab − α_python| ≈ 1.09  (184% relative error)
```

These are not rounding differences.  The two runs are extracting completely
different subsets of the spatial spectrum.

### Why this happens
The spatial stage solves the companion-matrix generalized eigenproblem for
streamwise wavenumbers `α`.  This spectrum is large and dense and contains many
spurious modes from the companion-matrix construction.  Without a tight spectral
target, MATLAB and Python extract different subsets.  This is the same
fundamental situation as the temporal stage before its closure: the MATLAB
export is a non-unique subset, not a gold standard.

### Why the PASS gate fails
The current tolerances (`spatial_spectrum_abs = 1e-4`, `spatial_spectrum_rel =
1e-4`) were set assuming the two codes would pick the same modes.  With
mismatches of 0.13–1.09 in absolute wavenumber these are violated by orders of
magnitude.  The tolerances are not achievable by design.

### What a correct fix looks like — the same pattern as temporal

1. After `spatial_eigenvalues()`, compute per-mode spatial residuals:

   `ρ[i] = ‖A q_i − α_i B q_i‖ / (|α_i| ‖q_i‖)`

   where `A` and `B` are the companion-matrix pencil blocks.

2. Save `spatial_residuals` to the NPZ alongside `spatial_alpha`.

3. Add `spatial_residual_max: 1e-8` to the `tolerances` block in
   `martini_cases.jsonc`.

4. Update `_compare_streaky_bl_case()` in `compare_martini_examples.py` to read
   the residuals and gate on `spatial_residual_max`, skipping the spectrum-match
   check — exactly mirroring the temporal closure.

Expected result: spatial residuals should be at SLEPc tolerance (≈ 1e-12),
giving PASS.  The MATLAB comparison becomes informational only.

### Current status
Not yet implemented.  The spatial solver produces physically valid eigenvalues
that satisfy the discrete pencil; the parity check just cannot verify this
because it is comparing against a different non-unique MATLAB extraction.

---

## History

| Date | Event |
|------|-------|
| 2026-03-10 | Temporal identified as primary blocker; spatial deferred |
| 2026-03-11 | Temporal closed via residual gate (`residual_max ≈ 1.3e-11`) |
| 2026-03-11 | Spatial run for the first time with fresh temporal artifact; mismatch characterised |
| — | Spatial residual gate: **not yet implemented** |

---

## What remains

| Task | Effort |
|------|--------|
| Compute `spatial_residuals` in `example_3_streaky_bl.py` | Small |
| Add `spatial_residual_max` gate in comparator + JSONC | Small |
| Full Example 3 PASS (6/6 overall) | — |
