# Hamada Case: Why Python/PETSc Is Not Yet Matching MATLAB

## Objective
Use one small Hamada reference case to prove solver equivalence between:
- MATLAB reference workflow (`example_6_hamada/workflow/main.m`)
- Python + PETSc/SLEPc/MUMPS workflow (`example_6_hamada.py`, `example_6_hamada.sh`)

This case is the calibration gate before moving to realistic large cases that do not scale in MATLAB.

## Current Status
Pipeline is working end-to-end (solve + parity + figures), but numerical agreement is not yet achieved.

Latest spectrum parity (`artifacts/hamada_spectrum_parity.json`):
- `n_compare = 100`
- `max_abs_error = 9.077525914618567`
- `rms_abs_error = 5.815515993020508`
- `max_rel_error = 3.853871515724929`
- `pass = false`

## What This Means
The main issue is not execution or tooling. The issue is mathematical equivalence of the assembled linearized operator.

In practice:
- BC application path is mostly aligned, but still must be verified at row/block level against MATLAB matrices.
- The dominant mismatch appears in operator formation (linearized coefficients and block assembly), not in plotting or post-processing.

## Root Cause Focus (What We Must Verify)
1. **Operator matrix equivalence (`L0`)**
- Compare Python vs MATLAB matrix blocks for the linearized compressible system.
- Check variable ordering and block placement consistency (`rho, u, v, w, T`).

2. **Linearized coefficient equivalence**
- Validate coefficient terms used in assembly (`A0`, `Ax`, `Ay`, `Axx`, `Ayy`, `Axy`) against MATLAB at sampled points.
- Confirm metric terms and deformation/Jacobian usage are equivalent.

3. **Boundary condition row equivalence**
- Verify each BC group row-by-row:
- Dirichlet: `lt/ruvwT`, `b/ruvw`
- Neumann: `b/T`, `r/ruvwT`
- Confirm penalty scaling and target DOF sets match MATLAB exactly.

4. **Eigenvalue convention equivalence**
- Ensure both pipelines compare the same eigenproblem convention around the same target.
- Keep selection/sorting logic identical for the parity comparison.

## Required Evidence Before Declaring Agreement
1. Matrix-level checks
- `nnz` and sparsity pattern alignment
- Random-action residual small
- Blockwise residuals (interior + BC rows) small

2. Spectrum-level checks
- Hungarian-matched errors within agreed tolerances

3. Mode-level checks
- Phase-aligned field comparison for selected reference modes (for example mode 7)

## Practical Conclusion
No agreement yet because operator formation equivalence is not yet proven.
The next work must stay focused on matrix/BC/linearization parity, not on additional workflow features.

## Investigation Plan (Matrix-First)
1. Freeze one reference configuration.
- Use only one fixed case and one fixed set of runtime parameters.
- Do not change solver options while diagnosing equivalence.

2. Split parity into `pre-BC` and `post-BC`.
- Export/compare:
- `L0_preBC`: immediately after operator assembly.
- `L0_postBC`: after all BCs are applied.
- This isolates whether mismatch is from linearization/assembly or from BC enforcement.

3. Add operator parity diagnostics (not only spectrum parity).
- Create operator reports with:
- global relative matrix error,
- interior-only error,
- BC-row-only error,
- per-variable-block (`rho,u,v,w,T`) errors,
- worst offending rows/blocks.

4. Branch root-cause work based on where mismatch is largest.
- If `pre-BC` mismatch is large:
- investigate linearized coefficient formation and mesh metric mapping.
- If `pre-BC` is close but `post-BC` mismatch is large:
- investigate Dirichlet/Neumann row construction and BC index sets.

5. Check eigenvalue convention only after operator agreement.
- Spectrum/mode mismatch is expected when operator mismatch is still large.
- First make `L0_postBC` equivalent; then evaluate eig parity.

6. Acceptance gate for “equivalent”.
- Operator parity must be close (global + interior + BC rows).
- Then spectrum parity must pass.
- Then mode-shape parity (phase-aligned field comparisons) must pass.
