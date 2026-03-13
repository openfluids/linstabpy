# Plan 2026-03-13: Dual SciPy/PETSc Operator Comparison Harness

## Goal

Keep the current SciPy assembly path as the reference implementation while adding a PETSc-native assembly path that can be validated safely by building both matrices and comparing them numerically and structurally.

This plan is based on the current operator assembly and parity tooling in:
- `src/linstab2d/operators/assembly.py:22-70`
- `src/linstab2d/operators/assembly.py:191-286`
- `src/linstab2d/operators/assembly.py:357-432`
- `src/linstab2d/utils/petsc_core.py:59-77`
- `tools/parity/compare_operator_construction.py:69-136`
- `tools/parity/compare_operator_construction.py:216-223`
- `tests/test_case1_parametric_parity.py:131-140`

## Current state

1. The canonical LNSE and spatial operators are built in SciPy sparse format, not directly in PETSc.
2. PETSc currently enters through conversion utilities that transform SciPy matrices into PETSc AIJ matrices for solver use.
3. The repository already contains parity-style diagnostics for sparse deltas and PETSc-vs-SciPy matvec agreement.

## Design principles

1. Keep the existing SciPy builder unchanged as the reference path.
2. Add PETSc-native assembly incrementally rather than replacing the current path immediately.
3. Compare both paths before any solver default changes.
4. Return structured mismatch reports rather than only pass/fail booleans.
5. Start with sequential/local PETSc validation before handling distributed comparison.

## Proposed interfaces

### 1. Keep separate builders

Retain:
- `build_lnse_operator(...) -> sp.csc_matrix, idx`

Add:
- `build_lnse_operator_petsc(...) -> PETSc.Mat, idx`

Rationale:
- separate functions make parity work explicit and easier to debug;
- the SciPy path remains the trusted reference while the PETSc path matures.

### 2. Add PETSc -> SciPy extraction utility

Add a reverse conversion helper in the PETSc utility layer:
- `petsc_to_scipy(mat: PETSc.Mat, *, format: str = "csr") -> sp.spmatrix`

Suggested first version:
1. Support sequential/local AIJ matrices only.
2. Normalize the extracted matrix to CSR for comparison.
3. Add distributed support later, either by gathering to one rank or by comparing ownership blocks.

Natural home:
- `src/linstab2d/utils/petsc_core.py`

This complements the existing forward conversion utilities in:
- `src/linstab2d/utils/petsc_core.py:59-77`
- `src/linstab2d/utils/petsc_core.py:170-269`

### 3. Add sparse comparison helper

Add a helper with a structured report, for example:
- `compare_sparse_matrices(reference: sp.spmatrix, candidate: sp.spmatrix, *, abs_tol: float, rel_tol: float, check_structure: bool = False, max_terms: int = 20) -> dict`

Responsibilities:
1. Convert both matrices to the same sparse format, preferably CSR.
2. Check shape equality.
3. Optionally check exact sparse structure (`indptr`, `indices`).
4. Compute `delta = candidate - reference`.
5. Report:
   - reference nnz
   - candidate nnz
   - delta nnz
   - max absolute residual
   - max relative residual
   - top mismatched entries
   - overall pass/fail

Implementation style should follow the existing unresolved-delta reporting pattern in:
- `tools/parity/compare_operator_construction.py:69-136`

### 4. Add PETSc-vs-SciPy matvec parity helper

Add a helper such as:
- `compare_matvec_parity(reference_sp: sp.spmatrix, candidate_petsc: PETSc.Mat, *, seed: int = 0, trials: int = 3, tol: float = 1e-12) -> dict`

Responsibilities:
1. Draw one or more random complex vectors.
2. Compute SciPy reference action.
3. Compute PETSc action.
4. Report max relative residual across trials.

This should generalize the existing approach in:
- `tools/parity/compare_operator_construction.py:216-223`

### 5. Add a high-level dual-build harness

Add a validation entry point such as:
- `compare_lnse_builds(...) -> dict`

Suggested behavior:
1. Build the reference SciPy operator with `build_lnse_operator(...)`.
2. Build the candidate PETSc operator with `build_lnse_operator_petsc(...)`.
3. Convert PETSc -> SciPy with `petsc_to_scipy(...)`.
4. Run sparse parity comparison.
5. Run matvec parity comparison.
6. Return one structured report.

This harness is for validation and testing, not necessarily for production solver paths.

## Recommended implementation order

### Phase 1: Pre-BC temporal operator parity

Target only the pre-BC LNSE operator `L0`.

Why this first:
1. It isolates core operator assembly.
2. It avoids boundary-condition complexity.
3. It directly exercises the main SciPy assembly path in:
   - `src/linstab2d/operators/assembly.py:191-286`
   - `src/linstab2d/operators/assembly.py:357-432`

Acceptance:
1. PETSc-native pre-BC `L0` matches SciPy `L0` within agreed tolerances.
2. Sparse comparison report passes.
3. Matvec parity report passes.

### Phase 2: Dirichlet BC parity

Add PETSc-side comparison after Dirichlet enforcement.

Why second:
- the current Dirichlet implementation is highly SciPy-specific and deserves isolated validation.

Current Dirichlet reference path:
- `src/linstab2d/boundary_conditions/dirichlet.py:39-129`

Acceptance:
1. PETSc and SciPy paths produce the same BC index set.
2. Post-Dirichlet operators match within tolerance.
3. Matvec parity still passes.

### Phase 3: Neumann BC parity

Add PETSc-side comparison after Neumann enforcement.

Current Neumann reference path:
- `src/linstab2d/boundary_conditions/neumann.py:88-154`

Acceptance:
1. Post-Neumann operators match within tolerance.
2. Matvec parity passes.
3. Largest-mismatch reporting is useful enough to debug row-construction errors.

### Phase 4: Spatial operator parity

Only after temporal `L0` parity is stable, extend to:
- `Lc`
- `R`
- `Lw`

Current reference path:
- `src/linstab2d/operators/spatial.py:16-69`

Acceptance:
1. Pre-BC spatial matrices match within tolerance.
2. BC-applied spatial matrices match within tolerance.
3. PETSc-vs-SciPy matvec parity passes for each matrix.

## Comparison policy

### Canonical sparse format

Use CSR as the canonical comparison format.

Reasons:
1. rowwise assembly is natural for PETSc AIJ construction;
2. BC modifications are row-heavy;
3. CSR arithmetic is convenient for `delta = candidate - reference`.

### Comparison levels

Use two levels of validation:

1. Numerical sparse comparison
   - always enabled;
   - compares `delta`, max abs residual, max rel residual.

2. Exact structure comparison
   - optional;
   - checks `indptr` and `indices` equality after canonicalization.

The second check is useful but should not be the only criterion, because two independently assembled sparse matrices can still be mathematically identical even if internal ordering differs.

### Suggested starting tolerances

Start with:
- `abs_tol = 1e-12`
- `rel_tol = 1e-12`

If assembly-order effects require it, relax to:
- `1e-10`

This is consistent with current parity expectations used in:
- `tests/test_case1_parametric_parity.py:137-140`

## Test structure

### Unit tests

Add utility-focused tests for:
1. PETSc -> SciPy extraction on small matrices.
2. Sparse comparison helper behavior on exact matches and controlled mismatches.
3. Matvec parity helper behavior on known-equal matrices.

Suggested location:
- `tests/test_petsc_core.py` or a new PETSc utility test module.

### Parity tests

Add a dedicated parity module, for example:
- `tests/test_petsc_direct_assembly_parity.py`

Initial cases:
1. small synthetic `2D` case;
2. small synthetic `axy` case.

Each test should validate:
1. shape equality;
2. sparse parity report pass;
3. matvec parity report pass.

### Script-level harness

Either extend or mirror the existing parity script style under:
- `tools/parity/compare_operator_construction.py`

Recommended script-level outputs:
1. JSON summary of matrix parity;
2. largest mismatched entries;
3. pass/fail per matrix and per case.

## Suggested report schema

For each compared matrix:

```json
{
  "matrix_name": "L0",
  "shape": [0, 0],
  "reference_nnz": 0,
  "candidate_nnz": 0,
  "delta_nnz": 0,
  "structure_match": true,
  "max_abs_residual": 0.0,
  "max_rel_residual": 0.0,
  "matvec_max_rel_residual": 0.0,
  "largest_mismatches": [],
  "pass": true
}
```

For a full comparison case:

```json
{
  "model": "2D",
  "stage": "pre_bc",
  "matrix_reports": {},
  "overall_pass": true
}
```

## What should remain unchanged initially

Do not change these production-facing interfaces yet:
- `src/linstab2d/operators/assembly.py:22-70`
- `src/linstab2d/solvers/eigenvalue.py:148-157`
- `src/linstab2d/solvers/linsolve.py:58-60`
- `src/linstab2d/solvers/spatial_eig.py:70-72`

Rationale:
- parity infrastructure should land before any solver-path behavioral change.

## Risks and mitigation

### Risk 1: sparse structure differs even when operator action matches
Mitigation:
- treat numerical sparse delta as the primary criterion;
- keep exact structure checking optional.

### Risk 2: distributed PETSc comparison is more complex
Mitigation:
- start with sequential/local validation only;
- add gathered or ownership-block comparison later.

### Risk 3: boundary-condition parity is harder than pre-BC parity
Mitigation:
- validate pre-BC `L0` first;
- then Dirichlet;
- then Neumann.

## Acceptance criteria for this plan

1. The repository can support both a SciPy reference build and a PETSc-native candidate build for at least the temporal `L0` operator.
2. PETSc matrices can be converted into a SciPy sparse format suitable for direct comparison.
3. A structured sparse parity report and a PETSc-vs-SciPy matvec parity report exist.
4. Pre-BC `L0` parity is implemented and passing before BC parity work starts.
5. No production solver default changes happen until parity checks pass.

## Recommended next implementation step

Implement only the parity infrastructure first:
1. `petsc_to_scipy(...)`
2. sparse comparison helper
3. matvec parity helper
4. first parity tests for pre-BC `L0`

Only after those are in place should direct PETSc assembly for `L0` be implemented.
