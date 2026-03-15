# MPI Option B Validation Plan

Date: 2026-03-14

## Goal

Deliver a minimal **Option B** path that is proven on real MPI for:

- distributed 2D PETSc assembly on **2 and 4 ranks**
- PETSc Dirichlet and Neumann BC application on **2 and 4 ranks**
- PETSc linear solves with **distributed PETSc Vecs**

This plan does **not** include a distributed resolvent-shell redesign.

## What already exists

- `build_lnse_system(...)` already routes through SciPy or PETSc:
  `src/linstab2d/operators/assembly.py:211-293`
- Direct distributed 2D PETSc assembly already exists:
  `src/linstab2d/operators/assembly.py:566-686`
- PETSc Dirichlet and Neumann BC paths already exist:
  `src/linstab2d/boundary_conditions/dirichlet.py:132-157`
  `src/linstab2d/boundary_conditions/neumann.py:203-246`
- The remaining gap is **real multi-rank proof** plus a **PETSc-native linear solve path**.

## Acceptance

Option B is complete only when all of the following are true:

1. A real **2-rank** MPI harness passes for distributed 2D PETSc assembly.
2. A real **4-rank** MPI harness passes for distributed 2D PETSc assembly.
3. Ownership is actually split across ranks in those runs.
4. Dirichlet PETSc BC succeeds under MPI.
5. Neumann PETSc BC succeeds under MPI.
6. Owned rows can be inspected and checked against a SciPy reference built outside the distributed PETSc path.
7. The linear solver exposes PETSc-native `Vec` solves.
8. A real **2-rank** MPI linear-solver harness passes with distributed PETSc Vecs.
9. A real **4-rank** MPI linear-solver harness passes with distributed PETSc Vecs.

## Plan

### Phase 1 — Add an owned-row inspection helper

Add a helper in `src/linstab2d/utils/petsc_core.py` that returns only the rows owned by the current rank.

Target shape:

- `petsc_owned_rows(mat: PETSc.Mat) -> dict`

Expected contents:

- ownership range
- local row ids
- CSR `indptr`
- CSR column indices
- CSR values

Implementation notes:

- keep the helper in `linstab2d.utils.petsc_core`
- do **not** export it through `src/linstab2d/utils/__init__.py`
- prefer wrapping PETSc local CSR access instead of inventing a second matrix traversal path

Validation:

- add focused coverage in `tests/test_utils.py`
- confirm the helper works for both sequential and distributed AIJ matrices

### Phase 2 — Add real MPI assembly and BC proof harnesses

Add subprocess-based MPI harnesses with `mpiexec`.

Use a small 2D case through:

- `build_lnse_system(..., backend="petsc", solver_cfg={"petsc_matrix_mode": "distributed"})`

Checks:

- operator builds successfully on 2 and 4 ranks
- ownership is split
- owned rows match a SciPy reference built separately
- supported 2D runs use the direct PETSc path, not the SciPy-to-PETSc fallback
- Dirichlet BC passes under MPI
- Neumann BC passes under MPI

Neumann remains the higher-risk BC case.

### Phase 3 — Add PETSc-native linear solves

Extend `src/linstab2d/solvers/linsolve.py` so the solver object supports PETSc-native solves in addition to the current NumPy API.

Target additions:

- `solve_vec(rhs: PETSc.Vec) -> PETSc.Vec`
- `solve_transpose_vec(rhs: PETSc.Vec) -> PETSc.Vec`

Requirements:

- keep the existing NumPy-facing API for compatibility
- avoid `petsc_vec(...)` and `vec_to_numpy(...)` in the PETSc-native path
- keep distributed RHS and solution vectors distributed

This phase is required for **real MPI correctness** in the solve path, not just for performance cleanup.

### Phase 4 — Run the proof set

Required validation:

1. `tests/test_utils.py` coverage for the owned-row helper
2. 2-rank MPI assembly harness
3. 4-rank MPI assembly harness
4. 2-rank MPI Dirichlet harness
5. 4-rank MPI Dirichlet harness
6. 2-rank MPI Neumann harness
7. 4-rank MPI Neumann harness
8. 2-rank MPI linear-solver harness
9. 4-rank MPI linear-solver harness
10. Existing local suite:
   - `tests/test_utils.py`
   - `tests/test_operators.py`
   - `tests/test_solvers.py`
   - `tests/test_case1_smoke.py`

## Out of scope

- distributed resolvent-shell redesign
- distributed vector-native resolvent Krylov flow
- 8-rank scaling claims

## Definition of done

This plan is done when we can truthfully say:

> The distributed 2D PETSc path is proven on 2 and 4 ranks for assembly, BC application, and linear solves with distributed PETSc Vecs.

## Immediate next step

Start with **Phase 1 only**:

1. add `petsc_owned_rows(...)` in `src/linstab2d/utils/petsc_core.py`
2. add focused tests in `tests/test_utils.py`
3. validate locally before adding MPI harnesses
