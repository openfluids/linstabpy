# MPI Option B Validation Plan

Date: 2026-03-14

## Goal

Deliver an **Option B** distributed PETSc path that is:

- correct under real MPI on **2 and 4 ranks**
- useful on **4-8 ranks** for moderate-to-large problems
- no longer bottlenecked by NumPy replication in the **linear solve** path
- still allowed to defer a full distributed resolvent-shell redesign

## Current status

### Already strong
- Unified backend-aware LNSE workflow exists in `src/linstab2d/operators/assembly.py:211-293`
- Direct PETSc LNSE assembly exists in `src/linstab2d/operators/assembly.py:583-707`
- PETSc Dirichlet BC path exists in `src/linstab2d/boundary_conditions/dirichlet.py:132-157`
- PETSc Neumann BC path exists in `src/linstab2d/boundary_conditions/neumann.py:203-246`
- PETSc-aware solver entry points already exist in:
  - `src/linstab2d/solvers/linsolve.py:186-205`
  - `src/linstab2d/solvers/resolvent.py:216-222`
- The local/sequential parity suite is already broad and strong.

### Remaining gap
The remaining proof gap is **real multi-rank MPI validation** plus removal of the worst **distributed-vector bottleneck** in the linear solve path.

## Confirmed bottlenecks

### Matrix/path correctness gap
The code has a distributed direct assembly branch, but we still need actual `mpiexec` proof:
- backend routing and workflow helper: `src/linstab2d/operators/assembly.py:211-293`
- direct distributed block insertion: `src/linstab2d/operators/assembly.py:635-672`
- distributed sponge handling: `src/linstab2d/operators/assembly.py:494-520`

### Distributed vector bottlenecks
These remain the main scalability limiters:
- PETSc/Numpy vector helpers still serialize global vectors: `src/linstab2d/utils/petsc_core.py:564-593`
- Linear solver interface is still NumPy-in / NumPy-out: `src/linstab2d/solvers/linsolve.py:186-205`
- Resolvent shell matvec still extracts global NumPy vectors: `src/linstab2d/solvers/resolvent.py:216-222`

## Option B acceptance criteria

We can call Option B complete only when all of the following are true.

### A. MPI correctness proof
1. A real **2-rank** MPI harness passes for distributed 2D PETSc assembly.
2. A real **4-rank** MPI harness passes for distributed 2D PETSc assembly.
3. Ownership is actually split across ranks in those runs.
4. Dirichlet PETSc BC succeeds under MPI.
5. Neumann PETSc BC succeeds under MPI.
6. Owned-row parity can be inspected and checked against a trustworthy reference.

### B. MPI-useful linear solve
7. The linear solver exposes a PETSc-native Vec interface, so distributed callers do not need NumPy gather/scatter.
8. A real **2-rank** MPI linear-solver smoke test passes using distributed PETSc Vecs.
9. A real **4-rank** MPI linear-solver smoke test passes using distributed PETSc Vecs.

### Explicitly deferred
- Full distributed resolvent-shell redesign is deferred.
- Full distributed vector-native Krylov flow in the resolvent path is deferred.
- 8-rank stress testing is optional after 2-rank and 4-rank are stable.

## Execution plan

## Phase 1 — Add a distributed owned-row inspection helper

### Purpose
Enable strong MPI validation without relying on the current sequential-only PETSc-to-SciPy extractor.

### Why first
`petsc_to_scipy(...)` is still sequential-only: `src/linstab2d/utils/petsc_core.py:320-363`

### Deliverable
Add a small helper in `src/linstab2d/utils/petsc_core.py` that extracts **owned rows only** from a PETSc matrix into a Python-serializable structure, e.g.:
- ownership range
- row ids
- column indices
- row values

### Suggested shape
A helper like:
- `petsc_owned_rows(mat: PETSc.Mat) -> dict`

That helper should work for both sequential and distributed PETSc AIJ matrices.

### Validation
- add focused unit coverage in `tests/test_utils.py`
- verify local structure matches expected rows on a small PETSc matrix

## Phase 2 — Add real MPI assembly/BC proof harnesses

### Purpose
Move from single-rank structural confidence to real MPI proof.

### Harness style
Use a real subprocess-based MPI harness with `mpiexec`, not ordinary in-process pytest.

### First target
A small **2D** case through:
- `build_lnse_system(..., backend="petsc", solver_cfg={"petsc_matrix_mode": "distributed"})`
  at `src/linstab2d/operators/assembly.py:211-293`

### Checks
#### 2-rank harness
- distributed PETSc operator builds successfully
- ownership is actually split
- owned rows match the corresponding SciPy reference rows
- direct PETSc path is used for supported distributed 2D assembly

#### 4-rank harness
- same checks as above
- specifically catches finer ownership fragmentation bugs

### BC extension
Add harness cases for:
- Dirichlet PETSc BC: `src/linstab2d/boundary_conditions/dirichlet.py:132-157`
- Neumann PETSc BC: `src/linstab2d/boundary_conditions/neumann.py:203-246`

Neumann should be treated as the higher-risk case.

## Phase 3 — Add PETSc-native linear solver Vec interface

### Purpose
Remove the worst distributed-vector replication bottleneck without redesigning the resolvent shell yet.

### Current bottleneck
`make_linear_solver(...)` currently returns NumPy-facing closures only: `src/linstab2d/solvers/linsolve.py:186-205`

### Deliverable
Extend the linear solver object with PETSc-native methods alongside the existing NumPy API, e.g.:
- `solve_vec(rhs: PETSc.Vec) -> PETSc.Vec`
- `solve_transpose_vec(rhs: PETSc.Vec) -> PETSc.Vec`

### Requirements
- preserve the current NumPy API for backward compatibility
- avoid `petsc_vec(...)` / `vec_to_numpy(...)` in the PETSc-native path
- ensure distributed RHS and solution vectors remain distributed

## Phase 4 — Add MPI linear-solver smoke tests

### Purpose
Prove the distributed PETSc path is not just assembly-correct, but solver-useful for Option B.

### Checks
Run under:
- 2 ranks
- 4 ranks

### Test shape
- build distributed 2D PETSc operator
- create distributed PETSc RHS Vec
- solve with the new PETSc-native linear solver path
- compare against a reference residual and/or a trustworthy SciPy/local result

### Relevant code
- workflow build: `src/linstab2d/operators/assembly.py:211-293`
- linear solver closure area to extend: `src/linstab2d/solvers/linsolve.py:186-205`

## Phase 5 — Validation sweep

### Required runs
1. Focused utility/unit tests for the owned-row helper
2. 2-rank MPI assembly harness
3. 4-rank MPI assembly harness
4. 2-rank MPI Dirichlet harness
5. 4-rank MPI Dirichlet harness
6. 2-rank MPI Neumann harness
7. 4-rank MPI Neumann harness
8. 2-rank MPI linear-solver smoke
9. 4-rank MPI linear-solver smoke
10. Existing broad local suite

### Broad local suite to keep green
- `tests/test_utils.py`
- `tests/test_operators.py`
- `tests/test_solvers.py`
- `tests/test_case1_smoke.py`

## Risks and watch items

### Highest-risk technical areas
- distributed owned-row slicing in direct assembly: `src/linstab2d/operators/assembly.py:635-672`
- distributed sponge diagonal handling: `src/linstab2d/operators/assembly.py:494-520`
- Neumann row replacement under ownership splitting: `src/linstab2d/boundary_conditions/neumann.py:226-245`
- PETSc Vec replication bottleneck in linear solve: `src/linstab2d/solvers/linsolve.py:186-205`

### Known deferred bottleneck
The resolvent shell still serializes vectors through NumPy in `_HtrHShell.mult()`: `src/linstab2d/solvers/resolvent.py:216-222`

This is explicitly **not** part of Option B completion.

## Definition of done for this plan

This plan is complete when we can truthfully say:

> The distributed 2D PETSc path is proven on 2 and 4 ranks for assembly and BC application, and the linear solve path can run on distributed PETSc Vecs without mandatory NumPy replication.

## Immediate next action

Start with **Phase 1 only**:
1. add the owned-row inspection helper in `src/linstab2d/utils/petsc_core.py`
2. export it through `src/linstab2d/utils/__init__.py`
3. add focused unit tests in `tests/test_utils.py`
4. run targeted validation before touching the MPI harness
