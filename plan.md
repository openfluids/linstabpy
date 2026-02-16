# linstab2d — Project Plan

**Last updated**: 2026-02-16
**Status**: Core functionality complete. PETSc/SLEPc migration done for direct solves and eigenvalue problems.
**Test suite**: 321 tests, ~12 seconds.

---

## Completed Work

### Part 1: Bug Fixes (commit 566f9ac)

Fixed 5 bugs that prevented examples 2–5 from running:

| # | Bug | Fix |
|---|-----|-----|
| 1 | `c2p_idx` permutation swapped v_r/v_θ | Correct mapping: `[0, 2, 3, 1, 4]` |
| 2 | Parity tuples in coefficient ordering | Physical ordering `[ρ,u,v,w,T]` |
| 3 | Penalty on R matrix created spurious eigenvalues | Penalty only on Lc; R gets zero rows/cols |
| 4 | `delta=1e-6` caused catastrophic cancellation in spatial EVP | `delta=1.0` (exact for quadratic α-dependence) |
| 5 | Generalized EVP zero-norm eigenvectors | Standard EVP via `eigs(inv(A-σR)·R)` |

### Part 2: PETSc/SLEPc Migration (commit 9f7c120)

| Solver | Before | After |
|--------|--------|-------|
| Direct solves | SciPy SuperLU | PETSc KSP + MUMPS (`linsolve.py`) |
| Temporal eigenvalues | SciPy ARPACK | SLEPc EPS Krylov-Schur + SINVERT (`eigenvalue.py`) |
| Adjoint eigenvalues | SciPy ARPACK | SLEPc EPS Krylov-Schur + SINVERT (`eigenvalue.py`) |
| Spatial eigenvalues | SciPy ARPACK | SLEPc EPS GNHEP + SINVERT (`spatial_eig.py`) |
| Resolvent analysis | SciPy SuperLU + ARPACK | MUMPS factorization + SciPy ARPACK (`resolvent.py`) |

### Part 3: Optimization & Audit (Plan 5)

| # | Change | Files |
|---|--------|-------|
| 1 | Cylindrical `inv(B)` analytical block inverse | `assembly.py` |
| 2 | MUMPS METIS ordering (ICNTL(7)=5) | `petsc_utils.py`, `linsolve.py` |
| 3 | Kronecker product fast path for unmasked meshes | `mesh.py` |
| 4 | SBP `lru_cache` | `sbp.py` |
| 5 | Spatial operator 1× coefficient computation | `assembly.py`, `spatial.py` |
| 6 | Resolvent diagonal shift + format cleanup | `resolvent.py`, `linsolve.py` |

### Part 4: Test Coverage Expansion (Plan 5b)

312 tests across 5 files. Key additions:

- **Parity tests** (7): m=0, m=1, m≥2 selection, fallback, axial derivatives
- **inv(B) block inverse** (3): algebraic identity `-inv(B)·B = -I`, Cartesian diagonal, uniform flow
- **Cylindrical temporal eigenvalues** (4): eigenpair residuals for m=0/m=1, finiteness, shape
- **Cylindrical spatial eigenvalues** (3): shape, finiteness, sparsity
- **Cylindrical integration** (5): full pipeline, eigenpair residual, reference regression, adjoint, energy norm
- **Numerical soundness** (11): grid convergence, resolvent bounds/orthonormality, spectrum properties, adjoint consistency, inv(B) cylindrical→Cartesian limit, spatial EVP block structure

### Part 5: Analytical Benchmark Tests (Plan 6)

9 tests in `tests/test_analytical.py` comparing eigenvalues against classical results from hydrodynamic stability theory. Uses `floquet_exp=α` on periodic x-meshes to reduce 2D global problems to 1D Orr-Sommerfeld equivalents.

**Eigenvalue convention discovery**: `ω_code = eig(iL0) = -ω_OS`. Unstable when `Im(ω_code) < 0`. The assembly docstring `dq/dt = L0·q` is incorrect — the actual PDE is `dq/dt = -L0·q`.

| Class | Tests | Reference |
|-------|-------|-----------|
| **TestPoiseuilleFlow** (3) | Orszag eigenvalue to 1%, grid convergence (ny=64→256), stable below Re_c≈5772 | Orszag (1971), JFM 50:689–703 |
| **TestCouetteFlow** (1) | All modes damped at Re=2000 (negative test) | Romanov (1973) |
| **TestMixingLayerKH** (3) | KH instability at α=0.4, phase speed ≈ 0.5, stable at α=1.5 | Michalke (1964), JFM 19:543–556 |
| **TestUniformFlowEigenvalues** (2) | Convective cluster near ω≈αU, all modes damped | Compressible dispersion relation |

---

## Remaining Work

### A. SLEPc SVD for Resolvent [HIGH — performance]

**Current**: MUMPS factorization (PETSc KSP) + SciPy ARPACK for the top-level eigensolve of the composite operator `W⁻¹ H^H W H`.

**Target**: SLEPc SVD (TRLANCZOS or CROSS) with matrix-free shell wrapping the MUMPS factorization. Avoids the explicit `H^H W H` composite and works directly on the transfer function `H(ω) = C·(iωI - L0)⁻¹·B`.

**Why**: SLEPc SVD is purpose-built for singular value decomposition of large operators. It returns singular triplets (σ, u, v) directly — no need for the composite resolvent operator. Expected to be more numerically robust and faster for large problems.

**Files**: New `linstab2d/solvers/resolvent_slepc.py` or extend `resolvent.py`.

### B. SLEPc PEP for Spatial EVP [HIGH — memory]

**Current**: Companion linearization doubles the DOF count (10×ngp instead of 5×ngp). For ngp=150k this means ~5 GB extra matrix storage.

**Target**: SLEPc PEP (Polynomial Eigenvalue Problem) with TOAR method. Operates on `(R0 + α·R1 + α²·R2)q = 0` natively — no companion matrix needed.

**Why**: Halves memory for spatial eigenvalue problems. TOAR is mathematically equivalent to the companion linearization but avoids explicit doubling.

**Files**: New `linstab2d/solvers/spatial_slepc.py` or extend `spatial_eig.py`.

### C. Backend CLI [MEDIUM — usability]

Add `--petsc` / `--slepc` / `--both` flags to all 5 examples via a shared argument parser. `--both` runs SciPy then PETSc/SLEPc, compares results, and reports timing.

**Files**: New `linstab2d/solvers/backend.py` with `Backend` enum + `parse_backend_args()`. Wire into each example's `main()`.

### D. Timing Infrastructure [MEDIUM — diagnostics]

`SolveTimings` dataclass tracking mesh/assembly/factorization/solve/total times + peak RSS. `Timer` context manager. JSON save/load per example.

**Files**: New `linstab2d/solvers/timing.py`.

### E. BAIJ(5) Block Format [LOW — performance]

PETSc BAIJ with block size 5 exploits the 5-variable structure: 5× less index storage, BLAS-2 on dense 5×5 blocks, better cache utilization. Requires PETSc Mat assembly path.

### F. MPI Parallelism [LOW — scalability]

Row-distributed PETSc Mat across MPI ranks. MUMPS distributed factorization. SLEPc distributed eigensolves. Rank 0 reads baseflow, broadcasts.

### G. Benchmark at Scale [LOW — informational]

Standalone timing script comparing assembly paths and solver backends across grid sizes (50×20 through 500×100). No code changes required.

---

## Implementation Order

```
A. SLEPc SVD for resolvent         ← biggest user-facing improvement
B. SLEPc PEP for spatial EVP       ← biggest memory improvement
C. Backend CLI + D. Timing          ← enables A/B comparison with --both
E. BAIJ(5)                          ← requires PETSc Mat assembly path
F. MPI                              ← final scalability layer
G. Benchmark                        ← measure everything
```

Items A and B are independent and can be tackled in either order. C+D should follow to enable `--both` comparison mode. E and F are longer-term goals.
