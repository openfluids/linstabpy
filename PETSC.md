# PETSc/SLEPc Notes for `linstab2d` (Hamada case)

Date: 2026-02-26  
Scope: `example_6_hamada` operator structure, solver behavior, and PETSc/SLEPc best-practice alignment.

## 1) Measured operator diagnostics (current Python assembly)

Matrix analyzed: `L0` after BC application (`beta=0`, `fd_order=4`, `170x150` grid, `5` variables).

- shape: `127500 x 127500`
- nnz: `5,678,509`
- density: `3.493124e-04` (`0.0349%`)
- row nnz stats: `min=1`, `p10=25`, `p50=44`, `p90=68`, `max=68`, `mean=44.54`
- BC rows: `3180` (`2.49%` of DOFs)
- asymmetry sample metric: `|| (A-A^T)x || / ||Ax|| = 1.112e+00` (strongly non-symmetric / non-normal)
- strict diagonal dominance fraction: `0.019` (weak also `0.019`)

Diagonal magnitude spread:

- `min_nonzero = 1.387e-05`
- `p50 = 1.647e+01`
- `p90 = 2.005e+04`
- `p99 = 1.000e+08`
- `max = 3.547e+12`
- BC diagonal median: `1.000e+08`
- interior diagonal median: `1.422e+01`

Interpretation: the system is sparse and highly non-normal with severe scaling disparity (mostly BC-penalty driven).

## 2) Block structure check (rho,u,v,w,T)

Variable block sizes are uniform: `25500` each.

5x5 nonzero block pattern (rows/cols in order `rho,u,v,w,T`):

```text
222940 221120 221120      0      0
221120 610210 608390      0 221564
221120 608390 610210      0 221564
     0      0      0 610210      0
 24864 221120 221120      0 613447
```

Key points:

- not a classic incompressible saddle-point matrix (`[A B^T; B 0]`) because diagonal physics blocks are populated
- `w` is mostly decoupled in this 2D (`beta=0`) setup
- strong coupled block exists for `rho/u/v/T`

## 3) Practical conclusion from observed runs

Validated runs now show both robust direct and a working distributed iterative path:

- MUMPS baseline (`SINVERT + LU/MUMPS`, `MPI=1`, `threads=6`, `n_modes=30`)
  - converged in about `98 s`
  - spectrum parity pass
  - log: `results/case1/output/logs/example_6_hamada_20260226_085632.log`
- Iterative ASM (`SINVERT + GMRES/ASM/sub-LU`, `MPI=1`, `threads=6`, `n_modes=30`)
  - converged in about `217 s`
  - spectrum parity pass
  - log: `results/case1/output/logs/example_6_hamada_20260226_085247.log`
- Iterative ASM distributed (`SINVERT + GMRES/ASM/sub-LU`, `MPI=2`, `threads/rank=3`, `n_modes=30`)
  - converged in about `185 s`
  - spectrum parity pass
  - operator-stage matrix parity pass
  - log: `results/case1/output/logs/example_6_hamada_20260226_115755.log`
- One JD/PRECOND trial (`FGMRES+bjacobi+sub-LU`, `MPI=2`) still stalled (`nconv=0` at ~320 s)
  - log: `results/case1/output/logs/example_6_hamada_20260226_092026.log`

Current conclusion: keep MUMPS as robust baseline, and use the iterative MPI path on the now-validated SINVERT+ASM configuration while continuing preconditioner improvements.

## 4) Cross-check with PETSc/SLEPc documentation (official)

### SLEPc spectral transform behavior

- `STSINVERT` solves linear systems `(A - sigma B) y = ...` at eigensolver steps.  
  Ref: https://slepc.upv.es/release/manualpages/ST/STSINVERT.html
- `STPRECOND` is intended for preconditioned eigensolvers (e.g. JD), not an exact transform.  
  Ref: https://slepc.upv.es/release/manualpages/ST/STPRECOND.html
- ST docs explicitly state users should tune ST/KSP/PC through the EPS-owned ST object/options.  
  Ref: https://slepc.upv.es/release/documentation/manual/st.html
- `-eps_true_residual` is recommended when residual estimates can be inaccurate under transforms.  
  Ref: https://slepc.upv.es/release/manualpages/EPS/EPSSetTrueResidual.html

### PETSc linear-solver and preconditioner behavior

- `KSPSetFromOptions()` must happen before setup for CLI options to actually apply.  
  Ref: https://petsc.org/release/manualpages/KSP/KSPSetFromOptions/
- `-ksp_error_if_not_converged` is useful to fail fast and avoid silently accepting bad inner solves.  
  Ref: https://petsc.org/release/manualpages/KSP/KSPSetErrorIfNotConverged/
- `PCASM` overlap and block controls are central for domain-decomposition tuning (`-pc_asm_overlap`, `-pc_asm_blocks`).  
  Ref: https://petsc.org/release/manualpages/PC/PCASMSetOverlap/
- `PCFIELDSPLIT` is the right tool for block-physics preconditioning (not tied only to saddle-point systems).  
  Ref: https://petsc.org/main/manualpages/PC/PCFIELDSPLIT/
- saddle-point auto-detect in `PCFIELDSPLIT` is based on zero diagonals; this does not match the current operator.  
  Ref: https://petsc.org/main/manualpages/PC/PCFieldSplitSetDetectSaddlePoint/
- `PCILU` is only implemented for some formats/sequential usage; parallel recommendation points to alternatives (e.g. hypre ILU).  
  Ref: https://petsc.org/main/manualpages/PC/PCILU/
- `KSPSetDiagonalScale` can help scaling, but it mutates the system and must be used carefully.  
  Ref: https://petsc.org/main/manualpages/KSP/KSPSetDiagonalScale/

### MUMPS backend in PETSc

- PETSc `MATSOLVERMUMPS` supports sequential and MPI distributed AIJ factorizations and can run hybrid MPI+OpenMP.  
  Ref: https://petsc.org/release/manualpages/Mat/MATSOLVERMUMPS/
- standard direct setup path in PETSc is `-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps`.  
  Ref: https://petsc.org/release/manualpages/PC/PCLU/  
  Ref: https://petsc.org/release/manualpages/PC/PCFactorSetMatSolverType/

## 5) Recommended next technical path (iterative hardening)

1. Keep the validated distributed baseline fixed:
   - `ST=SINVERT`, `KSP=GMRES`, `PC=ASM`, `sub-KSP=PREONLY`, `sub-PC=LU`
   - enforce parity and convergence gates on every change

2. Preserve root-owned SciPy->PETSc transfer for MPI iterative runs:
   - rank 0 assembles SciPy operator
   - PETSc distributed AIJ receives row-owned CSR blocks by scatter
   - avoids per-rank full-matrix transfer path in iterative mode

3. Add explicit scaling/equilibration experiments:
   - evaluate diagonal/row scaling impact on ST inner solve convergence and parity

4. Continue `PCFIELDSPLIT` as optimization track:
   - physics split `{rho,u,v,T}` vs `{w}`
   - only promote when robustness and performance are both better than baseline

5. Keep strict failure guards:
   - `ksp_error_if_not_converged`
   - `eps_true_residual` for parity-critical runs

## 6) Immediate option templates

Robust direct reference:

```bash
-st_type sinvert -st_ksp_type preonly -st_pc_type lu -st_pc_factor_mat_solver_type mumps
```

Iterative candidate baseline (to improve):

```bash
-st_type sinvert \
-st_ksp_type gmres -st_ksp_rtol 1e-8 -st_ksp_max_it 20000 -st_ksp_error_if_not_converged \
-st_pc_type asm -st_pc_asm_overlap 2 \
-st_sub_ksp_type preonly -st_sub_pc_type lu \
-eps_true_residual
```

Field-split direction (next step):

```bash
-st_ksp_type fgmres -st_pc_type fieldsplit
```

Then define splits from index sets in code (`rho/u/v/w/T`) and tune:

- `-pc_fieldsplit_type additive|multiplicative|schur`
- `-fieldsplit_*` inner KSP/PC choices
