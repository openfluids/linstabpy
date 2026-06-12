# Compute Backends

`linstabpy` uses PETSc LU factorization for sparse-direct solve paths. The
default factor backend is `mumps`, and that remains the reference path for
temporal EVP, spatial EVP, resolvent, and linear-solve validation.

Backend selection precedence is:

1. JSONC solver subsection `factor_backend`;
2. existing per-solver environment override, such as
   `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE`;
3. `LINSTABPY_FACTOR_BACKEND`;
4. the call-site default.

Requested backends are checked with a tiny complex PETSc LU setup probe. An
unavailable backend fails loudly instead of silently falling back.

## Mixed-Precision Factorization Knob

`LINSTABPY_FACTOR_PRECISION` and the matching JSONC `factor_precision` field
select the factor precision surface. The default is `double`, which is inert and
does not add any PETSc options. The experimental value `single` is accepted only
with the `superlu_dist` factor backend, where it inserts the PETSc option
`pc_precision single` alongside `pc_factor_mat_solver_type`. Any other backend
with `single` raises a `ValueError` naming the backend and
`LINSTABPY_FACTOR_PRECISION`; it is never silently ignored.

Today only the environment variable is enforced at factorization time. The
JSONC `factor_precision` field is validated and carried in the typed config
but is not yet consumed by the solver call sites; the Phase B.4
mixed-precision iterative-refinement work wires it through.

This knob only exposes configuration for later mixed-precision iterative
refinement work. It is not a production recommendation, and `double` remains the
reference/default path.

## Current Status

| Backend | Status | Notes |
|---|---|---|
| `mumps` | Default/reference. | CPU sparse-direct path used for accepted solver results. Keep this path byte-identical unless a separate change explicitly updates the default solver contract. |
| `strumpack` | Experimental. | STRUMPACK can be selected through PETSc as a LU factor backend. On the tested CUDA build, `KSPPREONLY+PCLU+STRUMPACK` is not accurate enough for the near-singular shifted EVP operator because static pivoting leaves a high residual. Wrapping the same LU preconditioner in right-preconditioned FGMRES recovers the shifted solve to about `4e-13` relative residual on the probe RHS and gives a Blasius temporal EVP spectrum within about `1e-10` of the MUMPS reference. |
| `superlu_dist` | Rejected for the tested CUDA build. | The complex SuperLU_DIST path produced order-one solve residuals and an inconsistent EVP spectrum in the local CUDA build. Changing iterative-refinement and GPU-offload flags did not fix the complex solve. |
| `cudss` | Not wired. | There is no `linstabpy` PETSc factor-backend interface for cuDSS today. |

The STRUMPACK result is a diagnostic result, not a production default. The
accurate local path is:

```text
ST = sinvert
KSP = fgmres
PC = lu
PCFactorSetMatSolverType = strumpack
KSP side = right
KSP rtol = 1e-10
```

Do not replace the default MUMPS `PREONLY+LU` ST path with this configuration
without a separate productization pass. That pass needs a solver-surface design,
failure handling, and performance measurements on the target hardware.

## Correctness Gate

Backend choice does not decide correctness. Stability acceptance is enforced by
the existing postcheck gate in `linstabpy.utils.stability_acceptance` using the
explicit full-operator residual and the target-distance gate. A backend may
converge according to PETSc/SLEPc and still be rejected if the postcheck fails.

The STRUMPACK+FGMRES comparison test is therefore diagnostic:

- CPU MUMPS build: MUMPS reference solve passes, STRUMPACK branch skips when the
  backend is unavailable.
- CUDA STRUMPACK build: STRUMPACK+FGMRES solve passes against the stored MUMPS
  reference spectrum, while MUMPS skips if that PETSc build lacks MUMPS.

This split is intentional because the tested CUDA STRUMPACK environment does not
also provide MUMPS. A same-build MUMPS-vs-STRUMPACK comparison should replace
the stored reference check when a PETSc build contains both backends.

## What Is Deferred

No speedup claim is made here. The evidence so far is numerical:

- STRUMPACK `PREONLY+LU` is not acceptable for this shifted EVP operator.
- STRUMPACK `FGMRES+LU` can recover a small shifted solve and the Blasius
  temporal EVP comparison.
- SuperLU_DIST complex CUDA solves are not reliable in the tested build.

Real backend adoption still needs end-to-end timings, peak memory, factorization
reuse behavior, and repeated right-hand-side costs on the actual case sizes.
