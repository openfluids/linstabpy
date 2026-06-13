# Compute Backends

`linstabpy` uses PETSc LU factorization for sparse-direct solve paths. The
default factor backend is `mumps`, and that remains the reference path for
temporal EVP, spatial EVP, resolvent, and linear-solve validation.

## Build-Stack Isolation Strategy

`install.sh` keeps one virtual environment per PETSc/SLEPc build stack. The
unselected/default stack is the CPU MUMPS reference build and continues to use
`.venv`. Non-default stacks use side-by-side directories named
`.venv-<stack>`, for example `.venv-gpu-strumpack`. This layout is deliberate:
`--force` for a non-default stack may only remove that stack's own directory and
must never delete or overwrite the validated CPU `.venv`.

Use `bash install.sh --dry-run` to inspect the resolved stack, virtual
environment path, PETSc configure branch, and any planned destructive action
without creating or deleting a virtual environment. The stack can be selected
with `--stack <name>` or `LINSTABPY_BUILD_STACK=<name>`.

| Stack | Virtual environment | PETSc configure branch | Rule |
|---|---|---|---|
| default / CPU reference | `.venv` | complex scalar PETSc/SLEPc with MUMPS, METIS, and ParMETIS downloads | This remains the only reference build. Default behavior must not change unless a separate change explicitly updates the CPU build contract. |
| `gpu-strumpack` design branch | `.venv-gpu-strumpack` | complex scalar PETSc/SLEPc with `--with-cuda` and `--download-strumpack` plus graph-partitioning dependencies | Strategy only until GPU-build recipe beads run. `--force` is isolated to `.venv-gpu-strumpack` and must refuse any action that targets `.venv`. |

This section documents the intended build separation only. It is not a claim
that the GPU branch has been built, validated, or measured. The Line-3 GPU
FP64 decision gate is documented in `docs/GPU_FP64_GATE.md`.

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

## MUMPS BLR Low-Rank Compression Knob

MUMPS block-low-rank (BLR) compression can be requested only for the MUMPS factor
backend. It is off by default, so the default MUMPS options path is unchanged.
Set `LINSTABPY_MUMPS_BLR=1` or the matching JSONC field `mumps_blr: true` in the
relevant solver subsection to enable it. The BLR dropping tolerance is controlled
by `LINSTABPY_MUMPS_BLR_TOL` or `mumps_blr_tol`; the default tolerance is
`1e-4`. Environment variables take precedence over JSONC, which takes precedence
over the defaults.

Today only the environment variables are enforced at factorization time, and
they apply to every MUMPS LU `linstabpy` sets up in the process. The JSONC
`mumps_blr` and `mumps_blr_tol` fields are validated and carried in the typed
config but are not yet consumed by the solver call sites; wiring the
per-solver config through is future work, as with `factor_precision`.

When BLR is enabled with backend `mumps`, `linstabpy` sets these PETSc options
and leaves them in the PETSc options database for the process so MUMPS can
consume them during the later factorization setup:

```text
-mat_mumps_icntl_35 1
-mat_mumps_cntl_7 <tolerance>
```

`ICNTL(35)=1` requests MUMPS BLR factorization, and `CNTL(7)` sets the BLR
dropping tolerance. No `ICNTL(36)`, `ICNTL(37)`, or `ICNTL(38)` values are set by
`linstabpy`; PETSc/MUMPS defaults are left in charge unless a later, documented
configuration surface is added. Enabling BLR with any non-MUMPS backend raises a
`ValueError` naming `LINSTABPY_MUMPS_BLR` and the selected backend.

BLR changes the numerical factorization and can affect shifted-solve residuals,
eigenvalue postchecks, and resolvent gains. Treat it as an explicit accuracy /
memory trade-off to verify for each case. No performance claim is made here.

## MUMPS Ordering Knob

MUMPS fill-reducing ordering can be requested only for the MUMPS factor backend.
It is unset by default. When unset, `linstabpy` neither sets nor clears
`mat_mumps_icntl_7` — the knob expresses no opinion, so an `ICNTL(7)` value
installed by other writers (the legacy `mpi.mumps_icntl_7` config gate, or a
user-supplied `-mat_mumps_icntl_7` option) is preserved, and with no writer at
all MUMPS keeps its default automatic ordering selection (`ICNTL(7)=7`). Set
`LINSTABPY_MUMPS_ORDERING` (enforced today; the JSONC `mumps_ordering` field is
carried but not yet consumed — see below) to one of the named values below.
When set, it overwrites any externally installed `ICNTL(7)` value. The intended
precedence is env over JSONC over the unset default.

| `mumps_ordering` value | PETSc/MUMPS option |
|---|---|
| `amd` | `-mat_mumps_icntl_7 0` |
| `pord` | `-mat_mumps_icntl_7 4` |
| `metis` | `-mat_mumps_icntl_7 5` |
| `qamd` | `-mat_mumps_icntl_7 6` |
| `auto` | `-mat_mumps_icntl_7 7` |

Today only the environment variable is enforced at factorization time, and it
applies to every MUMPS LU `linstabpy` sets up in the process. The JSONC
`mumps_ordering` field is validated and carried in the typed config but is not
yet consumed by the solver call sites; wiring the per-solver config through is
future work, as with `factor_precision` and `mumps_blr`.

When the ordering knob is enabled with any non-MUMPS backend, `linstabpy` raises
a `ValueError` naming `LINSTABPY_MUMPS_ORDERING` and the selected backend.
Unknown ordering names also raise a `ValueError` listing the allowed values. No
performance claim is made for any ordering choice here.

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
