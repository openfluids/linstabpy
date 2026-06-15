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
| `superlu_dist` (manual, not in `install.sh`) | `.venv-superlu_dist` | complex scalar PETSc/SLEPc with `--download-superlu_dist` plus METIS/ParMETIS graph-partitioning downloads | `install.sh` does not ship a `superlu_dist` build stack today. Build manually: run `./install.sh --stack superlu_dist --force` only after adding the stack branch to `install.sh`, or configure PETSc directly with `--download-superlu_dist --download-metis --download-parmetis`. The GPU variant adds `--with-cuda`. Keep this isolated to `.venv-superlu_dist`; `--force` must never target `.venv`. |

This section documents the intended build separation only. It is not a claim
that any non-default branch has been built, validated, or measured. The Line-3 GPU
FP64 decision gate is documented in `docs/GPU_FP64_GATE.md`.

### SuperLU_DIST build notes

A PETSc build with SuperLU_DIST requires the graph-partitioning libraries that
the reference stack already downloads. A minimal configure addition on top of
the default stack is:

```text
--download-superlu_dist --download-metis --download-parmetis
```

For a CUDA-enabled build, append `--with-cuda`. Because `install.sh` does not
yet wire this stack, the build must be done manually in an isolated directory
and paired with a side-by-side `.venv-superlu_dist`. Do not target `.venv`.

SuperLU_DIST has two distinct roles in `linstabpy` (see Current Status and
Mixed-Precision Factorization Knob below):

- **CPU mixed-precision iterative-refinement (live)**: FP32 factor with FP64
  iterative refinement via `pc_precision single`. This is the B.4/B.6 path and
  is not rejected.
- **Complex sparse-direct solve on the tested CUDA build (rejected)**: the
  complex SuperLU_DIST CUDA path produced order-one residuals. See Current
  Status for the finding.

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

SuperLU_DIST is the only backend that currently accepts `factor_precision:
single`. When paired with this backend, `pc_precision single` enables FP32
factorization with FP64 iterative refinement — the CPU mixed-precision role
that is live and not rejected. See Build-Stack Isolation above for the manual
build recipe.

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
installed by other writers is preserved, and with no writer at all MUMPS keeps
its default automatic ordering selection (`ICNTL(7)=7`). The legacy
`mpi.mumps_icntl_7` / `LINSTABPY_PETSC_SET_MUMPS_ICNTL_7` boolean remains
accepted for compatibility, but is deprecated: it is now only an alias for
`mumps_ordering=metis` (`ICNTL(7)=5`). Prefer `mumps_ordering: "metis"` in
JSONC or `LINSTABPY_MUMPS_ORDERING=metis` in the environment.

When both surfaces are present, explicit `mumps_ordering` wins over the legacy
alias: environment `LINSTABPY_MUMPS_ORDERING` first, then JSONC
`mumps_ordering`, then legacy `mpi.mumps_icntl_7`, then no opinion. That order is
the intended resolution precedence; today only the environment variable is
enforced at factorization time (it applies to every MUMPS LU `linstabpy` sets
up), while the JSONC `mumps_ordering` field is validated and carried in the typed
config but is not yet consumed by the solver call sites — wiring it through is
future work, as with `factor_precision` and `mumps_blr`. User-supplied PETSc CLI
options are still process-global PETSc options and should not be mixed with these
higher-level config surfaces in the same run. Set `LINSTABPY_MUMPS_ORDERING` to
one of the named values below.

| `mumps_ordering` value | PETSc/MUMPS option |
|---|---|
| `amd` | `-mat_mumps_icntl_7 0` |
| `pord` | `-mat_mumps_icntl_7 4` |
| `metis` | `-mat_mumps_icntl_7 5` |
| `qamd` | `-mat_mumps_icntl_7 6` |
| `auto` | `-mat_mumps_icntl_7 7` |

When the ordering knob is enabled with any non-MUMPS backend, `linstabpy` raises
a `ValueError` naming `LINSTABPY_MUMPS_ORDERING` and the selected backend.
Unknown ordering names also raise a `ValueError` listing the allowed values. No
performance claim is made for any ordering choice here.

## Hybrid MPI + Threaded-BLAS Placement

Use one deliberate parallel shape at a time. The reference sparse-direct MUMPS
path is pure MPI: many ranks, `OMP_NUM_THREADS=1`, and matching BLAS mirrors set
to one. That is the validated default because MUMPS parallelism comes primarily
from MPI ranks. A hybrid shape -- fewer MPI ranks with threaded BLAS inside each
rank -- is a placement option for dense-heavy phases and for future many-core
BLAS stacks such as the NVPL BLAS work tracked by the Grace-class backend path.
This section is placement guidance only; it is not a measured speedup claim.

For a workstation with `physical_cores` physical cores and `ranks` MPI ranks,
choose

```text
threads_per_rank = physical_cores // ranks
```

Do not count SMT/HT siblings as physical cores. Keep the non-oversubscription
invariant explicit:

```text
ranks * OMP_NUM_THREADS <= physical cores
```

A portable starting point is:

```bash
export OMP_NUM_THREADS=<threads_per_rank>
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
```

Use pure MPI for the current CPU/MUMPS reference path, for example one rank per
physical core and `OMP_NUM_THREADS=1`. Use hybrid placement only when the run is
expected to spend meaningful time in threaded dense kernels, or when evaluating a
threaded vendor BLAS stack; then reduce the number of ranks so that the product
of ranks and threads never exceeds the physical-core count.

A copy-paste sanity check for a workstation with `N` physical cores is:

```bash
physical_cores=N; ranks=<ranks>; threads_per_rank=$(( physical_cores / ranks )); \
OMP_NUM_THREADS=$threads_per_rank OMP_PROC_BIND=close OMP_PLACES=cores \
OPENBLAS_NUM_THREADS=$threads_per_rank MKL_NUM_THREADS=$threads_per_rank \
NUMEXPR_NUM_THREADS=$threads_per_rank PHYSICAL_CORES=$physical_cores \
mpirun -n "$ranks" python -c 'import os; from mpi4py import MPI; rank = MPI.COMM_WORLD.rank; ranks = MPI.COMM_WORLD.size; threads = int(os.environ["OMP_NUM_THREADS"]); affinity = len(os.sched_getaffinity(0)); physical_cores = int(os.environ["PHYSICAL_CORES"]); print(f"rank={rank} threads={threads} affinity={affinity} ranks*threads={ranks * threads} physical_cores={physical_cores}")'
```

Pass rule: each rank must report `OMP_NUM_THREADS <= affinity` for its CPU
affinity size, and the job-level product must satisfy
`ranks * OMP_NUM_THREADS <= physical cores`. If either condition fails, reduce
`OMP_NUM_THREADS`, reduce the rank count, or fix the launcher binding before
using the result.

## Current Status

| Backend | Status | Notes |
|---|---|---|
| `mumps` | Default/reference. | CPU sparse-direct path used for accepted solver results. Keep this path byte-identical unless a separate change explicitly updates the default solver contract. |
| `strumpack` | Experimental with automatic MUMPS fallback. | STRUMPACK can be selected through PETSc as a LU factor backend, but larger shifted EVP operators can fail during `EPS.setUp`: STRUMPACK reports `ZERO_PIVOT` (PETSc err 76, surfaced as opaque `factorization failed`) when static pivoting encounters a pivot below about `sqrt(eps)·||A||_1` (~`1e-8` in the reproduced case). Tiny-pivot replacement is off and PETSc 3.24 does not expose STRUMPACK's `replace_tiny_pivots` option. `linstabpy` catches this STRUMPACK setup failure and retries once with MUMPS, whose dynamic partial pivoting factorizes the same matrices. The STRUMPACK+FGMRES path remains diagnostic for small/probe cases. |
| `superlu_dist` | Two roles — see notes. | **CPU mixed-precision IR (live)**: `factor_precision: single` uses `pc_precision single` to enable FP32 factorization with FP64 iterative refinement; this path is not rejected and is the target of the B.4/B.6 work. **Complex+CUDA GPU solve (rejected for the tested build)**: the complex SuperLU_DIST path on the local CUDA build produced order-one solve residuals and an inconsistent EVP spectrum; changing iterative-refinement and GPU-offload flags did not fix it. The rejection applies specifically to complex sparse-direct solving on that CUDA build, not to the backend wholesale. |
| `cudss` | Not wired. | There is no `linstabpy` PETSc factor-backend interface for cuDSS today. |

The STRUMPACK result is a diagnostic result, not a production default. For the
current direct shift-invert route, STRUMPACK `ZERO_PIVOT` falls back to MUMPS
automatically and records `factor_backend_fallback="mumps"` in the returned
solver info. The diagnostic local recovery path is:

```text
ST = sinvert
KSP = fgmres
PC = lu
PCFactorSetMatSolverType = strumpack
KSP side = right
KSP rtol = 1e-10
```

Do not replace the default MUMPS `PREONLY+LU` ST path with this configuration
without a separate productization pass. A proper STRUMPACK route needs PETSc to
expose `replace_tiny_pivots` plus FGMRES recovery, then a solver-surface design,
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

- STRUMPACK `PREONLY+LU` can fail with `ZERO_PIVOT` on larger shifted EVP
  operators because static pivoting cannot reorder around a tiny pivot; this is
  not fixed by 64-bit indices and is not a pure memory-size wall.
- MUMPS factorizes the same matrices through dynamic partial pivoting, so
  `linstabpy` falls back to MUMPS automatically for this STRUMPACK setup
  failure.
- STRUMPACK `FGMRES+LU` can recover a small shifted solve and the Blasius
  temporal EVP comparison, but enabling STRUMPACK tiny-pivot replacement through
  PETSc is an upstream item for PETSc versions that expose that option.
- SuperLU_DIST complex sparse-direct CUDA solves are not reliable in the tested build. The CPU mixed-precision iterative-refinement role (FP32 factor + FP64 refine via `pc_precision single`) is a separate live path and is not part of this rejection.

Real backend adoption still needs end-to-end timings, peak memory, factorization
reuse behavior, and repeated right-hand-side costs on the actual case sizes.
