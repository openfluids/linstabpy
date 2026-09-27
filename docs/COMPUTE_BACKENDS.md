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
| `gpu-strumpack` design branch | `.venv-gpu-strumpack` | complex scalar PETSc/SLEPc with `--with-cuda` and `--download-strumpack` plus graph-partitioning dependencies | Strategy only until the GPU-build recipes are run. `--force` is isolated to `.venv-gpu-strumpack` and must refuse any action that targets `.venv`. |
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

1. YAML solver subsection `factor_backend`;
2. existing per-solver environment override, such as
   `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE`;
3. `LINSTABPY_FACTOR_BACKEND`;
4. the call-site default.

Requested backends are checked with a tiny complex PETSc LU setup probe. An
unavailable backend fails loudly instead of silently falling back.

### NVPL CPU BLAS/LAPACK (Grace aarch64) build notes

NVIDIA Performance Libraries (NVPL) provide BLAS and LAPACK implementations
optimised for the Grace Armv9.0-A architecture. Linking PETSc/SLEPc against
NVPL at build time is the highest-ROI, zero-code path for accelerating dense
CPU operations (SBP assembly, Padé construction, BLAS-heavy solver phases) on
Grace aarch64 hardware. No code change is required; the linkage is entirely
a configure-time choice.

**When to use this:** only on a Grace aarch64 workstation with NVPL installed.
The recipe is written here as a preparation step; on-hardware activation and
verification are deferred to Grace hardware.

**Install NVPL.** Via the OS package manager (places libraries on standard
system paths, no further path configuration needed) or by extracting the
standalone tarball and setting `nvpl_ROOT`:

```bash
export nvpl_ROOT=/path/to/nvpl
export LD_LIBRARY_PATH="${nvpl_ROOT}/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${nvpl_ROOT}/lib:${LIBRARY_PATH}"
```

**Library variants.** NVPL ships four BLAS interface libraries and matching
LAPACK libraries; choose one BLAS and one LAPACK:

| Integer width | Threading | BLAS library | LAPACK library |
|---|---|---|---|
| LP64 (32-bit, default) | sequential | `libnvpl_blas_lp64_seq` | `libnvpl_lapack_lp64_seq` |
| LP64 | OpenMP (GCC gomp) | `libnvpl_blas_lp64_gomp` | `libnvpl_lapack_lp64_gomp` |
| ILP64 (64-bit) | sequential | `libnvpl_blas_ilp64_seq` | `libnvpl_lapack_ilp64_seq` |
| ILP64 | OpenMP (GCC gomp) | `libnvpl_blas_ilp64_gomp` | `libnvpl_lapack_ilp64_gomp` |

Each interface library must be paired with the shared core: `libnvpl_blas_core`
and `libnvpl_lapack_core`.

Only the LP64-sequential pair is reproduced from a published worked PETSc
configure example; the other three rows follow NVPL's documented
`libnvpl_{blas,lapack}_<int_type>_<threading_type>` naming convention but were
not each seen in a worked example. Confirm the exact `.so` name on the target
system (`ls "${nvpl_ROOT}/lib"`) before committing to an ILP64 or `_gomp`
variant.

For the default `linstabpy` case (32-bit PETSc indices, LP64 BLAS/LAPACK,
single-threaded BLAS within each MPI rank) the sequential LP64 pair is the
right choice. For a hybrid MPI + threaded-BLAS placement (see Hybrid MPI +
Threaded-BLAS Placement above), switch to the `_gomp` variant and set
`OMP_NUM_THREADS` accordingly.

If you build PETSc with `--with-64-bit-blas-indices`, use the ILP64 variant
instead. Note that MUMPS and most other PETSc external packages do not accept
64-bit BLAS indices; prefer LP64 unless the package list has been verified.

**PETSc configure.** Point PETSc at the chosen library pair with
`--with-blaslapack-lib`. Do not set `--with-blaslapack-dir` at the same
time — the two flags are mutually exclusive.

LP64 sequential example (replace `${nvpl_ROOT}/lib` with the actual
library directory, e.g. `/usr/lib/aarch64-linux-gnu` for a
package-manager install):

```text
./configure \
  --with-scalar-type=complex \
  --with-blaslapack-lib="-L${nvpl_ROOT}/lib \
    -lnvpl_lapack_lp64_seq -lnvpl_lapack_core \
    -lnvpl_blas_lp64_seq   -lnvpl_blas_core"
```

Keep all other flags from the reference CPU/MUMPS stack unchanged
(`--download-mumps`, `--download-metis`, etc.). SLEPc inherits the BLAS/LAPACK
linkage from its parent PETSc build automatically; no separate NVPL step is
needed for SLEPc.

Because `install.sh` does not yet wire an NVPL stack, the build must be done
manually in an isolated directory and paired with a side-by-side virtual
environment (e.g. `.venv-nvpl`). Do not target `.venv`.

**complex128 (Z) verification — deferred to Grace aarch64 hardware.**
`linstabpy` builds with `--with-scalar-type=complex`, so all shifted solves
use complex-double (Z) BLAS/LAPACK symbols (`zgemm`, `zgetrf`, etc.). Once
the build completes on Grace hardware, confirm the Z symbols resolve to NVPL
rather than a fallback implementation:

```bash
# 1. ldd on the built libpetsc shared library — look for libnvpl_blas* and
#    libnvpl_lapack* in the output:
ldd /path/to/petsc-install/lib/libpetsc.so | grep nvpl

# 2. Check NumPy's BLAS/LAPACK wiring (if NumPy was also built against NVPL):
.venv-nvpl/bin/python -c "import numpy; numpy.show_config()"

# 3. Confirm a PETSc complex-scalar hello builds and links:
.venv-nvpl/bin/python -c \
  "import petsc4py; petsc4py.init(); \
   from petsc4py import PETSc; \
   v = PETSc.Vec().createSeq(4); v.set(1+2j); print('complex PETSc OK', v.norm())"
```

The `ldd` check is the primary gate: if `libnvpl_blas*` and
`libnvpl_lapack*` appear in the output, the Z symbols are routing through
NVPL. Run this on Grace before using the build for production stability or
resolvent computations.

**NumPy and SciPy.** For the Python stack, modern Meson-based NumPy/SciPy
builds accept NVPL via:

```bash
python -m build --wheel \
  -Csetup-args=-Dblas=nvpl-blas-lp64-omp \
  -Csetup-args=-Dlapack=nvpl-lapack-lp64-omp
```

This is secondary to the PETSc linkage; the PETSc Z-symbol check above is
the verification that matters for `linstabpy` correctness.

<!-- SOURCES -->
<!--
NVPL BLAS Developer Guide — library variants (lp64/ilp64, seq/gomp), naming
convention, libnvpl_blas_core:
  https://docs.nvidia.com/nvpl/latest/blas/programming_model/building_and_linking.html
  https://docs.nvidia.com/nvpl/latest/blas/packaging.html

NVPL LAPACK Developer Guide — library variants, libnvpl_lapack_core:
  https://docs.nvidia.com/nvpl/latest/lapack/programming_model/building_and_linking.html
  https://docs.nvidia.com/nvpl/latest/lapack/index.html

NVPL install / nvpl_ROOT / LD_LIBRARY_PATH:
  https://docs.nvidia.com/nvpl/latest/install.html

TACC BLAS/LAPACK guide — confirmed PETSc configure example with
  -lnvpl_lapack_lp64_seq -lnvpl_lapack_core -lnvpl_blas_lp64_seq -lnvpl_blas_core:
  https://docs.tacc.utexas.edu/tutorials/blas-lapack/

PETSc configure BLAS/LAPACK flags (--with-blaslapack-lib, --with-64-bit-blas-indices,
  mutual exclusivity with --with-blaslapack-dir):
  https://petsc.org/release/install/install/
  https://petsc.org/release/manual/blas-lapack/

NVPL NumPy/SciPy Meson build (-Dblas=nvpl-blas-lp64-omp):
  https://docs.nvidia.com/nvpl/latest/numerical-python.html
-->

## cuDSS GPU Backend

`factor_backend: cudss` factors the operator with NVIDIA cuDSS through
`nvmath-python` and applies it as a preconditioner inside the existing
right-preconditioned FGMRES path. It is not a PETSc factor package: configure
the iterative solver family, not the direct one.

```yaml
solver:
  resolvent:
    type: iterative
    ksp_type: fgmres
    pc_type: lu
    ksp_pc_side: right
    factor_backend: cudss
    # factor_precision: single   # opt-in, see below
```

Install with `pip install "linstabpy[petsc,cudss]"`: the `cudss` extra brings
the nvmath-python/cuDSS wheels (needs a CUDA 12 runtime and an NVIDIA GPU
supported by cuDSS), and the `petsc` extra provides the PETSc/SLEPc bindings
the iterative path runs on. A standard CPU build of PETSc/SLEPc is enough --
the GPU work happens inside cuDSS, so this backend needs no GPU-enabled PETSc
build and no STRUMPACK. The backend is
one-rank: run without MPI parallelism. The adjoint uses a second cuDSS
factorization of the conjugate transpose because cuDSS has no transposed or
adjoint solve; the existing dual-factor machinery builds and gates it.

Measured on a 450k-unknown SWBLI resolvent operator (complex double,
2.0-2.3e7 nonzeros, one RTX A6000): factorization 5.7-9.0 s against
41-55 s for STRUMPACK on the same card; FGMRES converges in 1-2 iterations;
solutions match a MUMPS reference to 1.2e-12. A refactorization on the kept
analysis costs a full numeric factorization (9.0-9.2 s); the analysis
(3.3-3.4 s with the threading layer) is what a frequency line saves.

Two environment overrides: `LINSTABPY_CUDSS_MT_LIB` points at the cuDSS
OpenMP threading layer (auto-discovered in the installed wheels by default;
`none` disables it) and `LINSTABPY_CUDSS_THREADS` sets the host threads for
the analysis phase (default: the process CPU affinity).

## Mixed-Precision Factorization Knob

`LINSTABPY_FACTOR_PRECISION` (environment) and the YAML `factor_precision`
field select the factor precision. The default is `double`, which is inert.

`factor_precision: single` works with the `cudss` backend only. The
factorization runs in complex64 — measured 6-9x faster and half the device
memory — while FGMRES stays in complex double and restores the solution to
1e-12. The backend force-enables the two measures required for convergence:
MC64-style matching (`MAX_DIAG_PRODUCT`) and double-precision equilibration
before the cast. The validated scope is the production right-hand-side
family (forcing-masked vectors); the adjoint self-test for this
configuration draws its probes from that family and fails loudly when no
family is attached.

For PETSc factor backends, `single` is refused. The historical
`superlu_dist` route drove it through the PETSc `-pc_precision` option,
which PETSc 3.24 ignores — the factor would silently be computed in double
while provenance recorded single. The guard in
`linstabpy/linalg/backends.py` explains this and names the cudss route.
`double` remains the reference and default path everywhere.

## MUMPS BLR Low-Rank Compression Knob

MUMPS block-low-rank (BLR) compression can be requested only for the MUMPS factor
backend. It is off by default, so the default MUMPS options path is unchanged.
Set `LINSTABPY_MUMPS_BLR=1` or the matching YAML field `mumps_blr: true` in the
relevant solver subsection to enable it. The BLR dropping tolerance is controlled
by `LINSTABPY_MUMPS_BLR_TOL` or `mumps_blr_tol`; the default tolerance is
`1e-4`. Environment variables take precedence over YAML, which takes precedence
over the defaults.

Today only the environment variables are enforced at factorization time, and
they apply to every MUMPS LU `linstabpy` sets up in the process. The YAML
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
its default automatic ordering selection (`ICNTL(7)=7`). Set the ordering with
`mumps_ordering: metis` in YAML or `LINSTABPY_MUMPS_ORDERING=metis` in the
environment. The old `mumps_icntl_7` boolean is gone.

Environment `LINSTABPY_MUMPS_ORDERING` wins over YAML `mumps_ordering`. When
both are unset, `linstabpy` expresses no opinion. User-supplied PETSc CLI
options are still process-global and should not be mixed with these
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
| `cudss` | Validated GPU sparse-direct backend on the iterative path. | Not a PETSc factor package: the cuDSS LU runs as the preconditioner inside the right-preconditioned FGMRES path (`type: iterative`, `factor_backend: cudss`). Single rank, one GPU. See the "cuDSS GPU Backend" section above. |

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

## Harmonic Resolvent (resolvent4py)

Harmonic resolvent analysis via `resolvent4py` (MIT license, SoftwareX 2025) is available as an optional add-on in `linstabpy 1.2+`. The path requires a complex-valued PETSc build (same as the reference CPU stack) and MUMPS for the factorization, plus the `resolvent4py` package.

### Dependency and Installation

Install with `pip install "linstabpy[harmonic]"`. This brings resolvent4py 1.0.1 (the wheel API the wrapper targets). Note that resolvent4py declares only numpy/scipy/matplotlib as dependencies — it does not install petsc4py, so the complex PETSc/SLEPc stack must already be present. The same `.venv` or `-gpu-*` environment that provides complex PETSc/SLEPc is sufficient; no separate PETSc build is needed.

### Operator Size and Memory Scaling

The harmonic-resolvent generator `T` is a block-Toeplitz operator stacking `2*N_p+1` copies of the base-flow Jacobian, where `N_p = N_h + 3` is the padding (N_h = highest harmonic to keep). The total DOF is:

```
N_full = (2*N_p + 1) * n,  where n = base-flow DOF on one harmonic
```

For example, a base-flow operator with n = 1000 DOF and N_h = 10 gives:
- N_p = 13
- N_full = 27 * 1000 = 27,000
- Memory for the LU factor scales as O(N_full^2) with a large constant (dense triangular factors in a sparse-direct solve)

A 901k-DOF SWBLI operator with N_h would require stacking 25–27 copies, reaching 22–24M DOF in the full operator. A single-node MUMPS factorization at that scale risks out-of-core spillage or failure. The tag gates (resolvent4py toy model with 3 DOF and steady-limit with ~200 DOF) are scaled to stay within single-node scope by design.

### Configuration

The harmonic path is a Python API. The YAML case driver does not run it, and
`validate_config` rejects `periodic_baseflow:` or `harmonic_resolvent:` keys in
a case file. The call sequence is:

```python
from linstabpy.baseflow.periodic import load_periodic_baseflow
from linstabpy.operators.harmonic_blocks import assemble_harmonic_blocks, write_resolvent4py_coo
from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains

pbf = load_periodic_baseflow("periodic_qhat.npz", mesh_shape=(ny, nx))
blocks = assemble_harmonic_blocks(mesh, pbf, N_h=5, N_t=32, kx=beta, bc_recipe=recipe)
coo = write_resolvent4py_coo(blocks, "coo/")
gains, info = harmonic_resolvent_gains(coo, n, perts_freqs=perts_freqs, n_modes=10, kappa=1e-3)
```

`perts_freqs` holds the `2*N_p+1` perturbation frequencies, with `N_p = N_h + 3`.
`examples/harmonic/padovan_ode/padovan_toy.py` runs the whole sequence end to end.

The periodic base-flow loader validates the input `.npz` for canonical freestream normalization (ρ_∞ = u_∞ = T_∞ = 1) and rejects mismatched mesh shapes.

### Known Limitations

1. Harmonics at non-zero spanwise wavenumber (β ≠ 0) require two-sided storage (k = -N_h..N_h). The conjugation assumption Â_{-k} = conj(Â_k) fails when β ≠ 0 because the operator is genuinely complex through i·β derivatives. This is handled automatically by the FFT path.

2. Real SWBLI-scale cases are out of single-node reach at present. The 1.2.0 tag gate runs only the resolvent4py toy model (3 DOF) and a steady-limit validation on small meshes.

3. Time-stepping alternatives (Farghadan RSVD-Δt) are not yet implemented; the assembled `T` matrix must fit in memory.

