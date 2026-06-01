# Compute Backends

`linstabpy` uses PETSc LU factorization for sparse-direct solver paths. The
default factor backend is `mumps`.

Precedence for the factor backend is:

1. JSONC solver subsection `factor_backend`;
2. existing per-solver environment override, such as
   `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE`;
3. `LINSTABPY_FACTOR_BACKEND`;
4. the call-site default.

The temporal and spatial EVP direct paths intentionally preserve the historical
asymmetry: `st_solver = "mumps"` forces `mumps`, while `st_solver = "direct"`
or `"lu"` leaves PETSc's native LU backend unless a factor backend is explicitly
configured. The linear-solve direct path keeps forcing `mumps` for
`type = "mumps"`, `"direct"`, and `"lu"`.

Audit note for the original hardcoded sites:

- `src/linstabpy/mesh/filters.py`: the alpha-filter solve is a sequential
  mesh-filter LU used when filters are enabled, including default-numerics
  validation paths. It remains pinned to `mumps` and does not follow the global
  factor-backend knob.
- `src/linstabpy/solvers/eigenvalue.py`: temporal ST direct path.
- `src/linstabpy/solvers/spatial_eig.py`: spatial ST direct path.
- `src/linstabpy/solvers/linsolve.py`: reusable KSP/PC direct solve path used
  by resolvent solves.

Requested backends are checked with a tiny PETSc LU setup probe. An unavailable
backend fails loudly instead of falling back silently.

## Resolvent GPU economics

Recommendation: **do not make the GPU sparse-direct path the next production
priority for resolvent sweeps.** Keep CPU/MUMPS as the validated baseline, and
only attempt a GPU direct backend after a CPU run proves that a single frequency
is dominated by many repeated right-hand-side solves.

The current resolvent code factors one shifted operator per analyzed frequency:
`L_sigma = L0 - i*omega*I`. The factorization is reused for the forward and
adjoint actions inside that one frequency, including residual checks, but a
different `omega` changes the matrix values and requires a fresh numerical
factorization. A frequency sweep therefore pays roughly

```text
N_freq * T_factor + N_freq * N_rhs_per_freq * T_solve
```

not

```text
T_factor + N_freq * N_rhs_per_freq * T_solve
```

The local 50-point Case 2 resolvent configuration records the scale of the
current direct path as about 47 h serial and about 15 GiB peak RSS per point for
an fd6 stride-4 sweep
(`examples/case_studies/case2_lui/configs/resolvent/resolvent_grid_50pt_fd6_stride4.jsonc`).
That means backend economics are controlled first by repeated factorizations,
then by per-frequency SLEPc iterations and adjoint/forward solves.

The best published STRUMPACK GPU result relevant to direct factorization reports
about a 6.5x factorization speedup for an exact GPU solver on a large wave
problem, while also noting that multi-GPU triangular solves did not improve over
CPU solves in that experiment; efficient GPU triangular solves were observed only
when a single-GPU factorization fit in device memory
([Claus et al. 2025](https://journals.sagepub.com/doi/10.1177/10943420241288567)).
For a sweep, that speedup applies separately to each frequency. It does not
turn the sweep into a one-factor-many-solves workload.

Go/no-go:

- **Go** for a GPU direct prototype when the target run has a small number of
  frequencies, many SLEPc iterations or explicit right-hand sides per frequency,
  and factors that fit on the intended GPU path. A practical threshold is that
  the measured per-frequency solve work is at least comparable to the
  factorization time, or that CPU MUMPS cannot fit memory while a GPU-capable
  compressed or distributed factorization plausibly can.
- **No-go** as the default next production path for broad resolvent maps. A
  50-frequency sweep still pays 50 numerical factorizations; without measured
  local evidence that the GPU backend accelerates complex128 factorization for
  these LNSE matrices and preserves residual quality, the near-term return is
  too uncertain.

## Backend maturity

All GPU backend statements below are desk-research conclusions only. None of
these paths is built or validated in the default `linstabpy` environment.

| Backend | Complex128 GPU verdict | aarch64 + CUDA verdict | Recommendation |
|---|---|---|---|
| `mumps` | CPU baseline. No GPU sparse-direct path is selected by this seam. | Not applicable. | Keep as the validated default. |
| `strumpack` | **Yes, plausible first prototype.** STRUMPACK documents GPU acceleration in its sparse direct solver through CUDA/cuBLAS/cuSOLVER and optional SLATE; PETSc exposes `-mat_strumpack_gpu`; the 2025 STRUMPACK GPU paper reports exact-factorization GPU speedups and PETSc interfaces. Treat complex128 LNSE performance as unvalidated until measured. Sources: [STRUMPACK GPU support](https://portal.nersc.gov/project/sparse/strumpack/v6.3.1/GPU_Support.html), [PETSc MATSOLVERSTRUMPACK](https://petsc.org/release/manualpages/Mat/MATSOLVERSTRUMPACK/), [Claus et al. 2025](https://journals.sagepub.com/doi/10.1177/10943420241288567). | **Unknown from the solver docs.** CUDA itself has aarch64 distributions, but the STRUMPACK/PETSc documentation checked here does not state a supported aarch64+CUDA build recipe. | First backend to try only after a controlled PETSc rebuild experiment. |
| `superlu_dist` | **Yes, exposed by PETSc, but second choice.** PETSc documents `-mat_superlu_dist_gpuoffload` for SuperLU_DIST 8.0.0 or later and automatic GPU use when PETSc is configured with CUDA. SuperLU also documents GPU support in SuperLU_DIST and complex-code maintenance in release notes. Sources: [PETSc MATSOLVERSUPERLU_DIST](https://petsc.gitlab.io/petsc/main/manualpages/Mat/MATSOLVERSUPERLU_DIST/), [SuperLU FAQ](https://portal.nersc.gov/project/sparse/superlu/faq.html), [SuperLU change log](https://portal.nersc.gov/project/sparse/superlu/changes.html). | **Unknown from the upstream docs checked here.** The public package ecosystem has aarch64 SuperLU_DIST builds, but that is not evidence for a supported CUDA offload build on aarch64. | Try after STRUMPACK unless a target platform already ships a known-good SuperLU_DIST GPU PETSc build. |
| `cudss` | NVIDIA cuDSS is a GPU direct sparse solver, but `linstabpy` has no PETSc `cuDSS` factor-solver interface to select today. Source: [NVIDIA cuDSS](https://developer.nvidia.com/cudss). | NVIDIA advertises Grace Hopper support, but without a released PETSc solver interface this is not actionable through the current seam. | Do not wire yet; revisit only if PETSc exposes a released `MATSOLVERCUDSS`-style interface. |

Backend go/no-go:

- **Go first:** STRUMPACK, as an experimental rebuild/prototype, because it has
  PETSc exposure, documented sparse-direct GPU acceleration, and recent
  published GPU sparse-direct results.
- **Hold:** SuperLU_DIST until there is a known target build or a reason to
  prefer its static-pivoting path for a specific matrix family.
- **No-go now:** cuDSS through `linstabpy`, because the generic seam can name the
  backend but PETSc does not provide a released cuDSS factor-solver interface in
  the checked documentation.

## Version floors

| Backend | Minimum PETSc/SLEPc and external package floor | Status in default install |
|---|---|---|
| `mumps` | PETSc/SLEPc complex build with MUMPS enabled. CPU-only default. | Validated/default. |
| `strumpack` | PETSc exposing `MATSOLVERSTRUMPACK`, with PETSc configured for STRUMPACK and CUDA-capable STRUMPACK for GPU numerical factorization. PETSc documents `-mat_strumpack_gpu` and recommends one MPI process per GPU. | Experimental; not built by default. |
| `superlu_dist` | PETSc exposing `MATSOLVERSUPERLU_DIST` plus SuperLU_DIST >= 8.0.0 for `-mat_superlu_dist_gpuoffload`. | Experimental; not built by default. |
| `cudss` | No released PETSc factor-solver interface confirmed in the checked PETSc 3.25 documentation. | Not available through the seam. |
