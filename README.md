![linstabpy banner](docs/assets/readme-banner-v3.jpg)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PETSc](https://img.shields.io/badge/solvers-PETSc%20%7C%20SLEPc%20%7C%20MUMPS-blue.svg)](https://petsc.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

`linstabpy` performs sparse linear stability and resolvent analysis of
compressible viscous flows. It builds the linearized compressible Navier–Stokes
operators for two-dimensional structured base flows and solves the eigenvalue and
resolvent problems through PETSc/SLEPc with sparse direct factorization (MUMPS).

It began as a Python port of the MATLAB `LinStab2D` code (Martini and Schmidt,
2024) and has grown into a standalone framework: it adopts the operator
abstractions of `resolvent4py` (Padovan et al., 2025), extends the discretization
to fd2/fd4/fd6/fd8 summation-by-parts operators, and verifies the assembled
operator against independent derivations rather than against MATLAB output alone.

Memory was the original motivation. Stability and resolvent analysis of 2D base
flows produces large linear systems, and MATLAB's dense `backslash` path scales
as `O(n^2)` in storage. A 150k-DOF operator needs about 180 GB of RAM that way,
which puts most LES/DNS-derived base flows out of reach on an ordinary
workstation. Keeping the operators sparse and sending the linear algebra through
PETSc/SLEPc and MUMPS brings those same problems back inside a workstation's
budget.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Numerical setup](#numerical-setup)
- [Nondimensionalization](#nondimensionalization)
- [Boundary conditions](#boundary-conditions)
- [Parity with LinStab2D](#parity-with-linstab2d)
- [Case studies](#case-studies)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [References](#references)
- [License and citation](#license-and-citation)

## Overview

The package targets two-dimensional structured base flows and builds operators
for the linearized compressible Navier–Stokes equations. The public project name
is `LinStabPy`; the repository, distribution, and import package name is
`linstabpy`.

It currently supports:

- **Resolvent analysis:** leading gain, forcing modes, and response modes
- **Temporal stability:** direct and adjoint eigenvalue problems
- **Spatial stability:** streamwise-wavenumber eigenvalue problems
- **Coordinate systems:** Cartesian, cylindrical/axisymmetric, and curvilinear structured meshes
- **Boundary treatment:** Dirichlet rows, Neumann rows, sponge damping fields, and masking for structured grids with excluded points
- **Energy norms:** Chu compressible energy norm with quadrature weighting

The compressible formulation follows Martini and Schmidt (2024), and the
operator-handling design draws on `resolvent4py` (Padovan et al., 2025). Where the
original code was validated mainly against its own MATLAB output, LinStabPy adds a
layer of provenance-independent checks: a symbolic re-derivation of the
coefficients, a complex-step Jacobian, and direct verification of the curvilinear
transform, global assembly, boundary rows, and energy norm.

## Installation

### Prerequisites

Use the repository scripts first. They check compilers, MPI, Python bindings,
PETSc/SLEPc, and the runtime stack that the examples will use.

Required system tools:

- POSIX shell with `bash`
- C, C++, and Fortran compilers
- `make`, `cmake`, `curl`, and `pkg-config`
- one MPI stack with development headers (`mpicc`, `mpicxx`, `mpifort`, `mpiexec`)
- a Python environment managed by the repository installer
- HDF5 development files (optional, only for MATLAB v7.3 / HDF5 tooling)

Do not mix MPI stacks in one environment. PETSc, SLEPc, `mpi4py`, and the runtime
`mpiexec` must agree on the same MPI ABI.

### Linux (recommended path)

Clone the repository and run the native checks:

```bash
git clone https://github.com/openfluids/linstabpy.git
cd linstabpy

bash diagnose.sh
bash install.sh
bash install.sh --check
bash benchmark.sh
```

The reference MATLAB base flows (`M09jet.mat`, `BL_BF.mat`, `Step_BF.mat`,
`Paraboloid_BF.mat`) used by the Martini example loaders are not bundled. See
[Fetching reference base flows](#fetching-reference-base-flows) for how to point
linstabpy at a local clone of LinStab2D.

`diagnose.sh` is read-only. It checks compilers, MPI launchers, development
headers, dual-MPI conflicts, and whether `mpiexec -n 1 /bin/true` works.

`install.sh` creates `.venv`, builds or installs the Python/PETSc/SLEPc stack,
repairs `mpi4py` when the wheel ABI does not match the selected MPI stack, and
installs this repository in editable mode.

Use `--force` for a clean reinstall. It removes the existing `.venv` first, then
rebuilds the full Python, PETSc/SLEPc, `mpi4py`, `petsc4py`, and `slepc4py`
environment:

```bash
bash install.sh --force
```

Use `--latest` only when you deliberately want the newest compatible package
releases instead of the pinned PETSc/SLEPc family used by the default installer:

```bash
bash install.sh --force --latest
```

Python 3.13 is the default tested interpreter class. Python 3.14 can be tried
when your `uv` installation can resolve it and all PETSc/SLEPc bindings have
compatible releases:

```bash
LINSTABPY_PYTHON_VERSION=3.14 bash install.sh --force --latest
```

Treat Python 3.14 as a freshness test until `install.sh --check`, the pytest
suite, and the examples pass on your machine. For reproducible comparisons,
record the Python version printed by `install.sh --check`.

`install.sh --check` verifies that `petsc4py` and `slepc4py` import, that PETSc
uses complex scalars, and that a small SLEPc non-Hermitian Krylov-Schur solve
works from Python.

`benchmark.sh` runs a small smoke benchmark and writes benchmark artifacts under
`benchmark_runs/<run-id>/`.

By default, the installer does not call `sudo` or a package manager. If required
system tools are missing, it prints the commands and stops. On a normal
workstation, opt in to package-manager installation explicitly:

```bash
LINSTABPY_AUTO_INSTALL_SYS_DEPS=1 bash install.sh
```

On a provisioned workstation or cluster node, skip the system dependency check:

```bash
LINSTABPY_SKIP_SYS_DEPS=1 bash install.sh
```

### macOS

macOS uses the same `diagnose.sh` → `install.sh` → `install.sh --check` →
`benchmark.sh` flow as Linux. The only difference is the BLAS/ScaLAPACK setup
below.

For the macOS PETSc/SLEPc path, install Homebrew OpenBLAS and ScaLAPACK before
building when possible:

```bash
brew install openblas scalapack
```

The reason is a macOS runtime-library ordering issue in Python. PETSc's
downloaded ScaLAPACK can resolve LAPACK symbols to Apple Accelerate when loaded
inside Python. The failing path observed in this repository was Python
`slepc4py` `EPS_NHEP` with Krylov-Schur / Arnoldi, aborting inside Accelerate
`ZLADIV`. Native C SLEPc NHEP worked, and Python HEP/LAPACK paths worked, which
pointed to dynamic-library symbol interposition rather than the examples.

The current installer handles this by preferring Homebrew OpenBLAS/ScaLAPACK at
PETSc configure time and by wrapping `.venv/bin/python` so Homebrew OpenBLAS is
loaded before Python imports NumPy, PETSc, or SLEPc. `install.sh --check` runs a
tiny Python `EPS_NHEP` Krylov-Schur probe, so this crash is detected before
examples are run.

On macOS, do not use `uv run` to run LinStabPy examples after installation. It
can bypass the patched `.venv` and try to rebuild `petsc4py`/`slepc4py` from
source with the wrong runtime layout. Use `.venv/bin/python` directly:

```bash
.venv/bin/python -m examples.example_1_blasius_smoke
```

### Clusters and Intel MPI

On clusters, load the compiler/MPI modules before running the installer. The
repository scripts should see the same `mpicc`, `mpicxx`, `mpifort`, and
`mpiexec` that will be used at runtime.

Generic module workflow:

```bash
module load <compiler-module> <mpi-module>
bash diagnose.sh
bash install.sh
bash install.sh --check
```

Intel MPI / oneAPI workflow:

```bash
source /opt/intel/oneapi/setvars.sh --force
export LINSTABPY_MPI_STACK=intelmpi
export I_MPI_CC=gcc
export I_MPI_CXX=g++
export I_MPI_F90=gfortran
export I_MPI_FC=gfortran

bash diagnose.sh
bash install.sh
bash install.sh --check
```

Use Intel MPI only when `diagnose.sh` shows that the active MPI compiler wrappers
and `mpiexec` come from the Intel MPI environment and the MPI compile probe
passes. Do not use an Intel MPI launcher with Open MPI or MPICH Python bindings.

### Environment variables

| Variable | Default | Use |
|---|---:|---|
| `LINSTABPY_WITH_H5PY` | `0` | Set to `1` to install `h5py` for MATLAB v7.3 / HDF5 parity tooling. Leave unset for the normal solver install. |
| `LINSTABPY_MPI_STACK` | auto | Set to `intelmpi` when an Intel MPI environment is already loaded and should be used. Otherwise the scripts infer the active stack. |
| `LINSTABPY_AUTO_INSTALL_SYS_DEPS` | `0` | Set to `1` to allow `install.sh` to call the system package manager for missing Linux dependencies. |
| `LINSTABPY_PETSC_MPIEXEC` | auto | Path to the `mpiexec` PETSc/SLEPc configure should use. Set this when the default launcher hangs or is not the runtime launcher you want. |
| `LINSTABPY_MPI4PY_NO_BINARY` | `0` | Set to `1` to force `mpi4py` to build from source against the active MPI stack. |
| `LINSTABPY_VERIFY_MPIEXEC` | auto | Path to the `mpiexec` used by `install.sh --check` for MPI import/probe checks. |
| `LINSTABPY_SKIP_SYS_DEPS` | `0` | Set to `1` to skip Linux system dependency detection and package-manager advice. |

### Activation and shell aliases

You can run everything through `.venv/bin/python` without activating the
environment:

```bash
.venv/bin/python -m pytest tests/
```

For interactive work, activation is still convenient:

```bash
source .venv/bin/activate
```

If you use the project often, add a small alias or shell function to your
`~/.bashrc` / `~/.zshrc` so you do not have to remember the activation path:

```bash
alias linstabpy-activate='cd /path/to/linstabpy && source .venv/bin/activate'
```

Keep the path local to your machine. The alias is optional and should not be used
in scripts; scripts should call `.venv/bin/python` directly so logs show which
interpreter ran.

### MPI stack verification

The installer chooses one active MPI stack from the shell environment. The same
stack must own all of these pieces:

- `mpicc`, `mpicxx`, and `mpifort`
- the `mpiexec` used by PETSc/SLEPc configure
- `mpi4py`
- `petsc4py` and `slepc4py`
- the `mpiexec` used for runtime examples and tests

Read the install log rather than guessing. A healthy install prints the MPI
launcher used for PETSc/SLEPc configure, the launcher used for verification, the
resolved MPI library directory, and the result of single-rank and multi-rank
checks. If those paths point to different MPI families, rebuild from a clean
environment:

```bash
rm -rf .venv
bash diagnose.sh
bash install.sh --force
bash install.sh --check
```

For clusters with Intel MPI, source oneAPI or the site module first and make that
choice explicit:

```bash
source /opt/intel/oneapi/setvars.sh --force
export LINSTABPY_MPI_STACK=intelmpi
bash diagnose.sh
bash install.sh --force
bash install.sh --check
```

Do not repair a mixed MPI install by editing `LD_LIBRARY_PATH` until it happens
to import. Reinstall with the intended MPI stack visible from the start.

### Known-good reference stacks

A healthy Linux install should satisfy this sequence:

```bash
bash diagnose.sh
bash install.sh --check
bash run_all_examples.sh --mpi -n 1 -m examples.example_1_blasius_smoke
bash benchmark.sh
```

One known-good Linux profile used during hardening was:

| Component | Reference value |
|---|---|
| OS family | Ubuntu 24.04 class |
| MPI | Open MPI 4.1.6 class |
| Python | 3.13 class |
| PETSc/SLEPc scalar type | complex |
| Sparse direct solver | MUMPS through PETSc |
| Thread policy | one BLAS/OpenMP thread per MPI rank |

Use this as a comparison shape, not a hard pin. Export a redacted stack manifest
from a working machine with:

```bash
.venv/bin/python scripts/tools/export_known_good_stack.py \
  --output known_good/<machine-label>.json \
  --redact-local-details
```

### MPI runtime rules

Use one MPI stack per install and one thread per rank unless you are measuring a
different layout on purpose.

For manual MPI runs:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

These defaults prevent every MPI rank from also starting a full BLAS/OpenMP
thread pool. They are the right starting point for sparse direct solves and
postcheck-heavy runs. Increase them only for a controlled benchmark, and record
the changed values with the run log because they affect wall time, memory
pressure, and reproducibility.

Prefer the canonical example runner for MPI examples:

```bash
bash run_all_examples.sh --mpi -m examples.example_1_blasius_smoke
bash run_all_examples.sh --mpi -n 10 -m examples.example_3_streaky_bl
bash run_all_examples.sh --mpi -n 1 -m examples.example_1_blasius_smoke -- -eps_monitor
```

If `-n/--ranks` is omitted in `--mpi` mode, the runner starts from half of the
detected physical cores. Override the launcher with `LINSTABPY_MPIEXEC` and the
default rank count with `LINSTABPY_MPI_RANKS` when the target machine needs an
explicit MPI runtime. For sparse direct factorization, more ranks are not
automatically faster; compare rank counts on the target machine.

### Fetching reference base flows

The Martini example loaders (`example_1_blasius`, `example_2_jet`,
`example_4_cavity`, `example_5_paraboloid`) consume four `.mat` base flows from
the upstream LinStab2D project: `M09jet.mat`, `BL_BF.mat`, `Step_BF.mat`,
`Paraboloid_BF.mat`. linstabpy does not bundle these files. There are two ways to
make them available.

Clone LinStab2D once and point linstabpy at it with an env var:

```bash
git clone https://github.com/LinStab2D/LinStab2D.git ~/linstab2d-reference
export LINSTABPY_BASEFLOW_DATA_DIR=$HOME/linstab2d-reference
```

Or copy the four `.mat` files into the gitignored data dir:

```bash
mkdir -p src/linstabpy/data/baseflows/BaseFlows
cp ~/linstab2d-reference/BaseFlows/*.mat src/linstabpy/data/baseflows/BaseFlows/
```

If neither is in place, the Martini examples raise a `FileNotFoundError` with the
exact instruction to set `LINSTABPY_BASEFLOW_DATA_DIR`. All non-Martini examples
(Case 1 Song, Case 2) ship their own base flows under `examples/case_studies/`
and are unaffected.

## Quick start

```bash
git clone https://github.com/openfluids/linstabpy.git
cd linstabpy

bash diagnose.sh
bash install.sh
bash install.sh --check
bash benchmark.sh
```

This repository has no required submodules. Reference Martini base flows are
handled separately in [Fetching reference base flows](#fetching-reference-base-flows).

Run the smallest Blasius example:

```bash
.venv/bin/python -m examples.example_1_blasius_smoke
```

Run a full example:

```bash
.venv/bin/python -m examples.example_1_blasius
```

Do not use `uv run` on macOS for PETSc/SLEPc examples. Use the interpreter
installed and patched by `install.sh`:

```bash
.venv/bin/python -m examples.example_1_blasius_smoke
```

Collect the test suite:

```bash
.venv/bin/python -m pytest --collect-only tests
```

Run the suite with:

```bash
.venv/bin/python -m pytest tests/
```

#### MATLAB-parity tests are opt-in

This repository does **not** ship MATLAB reference data — neither the raw `.mat`
exports nor the derived `.npz` bundles. The tests that compare against that data
are therefore **deselected by default** via the `matlab_parity` marker, and run
only when you ask for them:

```bash
LINSTABPY_MATLAB_PARITY=1 .venv/bin/python -m pytest -m matlab_parity
```

Two consequences are deliberate:

- **A default run reports no skips it cannot justify.** These tests are
  *deselected*, not skipped, so they never pad the skip count. Every skip in a
  default run is an absent optional backend (STRUMPACK, CUDA, `superlu_dist`) —
  a true statement about the machine you are on.
- **Requesting them means they must run.** With `LINSTABPY_MATLAB_PARITY=1`, a
  missing reference is a **failure**, not a skip. A requested comparison that
  quietly does not execute is indistinguishable from one that passed.

Supply the reference data through `LINSTABPY_BASEFLOW_DATA_DIR`, the
`martini_examples/` export directory, `LINSTABPY_TEST_REFERENCE_SPECTRUM`, and the
`scripts/validation/martini2024/fixtures/` bundles. The failure message names the
exact paths it wanted.

### Canonical example runner

Use `run_all_examples.sh` as the single example runner for serial acceptance,
cache cleanup, parity checks, and MPI launches. Direct module execution with
`.venv/bin/python -m ...` is still useful for focused debugging, but the runner
is the documented path for full example validation and MPI-safe launches. For a
figure-by-figure walkthrough of the Martini examples, see
`docs/tutorial/tutorial.pdf`, Section 5. That section also explains how to read
the generated baseflow, gain/spectrum, and support-component figures, and records
the plotting improvements proposed for the next figure pass.

| Task | Command |
|---|---|
| List available examples | `bash run_all_examples.sh --list` |
| Run all examples and Martini parity | `bash run_all_examples.sh` |
| Recompute selected examples from cached PNGs/NPZs | `bash run_all_examples.sh --fresh --example 1 --skip-parity` |
| Remove selected generated NPZ/PNG outputs first | `bash run_all_examples.sh --clean --example 1 --skip-parity` |
| Run one example under MPI | `bash run_all_examples.sh --mpi -n 10 -m examples.example_3_streaky_bl` |
| Pass PETSc/SLEPc options to one example | `bash run_all_examples.sh --mpi -n 1 -m examples.example_1_blasius_smoke -- -eps_monitor` |

`--mpi` folds the old one-module MPI wrapper behavior into the same script: the
runner probes a working MPI launcher, defaults to half of the detected physical
cores when `-n/--ranks` is omitted, and caps OpenMP/BLAS thread counts to one
thread per rank unless the caller already set them. Override the runtime with
`LINSTABPY_MPIEXEC`, `LINSTABPY_MPI_RANKS`, or `LINSTABPY_PYTHON` when a machine
needs explicit paths.

### Why JSONC configs?

JSONC is JSON with comments, and this repo uses it deliberately even though plain
JSON is more common. Each example is a numerical case definition: Reynolds number,
Mach number, grid size, boundary conditions, sponge settings, forcing masks,
solver targets, and PETSc/MPI choices should stay visible and diffable, with an
explanation sitting next to each value.

Plain JSON would remove those explanations. Long command-line flags would hide
too much state in shell history or runner scripts. Python config files would mix
data with executable code. JSONC keeps the inputs close to strict JSON while
still letting the case file document why each parameter exists.

## Repository layout

For a maintainer map of where numerical policy belongs, see
`docs/architecture.md`.

```text
.
|-- src/linstabpy/                 Importable Python package
|   |-- baseflow/                  Base-flow loaders and viscosity helpers
|   |-- boundary_conditions/       Dirichlet/Neumann row replacement and recipes
|   |-- config/                    Runtime validation metadata
|   |-- data/baseflows/            Optional local MATLAB reference assets
|   |-- mesh/                      SBP finite differences, grids, filters, deformation
|   |-- operators/                 LNSE coefficients, assembly, spatial operators, Chu norm
|   |-- solvers/                   PETSc/SLEPc resolvent and eigenvalue solvers
|   |-- utils/                     JSONC, plotting, PETSc helpers, acceptance checks
|   `-- results_runtime.py         Package runtime used by case study drivers
|-- examples/                      Runnable examples; generated .npz/.png ignored
|   |-- case_studies/              Larger validation and SWBLI case workflows
|   |   |-- case1_song/            Song/Robinet-style case with MATLAB parity assets
|   |   `-- case2_lui/             Larger curvilinear SWBLI configs and docs
|   `-- example_*.py               Martini-style compact examples
|-- scripts/                       Repository guardrails and validation utilities
|   |-- ci/                        Thin install/diagnose/benchmark harness
|   `-- validation/                Reproducibility harnesses and compact fixtures
|       |-- martini2024/           Martini parity fixtures, tools, and reports
|       `-- operator_fixtures/     Compact operator-parity fixtures and schema
|-- docs/                          Architecture, install, audit, and tutorial docs
|-- tests/                         Pytest suite and shared path helpers
|-- install.sh                     Main installer
|-- diagnose.sh                    Read-only environment preflight
|-- benchmark.sh                   Smoke benchmark
`-- run_all_examples.sh            Serial, parity, and MPI example runner
```

Keep reusable code under `src/linstabpy/`, runnable examples under `examples/`,
case-study workflows under `examples/case_studies/`, and validation harnesses
under `scripts/validation/`. Generated example outputs, campaign `output/`
directories, runtime caches, and build metadata are ignored rather than
represented as tracked layout.

Optional reference MATLAB code and `.mat` base-flow files may be placed under
`src/linstabpy/data/baseflows`, or supplied through `LINSTABPY_BASEFLOW_DATA_DIR`.
Treat that directory as local reference input, not production package code.

### Data and file-size policy

Tracked GitHub-facing files must stay at or below 50 MiB. Large generated
outputs, raw MATLAB exports, and machine-local benchmark products are kept out of
Git and either regenerated or stored externally. The guardrail is:

```bash
python scripts/check_tracked_file_size.py
```

The same guard is wired into the local `pre-commit` configuration and the GitHub
guardrails workflow. Ignored local outputs above the report threshold are listed
separately so they can be cleaned or archived without blocking normal source
commits.

Compact validation fixtures belong under `scripts/validation/operator_fixtures/`
or `scripts/validation/martini2024/fixtures/`. Raw MATLAB exports such as
`operator_convention.mat` and `coefficients_raw.mat` are intentionally not
tracked when they exceed the 50 MiB policy.

## Numerical setup

### Temporal stability

The current temporal stability path uses:

- SLEPc `EPS`
- `KRYLOVSCHUR` as the outer eigensolver
- shift-invert spectral transformation, `ST=SINVERT`
- direct LU factorization with MUMPS for shifted linear systems in the default production path

The temporal eigenproblem is formed on `i * L0`, so reported temporal eigenvalues
are `omega = eig(i * L0)`.

### Residual bookkeeping

The code records several residual-like quantities:

- SLEPc EPS residuals from `EPS.computeError(...)`
- the configured EPS convergence test: `abs`, `rel`, or `norm`
- the selected reported residual type: `absolute`, `relative`, or `backward`
- an explicit postcheck residual computed outside SLEPc on returned modes
- boundary-condition leakage diagnostics

These quantities are related but not interchangeable. `EPSSetTolerances()` sets
the tolerance and maximum iteration count used by the EPS convergence test.
`EPSSetConvergenceTest()` selects how the EPS convergence test is measured.
`EPSComputeError()` reports an error estimate for a returned eigenpair, but that
estimate is not always the same quantity used internally during the solve.

For shift-invert runs, `true_residual=true` is useful because cheap transformed
residual estimates can be optimistic for the original problem.

### PETSc KSP tolerances

`KSPSetTolerances()` controls a linear Krylov solve: relative tolerance, absolute
tolerance, divergence tolerance, and maximum iteration count. It is not an
eigenpair tolerance.

In the default direct path, shifted systems use:

```text
KSPPREONLY + PCLU + MUMPS
```

In that configuration, iterative `KSP` tolerances are not the main accuracy
knobs. The dominant knobs are EPS settings, Krylov subspace size `ncv`, target
placement, and the conditioning of the shifted operator.

When an iterative linear-solve path is selected, `ksp_rtol`, `ksp_atol`,
`ksp_max_it`, `pc_type`, subdomain settings, and `ksp_error_if_not_converged`
become meaningful.

### Compute backends

The validated sparse-direct default is PETSc LU with MUMPS. Backend selection is
centralized in the factor-backend seam and applies only to PETSc LU paths, not to
iterative Krylov-only solves.

Selection precedence is:

1. JSONC solver subsection `factor_backend`
2. per-solver environment override, such as `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE` or `LINSTABPY_RESOLVENT_PC_FACTOR_SOLVER_TYPE`
3. `LINSTABPY_FACTOR_BACKEND`
4. the call-site default

Examples:

```bash
LINSTABPY_FACTOR_BACKEND=mumps .venv/bin/python -m examples.example_1_blasius_smoke
LINSTABPY_FACTOR_BACKEND=strumpack .venv/bin/python -m examples.example_1_blasius_smoke
```

Unavailable backends fail during PETSc LU setup with a direct error message
instead of silently falling back to MUMPS or PETSc native LU.

| Backend | Status | Notes |
|---|---|---|
| `mumps` | Validated default | CPU sparse-direct backend used by the current test suite. |
| `strumpack` | Experimental | Not built by default; GPU claims are unvalidated for LinStabPy matrices. |
| `superlu_dist` | Experimental | Not built by default; PETSc requires a SuperLU_DIST build with GPU offload support for GPU use. |
| `cudss` | Not available by default | No released PETSc factor-solver interface is assumed by LinStabPy. |

See `docs/COMPUTE_BACKENDS.md` for the backend audit, version floors, and the
current GPU go/no-go notes.

### Resolvent region masks

`resolvent_map` accepts optional `input_region` and `output_region` dictionaries
that build sharp diagonal forcing/response masks from physical coordinates and
state variables. Omitted keys mean the full extent and all variables. Example
configuration fragment, assuming the usual example variables `base_flow` and
`case_cfg` are already loaded:

```python
from linstabpy import resolvent_map

cfg = {
    **case_cfg,
    "input_region": {"x": [0.02, 0.06], "y": [0.0, 0.01], "vars": ["u", "v", "w", "T"]},
    "output_region": {"x": [0.04, 0.08], "vars": ["rho", "u", "v", "w", "T"]},
}
result = resolvent_map(base_flow, beta_range=[20.0], f_range=[0.15], cfg=cfg)
```

A windowed/component-restricted gain changes the forcing/response question; it is
not an approximation of the global unmasked gain.

### Boundary-condition acceptance

Returned stability modes are postchecked against:

- full-operator residual
- boundary-condition leakage
- constrained-Dirichlet leakage
- target distance
- optional spectral boxes on `Re(omega)` and `Im(omega)`

An interior eigenvalue solve can report converged Ritz pairs and still hand back
modes that are unacceptable under the full operator and boundary treatment. The
postcheck is what catches that.

### Convergence principles

`eps_tol` alone does not define mode quality. For interior non-Hermitian problems,
tightening `eps_tol` does not guarantee proportional improvement in accepted
modes. Mode quality can be limited by `ncv`, target placement, operator
conditioning, non-normality, and spectral crowding.

`ncv` is a quality-control parameter. If the target neighborhood is crowded or
non-normal, too small an `ncv` can make accepted-mode quality plateau even when
the solve is active.

Eigenvalues can stabilize before eigenvectors. A spectrum that appears stable is
not enough; explicit postcheck residuals and boundary leakage still matter.

Shift-invert quality depends on the shifted operator. If `iL0 - sigma I` is poorly
conditioned, residuals can plateau even with direct LU.

Target choice matters. A target on the real axis can sit in a crowded
neighborhood. Moving the target slightly into the complex plane can isolate a
branch more cleanly. Treat this as a numerical search choice, not a change in
physics.

Branch-localize before scaling `nev`. Start with a small solve, test a few targets
around the branch of interest, judge success from postcheck residual and target
distance, then increase `nev` and `ncv`.

### What to check on every run

Before trusting a spectrum, check:

- requested `nev`
- actual `nconv`
- `ncv`
- EPS tolerance and EPS convergence test
- whether true residuals were enabled
- whether shifted solves were direct or iterative
- explicit postcheck residual counts
- boundary-condition leakage diagnostics
- target distance for accepted and rejected modes
- wall time
- peak RSS
- MPI rank count and thread policy
- the exact JSONC config and git commit that produced the artifact

### What not to assume

Do not assume:

- `nconv > 0` means the run succeeded
- passing the EPS convergence test implies a small full-operator residual
- lowering `eps_tol` is always the cheapest way to improve results
- a larger `nev` gives more trustworthy leading modes
- the fastest setup on one machine is the fastest setup on another
- a mean-flow linearization is physically meaningful just because the operator was assembled correctly

## Nondimensionalization

All flow fields stored in `BaseFlow` are nondimensional. The governing equations
are the compressible Navier–Stokes equations in the primitive variables
`[rho, u, v, w, T]`, linearized around the base flow. The nondimensionalization
follows the standard convective scaling described below.

### Reference scales

| Symbol | Definition | Units | Source in code |
|---|---|---|---|
| `rho_ref` | freestream density | kg/m³ | implicit: `RHO = 1` at freestream (`loaders.py`) |
| `U_ref` | freestream velocity | m/s | implicit: `U = 1` at freestream |
| `L_ref` | problem reference length | m | user-supplied (e.g. displacement thickness `delta*` for Blasius, shock-impingement length `X_sh` for SWBLI) |
| `T_ref` | freestream static temperature | K | `BaseFlow.T_0` (`baseflow.py:40`) |
| `p_ref` | `rho_ref * U_ref**2` | Pa | confirmed by Chu norm weights in `energy_norm.py` |
| `mu_ref` | `mu(T_0)` via Sutherland | Pa·s | `viscosity.py:46`, variable `mu_0` |
| `t_ref` | `L_ref / U_ref` | s | derived |

`L_ref` is not stored inside `BaseFlow`; it is the length scale used when
computing `Re` and when building the mesh coordinate arrays passed to the
loaders. Its physical interpretation depends on the case.

### Nondimensional variables

| Variable | Stored as | Nondim form | Freestream value |
|---|---|---|---|
| density | `BaseFlow.RHO` | `rho* = rho / rho_ref` | 1 |
| streamwise velocity | `BaseFlow.U` | `u* = u / U_ref` | 1 |
| wall-normal velocity | `BaseFlow.V` | `v* = v / U_ref` | 0 |
| spanwise velocity | `BaseFlow.W` | `w* = w / U_ref` | 0 |
| temperature | `BaseFlow.T` | `T* = T / T_ref` | 1 |
| viscosity | `BaseFlow.MU` | `mu* = mu / mu(T_0)` | 1 |
| streamwise coordinate | mesh `X` | `x* = x / L_ref` | — |
| wall-normal coordinate | mesh `Y` | `y* = y / L_ref` | — |
| time | operator eigenvalue `omega` | `t* = t * U_ref / L_ref` | — |

Pressure is not stored explicitly. From the ideal gas law nondimensionalized with
`p_ref = rho_ref * U_ref**2`:

```
p* = RHO * T / (kappa * Ma**2)
```

This `Ma**2` factor is absent in acoustic-scaled codes; see
[LES vs linstabpy convention notes](#les-vs-linstabpy-convention-notes).

### Dimensionless numbers

**Reynolds number**

```
Re = rho_ref * U_ref * L_ref / mu(T_0)
```

`mu(T_0)` is Sutherland's law evaluated at the freestream temperature `T_0`.
Enters viscous terms as `1/Re`; for example, `coefficients.py` has
`c_Arr_44 = -MU / Re` (streamwise momentum wall-normal diffusion).

**Mach number**

```
Ma = U_ref / a_ref,  a_ref = sqrt(kappa * R * T_ref)
```

Enters the operator via `c2 = kappa * Ma**2` (pressure-velocity coupling) and
`cv = 1 / (kappa * (kappa - 1) * Ma**2)` (energy equation). Both are derived in
`viscosity.py:34-36` and used throughout `coefficients.py`.

**Prandtl number**

```
Pr = mu(T_0) * cp / k_ref,  cp = kappa * R / (kappa - 1)
```

Enters thermal diffusion as `1 / (Re * Pr)`; for example,
`c_Arr_55 = -kappa * MU / Re / Pr` (temperature Laplacian in energy equation).

**Heat capacity ratio**

```
kappa = cp / cv  (gamma)
```

Stored as `BaseFlow.kappa`. The code uses `kappa` rather than `gamma`; the
`BaseFlow` docstring at `baseflow.py:27` notes "Heat capacity ratio (gamma)" to
keep the physical meaning explicit. Enters the ideal gas relation, the Chu norm
weights, and all compressibility terms.

Re, Ma, Pr, and kappa are user-supplied in the JSONC config under `physics`. The
code uses them as given without internal consistency checks.

### Sutherland viscosity

Dynamic viscosity follows Sutherland's law for air. The normalization reference is
`mu_0 = mu(T_0)`, the Sutherland value at the actual freestream temperature. The
relevant section of `viscosity.py`:

```python
    S1: float = 110.4  # Sutherland temperature [K]
    T_ref: float = 280.0  # reference temperature [K]
    mu_ref: float = 1.735e-5  # reference dynamic viscosity [Pa.s]

    Tstar = bf.T * T_0  # dimensional temperature

    # Reference viscosity at T_0
    mu_0 = mu_ref * (T_ref + S1) / (T_0 + S1) * (T_0 / T_ref) ** 1.5

    # Viscosity and its first two derivatives w.r.t. dimensional T
    mu = mu_ref * (T_ref + S1) / (Tstar + S1) * (Tstar / T_ref) ** 1.5
```

`T_ref = 280.0 K` and `mu_ref = 1.735e-5 Pa.s` are anchoring constants in the
Sutherland formula, not the nondimensionalization scale. The operative nondim
scale is `mu_0`, which depends on `T_0`. `MU = mu / mu_0`, so `MU = 1` at
freestream regardless of what `T_0` is.

The temperature derivatives are nondimensionalized consistently:

```
dmudT*  = d(mu/mu_0) / d(T*)  =  (dmu/dT_dim) * T_0 / mu_0
d2mudT2* = (d2mu/dT_dim**2) * T_0**2 / mu_0
```

### LES vs linstabpy convention notes

Three specific points where LES codes commonly differ.

**Pressure reference.** Some LES codes use `p_ref = rho_ref * R * T_ref`
(acoustic scaling). linstabpy uses `p_ref = rho_ref * U_ref**2` (dynamic
pressure). The two coincide only at `Ma = 1`. For `Ma != 1`, the nondim ideal gas
law differs by a factor of `kappa * Ma**2`:

| Quantity | linstabpy | acoustic-scaled LES |
|---|---|---|
| `p*` | `p / (rho_ref U_ref**2)` | `p / (rho_ref R T_ref)` |
| ideal gas | `p* = RHO * T / (kappa * Ma**2)` | `p* = RHO * T` |
| conversion | multiply acoustic `p*` by `kappa * Ma**2` | divide linstabpy `p*` by `kappa * Ma**2` |

**Energy equation scaling.** In linstabpy `cv* = 1 / (kappa * (kappa-1) * Ma**2)`.
An LES code that writes `cv* = 1 / (kappa - 1)` is using acoustic-based nondim. To
convert: `cv_LES = cv_linstabpy * kappa * Ma**2`.

**Temperature.** linstabpy stores `T* = T / T_ref` with `T* = 1` in the
freestream. Some LES post-processing outputs temperature as excess
`(T - T_inf) / T_inf` or as a ratio `(T - T_w) / (T_0 - T_w)`. If an external
base-flow file provides temperature already divided by `T_ref`, load it directly
into `BaseFlow.T`. If it provides temperature in Kelvin, divide by `T_0` before
assigning.

## Boundary conditions

Boundary setup is JSONC-driven. Row-replacement boundary conditions are listed
under `bc.recipe`; sponge damping is configured in JSONC as an operator damping
field or loaded from reference case data.

Example:

```jsonc
{
  "physics": {
    "sponge": {
      "enabled": true,
      "preset": "tanh",
      "amplitude": 0.1
    }
  },
  "bc": {
    "penalty": 1e8,
    "recipe": [
      {
        "type": "dirichlet",
        "borders": "lt",
        "variables": "ruvwT"
      },
      {
        "type": "dirichlet",
        "borders": "b",
        "variables": "ruvw"
      },
      {
        "type": "neumann",
        "borders": "b",
        "variables": "T",
        "exclude_corners": true,
        "curvilinear_normals": true
      },
      {
        "type": "neumann",
        "borders": "r",
        "variables": "ruvwT",
        "exclude_corners": true,
        "curvilinear_normals": true
      }
    ]
  },
  "analysis": {
    "stability_acceptance": {
      "postcheck_rel_max": 5e-5,
      "target_distance_max": 0.15,
      "omega_real_abs_max": 1.5,
      "omega_imag_abs_max": 2.0,
      "conjugate_aware": true
    }
  }
}
```

Recipe fields:

| Field | Meaning |
|---|---|
| `type` | `dirichlet` or `neumann` for row-replacement BCs. Sponge damping is configured separately under `physics.sponge` or loaded from case input data. |
| `borders` | Boundary letters. Common values are `l`, `r`, `t`, `b`, or combinations such as `lt`. |
| `variables` | Variable block selector, such as `u`, `T`, `ruvw`, or `ruvwT`. |
| `exclude_corners` | For Neumann rules, remove corner points that would overlap with other boundary rules. |
| `curvilinear_normals` | For Neumann rules, use per-face curvilinear normals instead of axis-aligned normals. |

Use `curvilinear_normals: true` for deformed or body-fitted curvilinear meshes.
The flag makes Neumann rows use per-face normals from the stored mesh Jacobian
entries. Set it to `false` when the base-flow mesh is Cartesian and the
curvilinear normal path is not needed; that preserves the axis-aligned path and
can save a small amount of work. Older parity configurations may spell this flag
explicitly to preserve historical behavior. New configs should spell the intended
value instead of relying on omission.

`analysis.stability_acceptance.conjugate_aware` defaults to `true` in the current
stability-acceptance path. It treats an eigenvalue and its complex conjugate as
equally close to the target:

```text
min(|omega - target|, |omega - conj(target)|)
```

This is the right default for real-valued base flows, where oscillatory modes come
in conjugate pairs. Set `conjugate_aware: false` when you need strict
non-redundant mode selection around one target branch, or when the operator is not
real-valued and conjugate symmetry should not be admitted.

## Parity with LinStab2D

### Example-level parity

Reference MATLAB examples:

- `linstabpy/data/baseflows/example_0_TestCurvilinearDerivative.m`
- `linstabpy/data/baseflows/example_1_Blasius.m`
- `linstabpy/data/baseflows/example_2_Jet.m`
- `linstabpy/data/baseflows/example_3_Streaky_BL.m`
- `linstabpy/data/baseflows/example_4_Cavity.m`
- `linstabpy/data/baseflows/example_5_Paraboloid.m`

Python counterparts:

- `examples/example_0_curvilinear_validation.py`
- `examples/example_1_blasius.py`
- `examples/example_2_jet.py`
- `examples/example_3_streaky_bl.py`
- `examples/example_4_cavity.py`
- `examples/example_5_paraboloid.py`

| MATLAB example | Python example | Status | Required artifacts |
|---|---|---|---|
| `example_0_TestCurvilinearDerivative.m` | `example_0_curvilinear_validation.py` | `implemented` | `example_0_curvilinear_validation.npz`, `example_0_curvilinear_validation.png`, `tests/fixtures/curvilinear_derivative_parity.json` |
| `example_1_Blasius.m` | `example_1_blasius.py` | `implemented` | `example_1_blasius.npz`, `example_1_blasius.png` |
| `example_2_Jet.m` | `example_2_jet.py` | `implemented` | `example_2_jet.npz`, `example_2_jet.png` |
| `example_3_Streaky_BL.m` | `example_3_streaky_bl.py` | `implemented` | `example_3_streaky_bl.npz`, `example_3_streaky_bl.png` |
| `example_4_Cavity.m` | `example_4_cavity.py` | `implemented` | `example_4_cavity.npz`, `example_4_cavity.png` |
| `example_5_Paraboloid.m` | `example_5_paraboloid.py` | `implemented` | `example_5_paraboloid.npz`, `example_5_paraboloid.png` |

### Operator parity matrix

Canonical machine-readable source: `scripts/validation/martini2024/tools/operator_matrix.json`.

Status semantics:

- `pending`: not implemented yet
- `implemented`: code exists but parity artifact/signoff is still pending
- `validated`: implemented and parity-validated by artifacts
- `deferred`: intentionally postponed
- `out_of_scope`: intentionally not targeted
- `blocked`: cannot proceed due to an external blocker

Every `required: true` entry must reach `status = validated`. Current gate status
is `pass` with `missing_required = 0`.

| ID | MATLAB | Python target | Status |
|---|---|---|---|
| `construct_get_lin_problem` | `ConstructOperators/GetLinProblem.m` | `linstabpy/operators/assembly.py::build_lnse_operator` | `validated` |
| `construct_get_spatial_lin_problem` | `ConstructOperators/GetSpatialLinProblem.m` | `linstabpy/operators/spatial.py::build_spatial_operator` | `validated` |
| `construct_cartesian_coefficients` | `ConstructOperators/getCoeffsLaminar_Cartesian.m` | `linstabpy/operators/coefficients.py::compute_cartesian_coefficients` | `validated` |
| `construct_axisymmetric_coefficients` | `ConstructOperators/getCoeffsLaminar.m` | `linstabpy/operators/coefficients.py::compute_cylindrical_coefficients` | `validated` |
| `construct_create_diff_matrices_axy` | `ConstructOperators/CreateDiffMatrices_Axy.m` | `linstabpy/operators/assembly.py::_create_axy_diff_matrices` | `validated` |
| `construct_lnseoperators_cyl` | `ConstructOperators/LNSEoperators_cyl.m` | `linstabpy/operators/assembly.py::build_lnse_operator(model="axy")` | `validated` |
| `construct_get_comp_energy_norm` | `ConstructOperators/GetCompEnergyNorm.m` | `linstabpy/operators/energy_norm.py::compute_chu_norm` | `validated` |
| `bc_dirichlet` | `BoundaryConditions/BC_Dirichlet.m` | `linstabpy/boundary_conditions/dirichlet.py::apply_dirichlet` | `validated` |
| `bc_dirichlet_spatial` | `BoundaryConditions/BC_Dirichlet_SpatialStability.m` | `linstabpy/boundary_conditions/dirichlet.py::apply_dirichlet_spatial` | `validated` |
| `bc_neumann` | `BoundaryConditions/BC_Neumann.m` | `linstabpy/boundary_conditions/neumann.py::apply_neumann` | `validated` |
| `bc_set_dirbc` | `BoundaryConditions/SetDirBC.m` | `linstabpy/boundary_conditions/dirichlet.py::build_dirichlet_restriction_matrices` | `validated` |
| `bc_set_dirichletbc` | `BoundaryConditions/SetDirichletBC.m` | `linstabpy/boundary_conditions/dirichlet.py::apply_dirichlet_identity_pair` | `validated` |

### Branch coverage matrix

Canonical machine-readable source: `scripts/validation/martini2024/tools/branch_matrix.json`.

Target dimensions:

- `model`: `2D`, `axy`
- `solver`: `temporal`, `spatial`, `adjoint`, `resolvent`
- `bc`: `Dirichlet`, `DirichletSpatial`, `Neumann`
- `deformation`: `none`, `x_only`, `y_only`, `full_2d`
- `backend`: `petsc_slepc`

Total combinations:

```text
2 * 4 * 3 * 4 * 1 = 96
```

All required combinations must be `validated`. Combinations omitted from `records`
default to `pending`. Scope is controlled by `required_exclusions` in
`scripts/validation/martini2024/tools/branch_matrix.json`. Current scoped gate
status is `pass` with `5` required validated combinations and `91` required
exclusions.

| model | solver | bc | deformation | backend | status |
|---|---|---|---|---|---|
| `2D` | `temporal` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `spatial` | `DirichletSpatial` | `none` | `petsc_slepc` | `validated` |
| `2D` | `adjoint` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `resolvent` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `temporal` | `Neumann` | `none` | `petsc_slepc` | `validated` |

### Running parity checks

Example-level parity:

```bash
.venv/bin/python scripts/validation/martini2024/tools/compare_martini_examples.py
```

Allow incomplete optional matrix entries while enforcing required scoped gates:

```bash
python scripts/validation/martini2024/tools/check_parity_matrices.py --allow-incomplete
```

Strict CI gate:

```bash
python scripts/validation/martini2024/tools/check_parity_matrices.py
```

Some parity tests compare against MATLAB reference export bundles (raw `.mat`
exports and compact `.npz` base bundles). Those bundles are intentionally kept out
of the repository and are not distributed. The corresponding tests skip cleanly
when the bundles are absent (so they are silent in CI) and validate against them
when the bundles are provided locally.

### Numerical equivalence after the assembly refactor

The operator-assembly path was refactored to fuse coefficient evaluation and to
build the block-sparse operator in a single pass. The assembled operator agrees
with the earlier element-by-element construction to machine precision (measured on
the Blasius case: `max |ΔL| ≈ 5e-16`, relative Frobenius `≈ 3e-17`). The one-pass
build also drops a handful of explicitly-stored numerical zeros, so its sparsity
pattern is a clean subset of the old one with the same values.

The matrices are not byte-identical. Fusing the coefficient expressions sums terms
in a different order, which flips the last floating-point bit, and the generalized
eigensolver amplifies that bit into eigenvalue movement of order `~1e-12`. That is
ordinary conditioning of the eigenproblem, not a loss of accuracy: the eigenpair
residual stays below `1e-6` and every parity check above passes at its stated
tolerance. `scripts/validation/lnse_equivalence_harness.py` compares assembly
paths against this contract.

Practical rule: order-sensitive checks must allow at least the solver's eigenvalue
tie tolerance (`tie_tol = 1e-10` in `solvers/eigenvalue.py`). Do not assert
bit-for-bit eigenvalue reproduction across the refactor boundary; that is not
achievable for any two mathematically-equal but differently-ordered assemblies in
float64.

## Case studies

### Case 1 (SWBLI)

Case 1 is the SWBLI parity harness used to prove that the Python/PETSc/SLEPc path
is equivalent to the MATLAB reference before scaling to larger production cases.

Objective:

- compare the Python stability and resolvent paths against MATLAB reference exports
- fix operator/source logic when parity fails
- avoid hardcoding MATLAB outputs as a substitute for correct assembly

Current status as of 2026-02-28:

- operator-stage parity: `PASS`
- resolvent parity: `PASS`
- stability spectrum parity is handled by the case script with awareness of which MATLAB export produced the available spectrum

Key fixes that made parity stable:

- preserve mesh filters during deformation, matching MATLAB behavior
- load `sponge_aux` from MATLAB `case_inputs_used.mat`
- use the MATLAB-equivalent `H^H W H` EVP route for resolvent extraction
- keep canonical parity parameters in the Case 1 fixture directory

Run from the repository root:

```bash
bash examples/case_studies/case1_song/case1.sh
```

Primary outputs:

- `examples/case_studies/case1_song/case1.npz`
- `examples/case_studies/case1_song/output/case1_operator_stage_parity.json`
- `examples/case_studies/case1_song/output/case1_spectrum_parity.json`
- `examples/case_studies/case1_song/output/case1_resolvent_parity.json`
- parity figures under `examples/case_studies/case1_song/output/`

`examples/case_studies/case1_song/` is the canonical Case 1 study path.

Environment overrides:

| Variable | Default | Use |
|---|---:|---|
| `LINSTABPY_CASE1_CASE_DIR` | `data/baseflow` | Case data directory, relative to `examples/case_studies/case1_song/` unless absolute. |
| `LINSTABPY_CASE1_RUN_RESOLVENT` | `false` | Enable or disable resolvent computation. |
| `LINSTABPY_CASE1_RUN_STABILITY` | `true` | Enable or disable temporal stability computation. |
| `LINSTABPY_CASE1_N_MODES` | `100` | Number of stability modes requested. |
| `LINSTABPY_CASE1_N_MODES_RESOLVENT` | `10` | Number of resolvent modes requested. |
| `LINSTABPY_CASE1_STRIDE_KSI` | `1` | Mesh stride in the first computational coordinate. |
| `LINSTABPY_CASE1_STRIDE_ETA` | `1` | Mesh stride in the second computational coordinate. |
| `LINSTABPY_CASE1_BC_PENALTY` | `1e8` | Boundary-condition penalty used by the case driver. |

Spectrum comparison from the repository root:

```bash
.venv/bin/python examples/case_studies/case1_song/scripts/check_spectrum.py \
  --python-npz examples/case_studies/case1_song/case1.npz \
  --matlab-spectrum examples/case_studies/case1_song/reference/matlab/MATLAB_results/spectrum.mat
```

Solver strategy A, the robust baseline:

- `ST=SINVERT`
- `KSP=PREONLY`
- `PC=LU`
- PETSc MUMPS factorization
- main accuracy/selection knobs are `eps_tol`, `conv_test`, `true_residual`, `nev`, `ncv`, and target shift
- use this path first when direct factorization fits memory

Solver strategy B, the scalable path:

- `ST=SINVERT` or a preconditioned eigensolver route
- iterative `KSP` with strict `ksp_error_if_not_converged`
- block Jacobi or ASM with local LU as the current baseline direction
- stronger physics-block preconditioners remain a tuning track
- use this path when the direct memory budget is exceeded, and keep parity and convergence gates enabled

Practical rule: validate reduced grids with Strategy A, then move to Strategy B
for cases that do not fit direct factorization.

### Case 2

Case 2 has its own active campaign files. Do not duplicate their content in the
top-level README.

- `examples/case_studies/case2_lui/docs/PLAN.md`: current phase plan, acceptance criteria, open blockers, and campaign state.
- `examples/case_studies/case2_lui/docs/REFERENCE.md`: stable technical background, governing equations, parameter map, physical framing, and references.
- `examples/case_studies/case2_lui/docs/README.md`: operational commands and artifact layout.

Read them in that order before changing Case 2 numerics. As with Case 1,
`examples/case_studies/case2_lui/` is the canonical Case 2 study namespace.

## Troubleshooting

### macOS SLEPc NHEP crash

Symptom:

- Python `slepc4py` `EPS_NHEP` with Krylov-Schur / Arnoldi aborts
- native C SLEPc NHEP works
- Python HEP and LAPACK-backed paths work

Root cause: Apple Accelerate LAPACK symbols can be interposed ahead of the LAPACK
symbols expected by PETSc-downloaded ScaLAPACK inside a Python process.

Fix:

- install Homebrew OpenBLAS and ScaLAPACK
- run `bash install.sh` so PETSc is configured with those libraries when possible
- use the `.venv/bin/python` wrapper generated by the installer

Detection:

```bash
bash install.sh --check
```

The check includes a tiny Python SLEPc `EPS_NHEP` Krylov-Schur solve.

### Linux MPI compile failure

If `diagnose.sh` reports an MPI compile failure with `mpi.h` missing, install the
development package for the MPI stack you intend to use. On Ubuntu with Open MPI,
the needed packages are:

```bash
sudo apt install -y libopenmpi-dev openmpi-bin
```

On clusters, load the complete MPI/compiler module instead of installing system
packages.

### Dual-MPI conflict

If Open MPI, MPICH, and/or Intel MPI are all visible, choose one stack, make it
first in `PATH`, and reinstall from a clean `.venv`. Do not let PETSc build against
one MPI while `mpiexec` launches another.

### `mpi4py` ABI mismatch

If `mpi4py` imports fail after switching MPI stacks:

```bash
rm -rf .venv
uv cache clean
LINSTABPY_MPI4PY_NO_BINARY=1 bash install.sh
```

### PETSc configure hang

If PETSc configure hangs or fails on the wrong launcher:

```bash
LINSTABPY_PETSC_MPIEXEC=/path/to/working/mpiexec bash install.sh
```

Use the same launcher for verification when needed:

```bash
LINSTABPY_VERIFY_MPIEXEC=/path/to/working/mpiexec bash install.sh --check
```

### Intel MPI

Load a complete Intel MPI / oneAPI environment before running `diagnose.sh` or
`install.sh`. Do not combine an Intel MPI `mpiexec` with Open MPI or MPICH Python
bindings.

```bash
source /opt/intel/oneapi/setvars.sh --force
export LINSTABPY_MPI_STACK=intelmpi
bash diagnose.sh
bash install.sh
```

### What to send when a machine is still broken

Send:

- full `diagnose.sh` output
- full `install.sh` output
- `benchmark_runs/<run-id>/benchmark_summary.csv`
- relevant `benchmark_runs/<run-id>/*.log`
- a redacted stack manifest:

```bash
.venv/bin/python scripts/tools/export_known_good_stack.py \
  --output stack_manifest.json \
  --redact-local-details
```

For deeper OS-specific notes and installer experiments, see `scripts/ci/`.

## Development

### Branching and releases

`main` is the stable branch: it always passes the full test suite, every
release is tagged on it, and it advances only through a reviewed pull request
from `dev`. Day-to-day work lands on `dev` (versioned `0.0.x.devN`), is
verified on a big-memory workstation, and is promoted to `main` in small,
individually tagged `0.0.x` increments. Branch roles, the versioning scheme,
the merge gate, and the release steps are documented in
`docs/DEVELOPMENT.md`.

Run tests with:

```bash
.venv/bin/python -m pytest tests/
```

Check the expected collection size:

```bash
.venv/bin/python -m pytest --collect-only tests
```

Current expected result: `535` collected tests.

Run MATLAB parity checks with:

```bash
.venv/bin/python scripts/validation/martini2024/tools/compare_martini_examples.py
python scripts/validation/martini2024/tools/check_parity_matrices.py --allow-incomplete
python scripts/validation/martini2024/tools/check_parity_matrices.py
```

Packaging:

- build metadata lives in `pyproject.toml`
- importable code uses the `src/` layout
- package name is `linstabpy`
- import package is `linstabpy`
- optional PETSc/SLEPc dependencies are declared as `pip install linstabpy[petsc]`

For development from a checkout, prefer the repository installer because it builds
and verifies PETSc/SLEPc:

```bash
bash install.sh
.venv/bin/python -m pip install -e .
```

Reference MATLAB source boundary:

- Python implementation: `src/linstabpy/`, `tests/`, `examples/`, `scripts/`, and repository docs
- MATLAB reference source and `.mat` assets: `src/linstabpy/data/baseflows`
- update or replace reference assets only with explicit intent
- do not edit reference MATLAB files while trying to fix Python parity

Roadmap:

| Area | Status | Notes |
|---|---|---|
| Core assembly and reference examples | Done | SBP operator family (fd2–fd8) and the Martini examples, parity-gated. |
| PETSc/SLEPc solvers | Done | MUMPS sparse-direct temporal/spatial/adjoint EVP and resolvent SVD paths. |
| Independent operator verification | Done | Provenance-independent oracles for the coefficients, global assembly, boundary rows, curvilinear transform, and energy norm. |
| SWBLI validation and resolvent results | In progress | Song/Robinet global-stability reproduction and a new SWBLI resolvent gain map. |
| Factor-backend seam | Done | Central backend selection for PETSc LU paths; `mumps` is the validated default. |
| GPU sparse-direct backends | In progress | `strumpack`, `superlu_dist`, and `cudss` paths are under active development; `mumps` stays the validated CPU default until the GPU paths are verified for LinStabPy matrices. |

## References

**Methodology and lineage**

Martini, E. & Schmidt, O. T. (2024). LinStab2D: stability and resolvent analysis
of compressible viscous flows in MATLAB. *Theoretical and Computational Fluid
Dynamics*, 38(5), 665-685. doi:10.1007/s00162-024-00706-0.

Padovan, A., Anantharaman, V., Rowley, C. W., Vollmer, B., Colonius, T. & Bodony,
D. J. (2025). resolvent4py: a parallel Python package for analysis, model
reduction and control of large-scale linear systems. *SoftwareX*, 31, 102286.
doi:10.1016/j.softx.2025.102286.

**Solver stack**

Balay, S. et al. (2025). *PETSc/TAO Users Manual* (ANL-21/39, Revision 3.24).
Argonne National Laboratory. doi:10.2172/2998643.

Hernández, V., Román, J. E. & Vidal, V. (2005). SLEPc: a scalable and flexible
toolkit for the solution of eigenvalue problems. *ACM Transactions on
Mathematical Software*, 31(3), 351-362. doi:10.1145/1089014.1089019.

Amestoy, P. R., Duff, I. S., Koster, J. & L'Excellent, J.-Y. (2001). A fully
asynchronous multifrontal solver using distributed dynamic scheduling (MUMPS).
*SIAM Journal on Matrix Analysis and Applications*, 23(1), 15-41.

**Discretization and norms**

Strand, B. (1994). Summation by parts for finite difference approximations for
d/dx. *Journal of Computational Physics*, 110(1), 47-67.

Mattsson, K. & Nordström, J. (2004). Summation by parts operators for finite
difference approximations of second derivatives. *Journal of Computational
Physics*, 199(2), 503-540.

Chu, B.-T. (1965). On the energy transfer to small disturbances in fluid flow
(Part I). *Acta Mechanica*, 1(3), 215-234.

Schmid, P. J. & Henningson, D. S. (2001). *Stability and Transition in Shear
Flows*. Springer.

## License and citation

LinStabPy is distributed under **GPL-3.0-only**. The license is inherited from its
MATLAB ancestor LinStab2D (Martini and Schmidt, 2024,
https://github.com/LinStab2D/LinStab2D), which is GPL-3.0. See `LICENSE` for the
full text and `NOTICE` for attribution and the modification history.

When using LinStabPy in published work, please cite the LinStabPy paper together
with the LinStab2D methodology paper:

> Hamada, G. Y. R. & Frantz, R. A. S. linstabpy: open-source compressible
> stability and resolvent analysis. *Theoretical and Computational Fluid
> Dynamics* (in preparation, 2026).

> Martini, E. & Schmidt, O. T. LinStab2D: stability and resolvent analysis of
> compressible viscous flows in MATLAB. *Theor. Comput. Fluid Dyn.* 38, 665-685
> (2024). doi:10.1007/s00162-024-00706-0

See `CITATION.cff` for the machine-readable entry. The software DOI and release
version will be added once the first tagged release is archived on Zenodo.
