# Installation

One supported numerical stack: **CPU, complex-scalar PETSc/SLEPc, MUMPS**.
That is the merge-gate and example path. Two ways to build it: `install.sh`
or `environment.yml`. GPU is not an install path yet.

| Path | Prefix | Role |
|---|---|---|
| `install.sh` | `.venv` | Reference workstation build. Default PETSc 3.24 with MUMPS. |
| `environment.yml` | conda env `linstabpy` | Same contract from conda-forge. pip installs this package only. |
| GPU / `environment-gpu.yml` | — | Not shipped. See [GPU and extra stacks](#gpu-and-extra-stacks). |

`docs/COMPUTE_BACKENDS.md` is the backend audit. This section is how you install.

## Prerequisites

Use the repository scripts first. They check compilers, [MPI](https://grokipedia.com/page/Message_Passing_Interface), Python bindings,
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

## Linux (recommended): `install.sh`

Clone the repository and run the native checks:

```bash
git clone https://github.com/openfluids/linstabpy.git
cd linstabpy

bash install.sh --diagnose
bash install.sh
bash install.sh --check
bash benchmark.sh
```

The reference MATLAB base flows (`M09jet.mat`, `BL_BF.mat`, `Step_BF.mat`,
`Paraboloid_BF.mat`) used by the Martini example loaders ship in
`examples/linstab2d/baseflows/`. See
[Reference base flows](https://github.com/openfluids/linstabpy#reference-base-flows)
for the data of the other examples.

`install.sh --diagnose` is read-only. It checks compilers, MPI launchers, development
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

`benchmark.sh` runs a small benchmark and writes benchmark artifacts under
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

## Conda: `environment.yml` (CPU)

This file is the CPU path for people who already use conda. conda-forge supplies
Python, NumPy/SciPy, and a **complex-scalar** PETSc/SLEPc/`mpi4py` stack.
pip then installs this package from `pyproject.toml` with `--no-deps` and
`-e .`, so pip does not pull a second NumPy or compile PETSc.

Python is `>=3.10,<3.14`. 3.13 is the class that matches the installer default.
3.14 stays an `install.sh` freshness test; conda-forge `petsc4py` may not
resolve it.

```bash
# Unset PETSC_DIR, PETSC_ARCH, and SLEPC_DIR first.
# A leftover PETSc prefix will bind petsc4py to the wrong build.
conda env create -f environment.yml
conda activate linstabpy
python -c "import numpy as np; from petsc4py import PETSc; assert np.issubdtype(PETSc.ScalarType, np.complexfloating)"
```

After the PyPI release, change the pip line `-e .` to `linstabpy` and recreate
the environment. Do not add `petsc4py`, `slepc4py`, or `mpi4py` to the pip
list. Those PyPI sdists compile PETSc from source and fight the conda MPI
stack.

`environment.yml` is not a second reference build. If conda-forge PETSc is a
different minor than `install.sh` (3.24 + MUMPS), record both versions. A
failing SLEPc NHEP probe or a real-scalar PETSc means the env is unusable for
this code, even if `import linstabpy` works.

## GPU and extra stacks

Do not add `environment-gpu.yml` or a second named conda env. CPU and GPU PETSc
cannot share one prefix, so two isolated installs will be the right shape
**after** a GPU stack has a green `install.sh --check` and a passing
EVP/resolvent residual. That has not happened.

| Stack | How it exists today |
|---|---|
| CPU, complex PETSc, MUMPS | Live. `.venv` via `install.sh`, or conda env `linstabpy`. |
| `gpu-strumpack` | Design only. `bash install.sh --stack gpu-strumpack --dry-run` prints the plan. `--force` may only touch `.venv-gpu-strumpack`, never `.venv`. No GPU build is claimed. |
| SuperLU_DIST + CUDA, complex | Tested on one CUDA build; solve residuals were order one. Rejected for that path. |

`LINSTABPY_FACTOR_BACKEND=strumpack` is an experimental **CPU** factor choice
when that library is in the PETSc build. It is not a GPU install. See
`docs/COMPUTE_BACKENDS.md`.

## macOS

macOS uses the same `install.sh --diagnose` → `install.sh` → `install.sh --check` →
`benchmark.sh` flow as Linux. The only difference is the [BLAS](https://en.wikipedia.org/wiki/BLAS)/ScaLAPACK setup
below.

For the macOS PETSc/SLEPc path, install Homebrew OpenBLAS and ScaLAPACK before
building when possible:

```bash
brew install openblas scalapack
```

The reason is a macOS runtime-library ordering issue in Python. PETSc's
downloaded ScaLAPACK can resolve [LAPACK](https://en.wikipedia.org/wiki/LAPACK) symbols to Apple Accelerate when loaded
inside Python. The failing path observed in this repository was Python
`slepc4py` `EPS_NHEP` with Krylov-Schur / [Arnoldi](https://grokipedia.com/page/Arnoldi_iteration), aborting inside Accelerate
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
.venv/bin/python -m examples.linstab2d.example_1_blasius
```

## Clusters and Intel MPI

On clusters, load the compiler/MPI modules before running the installer. The
repository scripts should see the same `mpicc`, `mpicxx`, `mpifort`, and
`mpiexec` that will be used at runtime.

Generic module workflow:

```bash
module load <compiler-module> <mpi-module>
bash install.sh --diagnose
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

bash install.sh --diagnose
bash install.sh
bash install.sh --check
```

Use Intel MPI only when `install.sh --diagnose` shows that the active MPI compiler wrappers
and `mpiexec` come from the Intel MPI environment and the MPI compile probe
passes. Do not use an Intel MPI launcher with Open MPI or MPICH Python bindings.

## Environment variables

| Variable | Default | Use |
|---|---:|---|
| `LINSTABPY_WITH_H5PY` | `0` | Set to `1` to install `h5py` for MATLAB v7.3 / HDF5 parity tooling. Leave unset for the normal solver install. |
| `LINSTABPY_MPI_STACK` | auto | Set to `intelmpi` when an Intel MPI environment is already loaded and should be used. Otherwise the scripts infer the active stack. |
| `LINSTABPY_AUTO_INSTALL_SYS_DEPS` | `0` | Set to `1` to allow `install.sh` to call the system package manager for missing Linux dependencies. |
| `LINSTABPY_PETSC_MPIEXEC` | auto | Path to the `mpiexec` PETSc/SLEPc configure should use. Set this when the default launcher hangs or is not the runtime launcher you want. |
| `LINSTABPY_MPI4PY_NO_BINARY` | `0` | Set to `1` to force `mpi4py` to build from source against the active MPI stack. |
| `LINSTABPY_VERIFY_MPIEXEC` | auto | Path to the `mpiexec` used by `install.sh --check` for MPI import/probe checks. |
| `LINSTABPY_SKIP_SYS_DEPS` | `0` | Set to `1` to skip Linux system dependency detection and package-manager advice. |

## Activation and shell aliases

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

## MPI stack verification

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
bash install.sh --diagnose
bash install.sh --force
bash install.sh --check
```

For clusters with Intel MPI, source oneAPI or the site module first and make that
choice explicit:

```bash
source /opt/intel/oneapi/setvars.sh --force
export LINSTABPY_MPI_STACK=intelmpi
bash install.sh --diagnose
bash install.sh --force
bash install.sh --check
```

Do not repair a mixed MPI install by editing `LD_LIBRARY_PATH` until it happens
to import. Reinstall with the intended MPI stack visible from the start.

## Known-good reference stacks

A healthy Linux install should satisfy this sequence:

```bash
bash install.sh --diagnose
bash install.sh --check
bash run_all_examples.sh --mpi -n 1 -m examples.linstab2d.example_1_blasius
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

Use this as a comparison shape, not a hard pin. A working install records the
Python version, PETSc/SLEPc versions, and scalar type in `install.sh --check`.

## MPI runtime rules

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
bash run_all_examples.sh --mpi -m examples.linstab2d.example_1_blasius
bash run_all_examples.sh --mpi -n 10 -m examples.linstab2d.example_3_streaky_bl
bash run_all_examples.sh --mpi -n 1 -m examples.linstab2d.example_1_blasius -- -eps_monitor
```

If `-n/--ranks` is omitted in `--mpi` mode, the runner starts from half of the
detected physical cores. Override the launcher with `LINSTABPY_MPIEXEC` and the
default rank count with `LINSTABPY_MPI_RANKS` when the target machine needs an
explicit MPI runtime. For sparse direct factorization, more ranks are not
automatically faster; compare rank counts on the target machine.
