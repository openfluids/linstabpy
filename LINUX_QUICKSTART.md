# Linux Quickstart

This is the short checklist for installing and validating `linstabpy` on a new
Linux machine without guessing MPI / OMP / BLAS settings by hand.

## Copy-paste install

```bash
git clone --recurse-submodules https://github.com/ricardofrantz/linstabpy.git
cd linstabpy

bash diagnose.sh
bash install.sh
bash install.sh --check
bash benchmark.sh
```

If `diagnose.sh` reports missing tools on a cluster, load the required modules
first and then rerun `install.sh`.

By default, `install.sh` does not call `sudo` or the package manager. If system
tools are missing, it prints the suggested commands and stops.

If the machine is already provisioned and you want to skip the system-deps
check entirely:

```bash
LINSTAB2D_SKIP_SYS_DEPS=1 bash install.sh
```

If you are on a normal workstation and want the installer to call the package
manager itself:

```bash
LINSTAB2D_AUTO_INSTALL_SYS_DEPS=1 bash install.sh
```

## What each step does

- `bash diagnose.sh`
  Checks compilers, MPI tools, dual-MPI conflicts, and probes which `mpiexec`
  actually works on the machine.
- `bash install.sh`
  Builds the Python environment, PETSc/SLEPc, and repairs `mpi4py` if the wheel
  ABI does not match the machine MPI. On Linux it refuses to auto-run
  `sudo`/package-manager commands unless you explicitly opt in with
  `LINSTAB2D_AUTO_INSTALL_SYS_DEPS=1`.
- `bash install.sh --check`
  Validates that `petsc4py`/`slepc4py` load correctly and that PETSc uses
  complex scalars.
- `bash benchmark.sh`
  Runs a small smoke benchmark. This is the fastest way to catch a machine that
  is technically installed but still misconfigured for performance.

## How to run examples after install

Serial / local:

```bash
.venv/bin/python -m examples.example_1_blasius_smoke
.venv/bin/python -m examples.example_3_streaky_bl
```

MPI with conservative thread settings:

```bash
bash run_mpi_example.sh -m examples.example_1_blasius_smoke
bash run_mpi_example.sh -n 10 -m examples.example_3_streaky_bl
```

`run_mpi_example.sh` exports:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

and picks a working `mpiexec` automatically.

## Rules that matter

1. Use exactly one MPI stack per environment.
   Do not mix Open MPI, MPICH, and Intel MPI on the same install session.
2. Under MPI, use one thread per rank unless you are benchmarking a different
   layout on purpose.
3. Start with half of the physical cores, then compare to all physical cores.
   More ranks are not always faster for sparse-direct workflows.
4. The shipped example JSONs now use the conservative SciPy assembly baseline.
   If you want to test direct PETSc assembly, opt into it explicitly in the
   JSON config instead of relying on `auto`.
5. On clusters, treat `diagnose.sh` as the preflight and `install.sh` as the
   Python/PETSc builder. Load modules first; do not rely on the installer to
   decide your system MPI/compiler stack for you.

## What “good” looks like

Typical healthy behavior:

- `diagnose.sh` ends with `ALL CHECKS PASSED`
- `install.sh --check` prints PETSc/SLEPc versions and `All checks passed.`
- `benchmark.sh` finishes and writes:
  - `benchmark_runs/<id>/benchmark_results.json`
  - `benchmark_runs/<id>/benchmark_summary.csv`

## What to send back if a machine is still weird

If the machine still behaves badly, send these files:

- the full `diagnose.sh` output
- the `install.sh` output
- `benchmark_runs/<id>/benchmark_summary.csv`
- the relevant `benchmark_runs/<id>/*.log`
- a local stack manifest from:
  `.venv/bin/python tools/export_known_good_stack.py --output docs/known_good/<machine>.json`

That is enough to tell whether the problem is:

- wrong MPI launcher
- MPI ABI mismatch
- thread oversubscription
- assembly-path slowdown
- or a genuine machine/topology-specific performance issue

## More detail

For the longer Linux policy and troubleshooting notes, see
[docs/linux_install.md](docs/linux_install.md).
