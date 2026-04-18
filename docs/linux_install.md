# Linux Install And Runtime Policy

This project is sensitive to the exact Linux runtime stack. The biggest
observed cross-machine failures were not "wrong answers" but extreme runtime
variance caused by:

- mixing MPI stacks on the same host
- rebuilding against one MPI and running with another
- letting MPI ranks and OpenMP / BLAS threads oversubscribe each other
- silently switching into host-sensitive direct PETSc assembly paths

The practical baseline below is the install policy that has been the most
consistent across workstations and cluster nodes.

## Baseline policy

1. Use exactly one MPI stack per environment.
   PETSc, `mpi4py`, and `mpiexec` must agree on the same ABI.
2. Run `bash diagnose.sh` before installing.
   The script checks compilers, MPI launchers, dual-MPI conflicts, and probes
   which `mpiexec` can actually run `-n 1 /bin/true`.
3. Install with `bash install.sh`.
   The main installer now prefers a verified `mpiexec`, rebuilds `mpi4py` from
   source when the wheel ABI does not match the system MPI, and verifies the
   stack under `mpiexec -n 1` on Linux. If required system packages are
   missing, it prints the suggested package-manager commands and exits instead
   of trying `sudo` by default.
4. Under MPI, pin threaded math libraries to one thread per rank.
   Use:

   ```bash
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1
   export VECLIB_MAXIMUM_THREADS=1
   ```

5. Start with a conservative rank count.
   On multi-socket machines, begin with half of the physical cores or one
   socket, then compare against all physical cores. Do not assume "more ranks"
   is faster for sparse-direct workflows.
6. Keep LNSE assembly on the SciPy baseline unless you explicitly want to test
   direct PETSc assembly.
   The shipped example JSON files now default to `"assembly_backend": "scipy"`.
   Use `"petsc"` only as an explicit opt-in after local benchmarking.

## Why this policy changed

The repository already contains evidence of unstable host-dependent performance
in `install_tester/benchmark_runs/`:

- `supermicro`: Blasius benchmark around `70 s`
- `lsa`: Blasius benchmark around `11,213 s`

The key point from those logs is that the bad host was not slow in the
PETSc/MUMPS solve. It was slow in `build_lnse_operator`, which means the
problem was in the assembly/runtime path, not the eigensolver itself.

## Recommended workflow on a new machine

```bash
git clone --recurse-submodules https://github.com/ricardofrantz/linstabpy.git
cd linstabpy

bash diagnose.sh
bash install.sh
bash install.sh --check

# quick smoke benchmark (Example 0 + Blasius smoke)
bash benchmark.sh
```

If you already manage compilers and MPI yourself:

```bash
LINSTAB2D_SKIP_SYS_DEPS=1 bash install.sh
```

If you are on a workstation and want the installer to use the package manager
for missing system tools:

```bash
LINSTAB2D_AUTO_INSTALL_SYS_DEPS=1 bash install.sh
```

Cluster-friendly flow:

```bash
bash diagnose.sh
module load <compiler> <mpi> <hdf5>
bash install.sh
```

Useful overrides:

- `LINSTAB2D_PETSC_MPIEXEC=/path/to/mpiexec`
- `LINSTAB2D_VERIFY_MPIEXEC=/path/to/mpiexec`
- `LINSTAB2D_MPI4PY_NO_BINARY=1`
- `LINSTAB2D_WITH_H5PY=1`
- `LINSTAB2D_SKIP_SYS_DEPS=1`
- `LINSTAB2D_AUTO_INSTALL_SYS_DEPS=1`

HDF5 policy:

- The default install and verification path does not require HDF5 or `h5py`.
- Enable `LINSTAB2D_WITH_H5PY=1` only if you explicitly need MATLAB v7.3 / HDF5
  parity tooling or the related optional tests/scripts.

## Known-good reference stack

This repository now carries one tested Linux reference stack from a machine
where `diagnose.sh`, `install.sh --check`, `run_mpi_example.sh`, and
`benchmark.sh local-smoke` all passed:

- [known_good/linux-ubuntu24.04-openmpi4.1.6-python3.13.json](known_good/linux-ubuntu24.04-openmpi4.1.6-python3.13.json)

Treat it as a comparison target, not a universal hard pin. The main use is to
diff a problematic machine against a working one:

```bash
.venv/bin/python tools/export_known_good_stack.py --output docs/known_good/<machine>.json
```

If you intend to commit the exported file, add `--redact-local-details`.

The exported manifest records:

- Python package versions
- compiler + MPI toolchain versions
- the MPI launcher that passed the `-n 1 /bin/true` probe
- the one-thread-per-rank runtime policy
- an optional benchmark summary if `benchmark_runs/*/benchmark_summary.csv` exists

## Running examples

Always run examples as modules from the repository root:

```bash
.venv/bin/python -m examples.example_1_blasius
.venv/bin/python -m examples.example_3_streaky_bl
```

For a small PETSc smoke test that does not overwrite the full Example 1
artifacts:

```bash
.venv/bin/python -m examples.example_1_blasius_smoke
bash run_mpi_example.sh -n 1 -m examples.example_1_blasius_smoke
```

For MPI runs in general, prefer the wrapper:

```bash
bash run_mpi_example.sh -m examples.example_1_blasius_smoke
bash run_mpi_example.sh -n 10 -m examples.example_3_streaky_bl
```

It probes a working `mpiexec` and exports `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and
`VECLIB_MAXIMUM_THREADS=1` unless you override them yourself.

## Scaling guidance

- Use the smoke benchmark first to catch obvious MPI/assembly issues.
- If the smoke case is healthy, benchmark the full case with explicit rank
  counts instead of relying on machine defaults.
- Compare at least:
  - half of physical cores, `OMP_NUM_THREADS=1`
  - all physical cores, `OMP_NUM_THREADS=1`
- For direct sparse solves, a second socket can help, hurt, or do almost
  nothing. Measure; do not assume.

## Install-time troubleshooting

- If PETSc configure hangs or fails on `/usr/bin/mpiexec`, rerun with
  `LINSTAB2D_PETSC_MPIEXEC=/path/to/working/mpiexec`.
- If `mpi4py` imports fail after switching MPI stacks, remove `.venv`, run
  `uv cache clean`, and reinstall.
- If both Open MPI and MPICH are installed, decide which one you want first,
  remove or hide the other from `PATH`, then reinstall from a clean venv.
