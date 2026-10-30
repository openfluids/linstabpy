# Troubleshooting

## Mixed MPI (most common broken install)

Symptom: configure picks one MPI, runtime loads another; crashes or
wrong-launcher errors. Fix: wipe and rebuild from a clean env — one MPI
stack for launcher, mpi4py, PETSc, SLEPc, and mpiexec. Never paper over
with `LD_LIBRARY_PATH`. Useful knobs: `LINSTABPY_MPI_STACK=intelmpi`,
`LINSTABPY_PETSC_MPIEXEC=/path/to/mpiexec` (configure),
`LINSTABPY_VERIFY_MPIEXEC=/path/to/mpiexec` (install checks),
`LINSTABPY_MPIEXEC=/path/to/mpiexec` (example runner only),
`LINSTABPY_MPI4PY_NO_BINARY=1` (build mpi4py from source),
`LINSTABPY_AUTO_INSTALL_SYS_DEPS=1` (let install.sh use the package
manager), `LINSTABPY_SKIP_SYS_DEPS=1` (provisioned cluster nodes).

## macOS BLAS abort in NHEP solves

PETSc-downloaded ScaLAPACK can resolve LAPACK symbols to Apple
Accelerate, aborting `slepc4py` solves inside `ZLADIV`. Prevention:
`brew install openblas scalapack` before installing; the installer
prefers Homebrew BLAS and wraps `.venv/bin/python` so the right BLAS
loads first. Always run `.venv/bin/python` directly, never `uv run`.
`install.sh --check` probes this before examples run.

## "Cache valid" but you changed something

Rank 0 compares the output NPZ's stored config hash + git SHA against
current ones and skips the compute on match. Either point each run at
its own output path or export `LINSTABPY_FORCE_RECOMPUTE=1`.

## Ran the wrong case (exit 0, warning on stderr)

`parse_known_args` warns (`warning: ignoring unknown arguments [...]`)
then drops unknown bare words and `--flags`. There are no case flags — check
`LINSTABPY_CASE_CONFIG` / `LINSTABPY_CASE_OUTPUT_NPZ`.

## Slow parallel runs

Set one thread per rank (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1`) or each rank spawns a full BLAS pool.
Factorization dominates: compare rank counts on your machine instead of
assuming more is faster. Report slowness with the NPZ `cost_wall_*`
keys plus `--diagnose` output.

## Missing data errors

Martini examples read the shipped `.mat` base flows in
`examples/linstab2d/baseflows/`. If you set `LINSTABPY_BASEFLOW_DATA_DIR`,
it must contain those files (`FileNotFoundError` otherwise). Savarino needs
`fetch_zenodo.py` first. Song/Sawtooth ship packed NPZs for checks.

## Conda path gotchas

Unset `PETSC_DIR PETSC_ARCH SLEPC_DIR` before creating the env; never
pip-install petsc4py/slepc4py/mpi4py into it (sdists compile PETSc and
fight the conda MPI). Gate: `python -c "from petsc4py import PETSc;
assert np.issubdtype(PETSc.ScalarType, np.complexfloating)"`.
