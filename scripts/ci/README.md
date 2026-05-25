# install_tester

Installer harness for testing the canonical LinStabPy checkout on another
machine.

`scripts/ci/` no longer contains a duplicate `src/` or `examples/` tree.
By default the harness treats the parent directory as the repository root. If
you copy the harness elsewhere, set:

```bash
export LINSTAB2D_REPO_ROOT=/path/to/linstabpy
```

## Layout

```text
scripts/ci/
  diagnose.sh             # read-only system check (recommended first step)
  install.sh              # wrapper around ../install.sh with preflight snapshot
  benchmark.sh            # wrapper around ../benchmark.sh
  run_install_tester.sh   # orchestrator: diagnose -> preflight -> install -> check
  README.md
  lovelace_requirements.md   # Cenapad Lovelace cluster: module load recipe
```

## Cluster: Lovelace (Cenapad / OHPC)

On **lovelace** (no `sudo`), load compiler, CMake, MVAPICH2, and **HDF5 for MVAPICH2** before install — see **[lovelace_requirements.md](lovelace_requirements.md)** for the exact `module load` lines (including `hdf5/1.12.1-mvapich2`) and `LINSTAB2D_SKIP_SYS_DEPS=1`.

## Benchmark examples

Only examples that are fully analytical (no external `.mat` baseflow data) are used:

- `example_0_curvilinear_validation.py` — curvilinear coordinate validation
- `example_1_blasius_smoke.py` — reduced Blasius boundary layer check

Other examples (2-5) require the `data/baseflows` submodule and are excluded
from the benchmark harness by default.

## Usage

### Step 1: Diagnose (recommended)

```bash
cd /path/to/linstabpy/install_tester
bash diagnose.sh
```

Reports all tool versions, detects MPI conflicts (e.g. both Open MPI and MPICH installed),
and prints exact `apt install` / `dnf install` commands to fix any gaps.
Nothing is installed or modified — output goes to stdout only.

Fix any issues reported, then proceed to Step 2.

### Step 2: Install

```bash
bash run_install_tester.sh <machine_label>
```

If `<machine_label>` is omitted, the hostname is used.

The orchestrator runs these steps in sequence:
1. **Diagnose** — runs `diagnose.sh` (aborts if critical gaps found).
2. **Preflight** — writes a machine/tool version snapshot only.
3. **Install** — runs the repository-root installer.
4. **Check** — post-install verification.

### Run steps individually

```bash
bash diagnose.sh              # read-only system check
bash install.sh --preflight   # version snapshot only
bash install.sh               # full install
bash install.sh --check       # verify existing installation
```

## Output

Each run writes artifacts to:

```text
install_runs/<run_id>/
  install_tester.log      # orchestrator, diagnose, install, and check output
  machine_versions.txt    # system/tool/Python package versions
```

## Step 3: Benchmark

After a successful install, run the benchmark suite:

```bash
bash benchmark.sh [machine_label]
```

This delegates to the repository-root benchmark runner. By default it runs:
- **Example 0** (curvilinear validation) — single process, cold run.
- **Example 1 smoke** (Blasius resolvent) — two MPI variants:
  - `all_cores`: 1 thread per rank, ranks = all physical cores.
  - `half_cores`: 1 thread per rank, ranks = half physical cores.

Physical cores are auto-detected via `lscpu`. Cache files (`.npz`) are deleted
before each run to ensure cold timings. Thread environment is locked to 1
(`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).

### Benchmark output

```text
benchmark_runs/<bench_id>/
  benchmark_results.json    # structured per-case results
  benchmark_summary.csv     # one-line-per-case for quick comparison
  example_*_<variant>.log   # full stdout/stderr per case
  example_*_<variant>.time  # /usr/bin/time -v output (peak RSS, etc.)
```

## Comparing machines

**Installation:**
1. Run `run_install_tester.sh` on each target machine.
2. Collect all `install_runs/*/` folders.
3. Compare:
   - `install_tester.log` for pass/fail and elapsed time.
   - `machine_versions.txt` for version deltas across machines.
   - `install_tester.log` around installer error markers for failure details.

**Benchmarks:**
1. Run `benchmark.sh` on each target machine.
2. Collect all `benchmark_runs/*/` folders.
3. Compare `benchmark_summary.csv` across machines for wall time and peak memory.

## Fast Blasius smoke (slow nodes, e.g. stab1)

After a successful install, the full Example 1 grid
(`examples/example_1_blasius.py` / `example_1_blasius.jsonc`) can take a very
long time on some machines. For a quick PETSc / MPI / assembly check without
overwriting full Example 1 outputs, run from the repository root:

```bash
cd /path/to/linstabpy
python -m examples.example_1_blasius_smoke
# optional MPI, adjust rank count as needed:
mpiexec.mpich -n 1 python -m examples.example_1_blasius_smoke
```

This uses `examples/example_1_blasius_smoke.jsonc` (small `nx` / `ny`) and writes only `example_1_blasius_smoke.*` artifacts. Delete `examples/example_1_blasius_smoke.npz` to force a cold recompute. The smoke config sets `assembly_backend` to **`scipy`** so LNSE assembly stays cheap on tiny grids (direct PETSc assembly uses many small insertions and can look “stuck” even at 64×24). The resolvent step still runs through PETSc/MUMPS. To benchmark the same direct-PETSc assembly path as full Example 1 (`auto` / `petsc`), edit that field in the JSONC and expect much higher `build_lnse_operator` wall time on small problems.

**MPICH:** `mpiexec -x VAR=value` is an Open MPI convention. On MPICH, `export VAR=...` in the shell before `mpiexec`, or pass env with MPICH’s `-genv VAR value` (see your `mpiexec` man page).

## Notes

- `LINSTAB2D_INSTALL_RUN_ID` can be set to override the auto-generated run id.
- `LINSTAB2D_INSTALL_LOG_DIR` overrides the default log directory (`install_runs/`).
- `LINSTAB2D_REPO_ROOT` points the harness at a checkout outside the parent directory.
- `LINSTAB2D_SKIP_SYS_DEPS=1` skips system dependency checks (pre-provisioned machines only).
- Progress tracking is at the repository root in `PROGRESS.md`.
