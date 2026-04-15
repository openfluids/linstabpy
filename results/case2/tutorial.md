# Case 2 — Hands-on tutorial

This is the "I just cloned the repo, how do I actually run something" guide for
`results/case2/`. Start here before reading `README.md`, which covers the
campaign *strategy* but assumes you already know the mechanics.

All commands below are run from the repo root (`linstabpy/`) unless stated.
All Python invocations use the patched venv — **never** use `uv run` (see
`CLAUDE.md`).

---

## 1. What Case 2 actually does

`case2.py` is a thin wrapper around `linstab2d.results_runtime` that:

1. Loads a JSONC config (physics, mesh, solver settings).
2. Builds the LNSE operator around the baseflow in `results/case2/baseflow/`.
3. Runs a temporal EVP (SLEPc shift-invert + MUMPS direct LU) and/or a
   resolvent analysis.
4. Saves results to a `.npz` + a config snapshot.

The entry point (`case2.py:33-40`) is cache-aware: if the `.npz` already
exists, it skips the compute and goes straight to plotting. To force recompute,
delete the `.npz`.

---

## 2. Files in this directory

Active path:

| File                                | Purpose                                                |
|-------------------------------------|--------------------------------------------------------|
| `case2.py`                          | Single-run entrypoint.                                 |
| `parity_config.jsonc`               | Canonical full-resolution Case 2 config.               |
| `evp_stride4_localize.jsonc`        | Campaign config: stride-4 branch localization.         |
| `evp_stride2_stationary_probe.jsonc`| Campaign config: stride-2 stationary probe.            |
| `perf_memory.py`                    | Campaign harness (runs a matrix of cases).             |
| `run_perf_queue_once.sh`            | Queue wrapper around `perf_memory.py`.                 |
| `socket0.rankfile` / `socket1.rankfile` | MPI pinning — 10 ranks on one physical socket.     |
| `plot_saved_spectra.py`             | Overlay spectra from a campaign's `runs/`.             |
| `plot_saved_modes.py`               | Render all saved eigenmodes as PNGs.                   |
| `baseflow/`                         | Input baseflow (git submodule).                        |
| `output/<campaign>/`                | Per-campaign `runs/`, `logs/`, `plots/`.               |
| `legacy/`                           | Frozen historical configs and scripts — ignore.        |

---

## 3. The fastest possible run (serial, 1 rank)

Good first sanity check — does the whole pipeline execute at all.

```bash
LINSTAB2D_CASE2_CONFIG=results/case2/parity_config.jsonc \
LINSTAB2D_CASE2_OUTPUT_NPZ=results/case2/output/runs/case2_first_smoke.npz \
.venv/bin/python results/case2/case2.py
```

This runs **serial** (one Python process, no MPI). Slow on the full
resolution, but guaranteed to reveal config errors immediately.

To re-run (same command): the `.npz` exists → it will **skip compute**.
Delete the `.npz` to recompute.

---

## 4. See what it is doing (observability)

The default output only prints at stage boundaries. If a stage takes 10 min
you see nothing in between. Three env vars fix this:

```bash
export LINSTAB2D_STAGE_HEARTBEAT_SECONDS=5   # "still alive" line every 5 s
export PYTHONUNBUFFERED=1                    # flush stdout immediately
export PETSC_OPTIONS="-eps_monitor -log_view"
```

What you will see:

```
stage begin: load_case
stage complete: stage=load_case elapsed=6.42s
stage begin: assemble_operator_bc
stage heartbeat: stage=assemble_operator_bc elapsed=5.01s
stage heartbeat: stage=assemble_operator_bc elapsed=10.02s
stage complete: stage=assemble_operator_bc elapsed=14.37s
stage begin: stability
  1 EPS nconv=0 first unconverged value ... residual norm 4.3e-02
  2 EPS nconv=3 first unconverged value ... residual norm 7.1e-04
  3 EPS nconv=5 first unconverged value ... residual norm 2.8e-09
stage complete: stage=stability elapsed=312.55s
```

Now you know whether it is in assembly, MUMPS factorization, or the Arnoldi
loop. At the end, `-log_view` prints a full timing table per PETSc operation —
use this to identify the real bottleneck.

Alternative to the env var: set it once in the JSONC:

```jsonc
"logging": { "stage_heartbeat_seconds": 5 }
```

Always pipe the run to a log file:

```bash
... .venv/bin/python results/case2/case2.py 2>&1 | tee run.log
```

---

## 5. Running with MPI (the right way)

### 5.1 The trap you hit: 10 duplicate serial runs

If you just do:

```bash
mpirun -n 10 python results/case2/case2.py     # WRONG in several ways
```

common failure modes:

1. **`python` resolves to the system Python** (not the patched venv) → import
   errors or 10 serial runs that each fail.
2. **`mpirun` from a different MPI than `mpi4py`/PETSc was built against** →
   each rank runs independently with `COMM_WORLD` size 1, so you get N
   duplicate serial runs that all write to the same `.npz` and race.
3. **No thread pinning** → every rank spawns `nproc` OpenBLAS threads,
   oversubscribing the box.

### 5.2 The right command

The repo ships a rankfile (`socket0.rankfile`) that pins 10 ranks to one
physical socket. Use it:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
LINSTAB2D_STAGE_HEARTBEAT_SECONDS=5 \
PYTHONUNBUFFERED=1 \
PETSC_OPTIONS="-eps_monitor -log_view" \
mpiexec --rankfile results/case2/socket0.rankfile --bind-to core \
  .venv/bin/python results/case2/case2.py 2>&1 | tee run.log
```

Key points:

- `mpiexec` (not `mpirun`) — on Linux MPICH this is `/usr/bin/mpiexec.mpich`;
  on Open MPI `/usr/bin/mpiexec.openmpi`. Both work as long as it matches the
  one `mpi4py`/PETSc were built against. `install.sh` prints the detected
  launcher at the end; use that one.
- `.venv/bin/python` — the absolute path to the patched interpreter.
- `--rankfile` — stable CPU placement across runs (reproducible timing).
- One thread per rank — the four env vars above prevent OpenBLAS/MKL from
  spawning extra threads per rank.

### 5.3 How to confirm it really ran in parallel

Three checks:

1. Log header: `rank=0 size=10` (not `size=1`).
2. One NPZ written, not 10 competing.
3. `-log_view` tail prints `Nranks:   10` and shows roughly N× speedup on
   `KSPSolve` / `MatMult` relative to serial.

---

## 6. Changing the case (JSONC knobs)

All physics, mesh, and solver settings live in the JSONC config. For a
single-run experiment, copy `parity_config.jsonc` and edit the copy:

```bash
cp results/case2/parity_config.jsonc results/case2/my_experiment.jsonc
# edit my_experiment.jsonc
LINSTAB2D_CASE2_CONFIG=results/case2/my_experiment.jsonc \
LINSTAB2D_CASE2_OUTPUT_NPZ=results/case2/output/runs/my_experiment.npz \
.venv/bin/python results/case2/case2.py
```

The knobs you will touch most often, with typical locations in the JSONC:

- **Physics** (`physics.*`): `Re`, `Ma`, `Pr`, `kappa`, `T_0`.
- **Mesh decimation** (`numerics.stride_ksi`, `numerics.stride_eta`):
  `4` is cheap, `2` is medium, `1` is full resolution.
- **Finite-difference order** (`numerics.fd_order`): `4` is the standard.
- **Filter** (`numerics.alpha_filter`): `"none"` for EVP work; don't filter
  unless you have a specific reason.
- **What to run** (`runtime.run_stability`, `runtime.run_resolvent`).
- **EVP target** (`stability.target_real`, `stability.target_imag`,
  `stability.nev`, `stability.ncv`, `stability.eps_tol`).
- **Solver backend** (`solver.assembly_backend`): `"petsc"` for MPI runs,
  `"scipy"` for serial debugging.
- **Matrix distribution** (`solver.petsc_matrix_mode`): `"distributed"` for
  real parallelism, `"local"` to replicate on every rank (debugging only).
- **Linear solve** (`solver.linsolve.type`): `"mumps"` (default, direct LU)
  or `"iterative"` (for very large problems).

---

## 7. Campaign mode — matrix of runs with `perf_memory.py`

When you want to sweep a parameter (e.g. `ncv`) across many runs and collect
timings, use the campaign harness:

```bash
LINSTAB2D_CASE2_PERF_CONFIG=results/case2/evp_stride4_localize.jsonc \
.venv/bin/python results/case2/perf_memory.py
```

The campaign config (`evp_stride4_localize.jsonc`) defines:

- `outputs.*`: where to write per-run NPZ, logs, summary CSV/JSON.
- `defaults.launcher.*`: the exact `mpiexec` command and rank count.
- `defaults.env.*`: the one-thread-per-rank enforcement.
- `defaults.case_defaults.config_overrides.*`: base config values.
- `cases[]`: the list of runs, each overriding a few fields (e.g. different
  `ncv`, different target shift).

You do **not** prepend `mpiexec` yourself here — `perf_memory.py` launches
each case as its own MPI job. It runs them sequentially by design (MUMPS +
EVP memory footprints do not compose).

Outputs:

```
results/case2/output/evp_stride4_localize/
  runs/        one NPZ + .config.json per case
  logs/        one text log per case
  plots/
    spectra/   overlay figures across cases
    modes/<case_name>/    per-mode PNGs
  case2_evp_stride4_localize_latest.json
  case2_evp_stride4_localize_table.csv
```

---

## 8. Looking at results

Overlay all spectra in a campaign:

```bash
.venv/bin/python results/case2/plot_saved_spectra.py \
  --run-dir results/case2/output/evp_stride4_localize/runs
```

Render every accepted mode:

```bash
.venv/bin/python results/case2/plot_saved_modes.py \
  --run-dir results/case2/output/evp_stride4_localize/runs
```

The CSV summary (`..._table.csv`) is the fastest way to compare wall time,
peak RSS, requested vs. converged mode counts, and postcheck residuals across
runs.

---

## 9. Troubleshooting

**The run finishes in 2 s and writes a tiny NPZ.** The cache hit. Delete the
target NPZ and rerun.

**`mpirun -n 10` gives 10 duplicate logs / `COMM_WORLD size=1`.** Wrong
launcher — use `mpiexec` from the same MPI that `mpi4py` was built against.
Check with: `.venv/bin/python -c "import mpi4py; print(mpi4py.get_config())"`.

**No output for minutes.** Heartbeat not enabled or stdout buffered.
`export LINSTAB2D_STAGE_HEARTBEAT_SECONDS=5 PYTHONUNBUFFERED=1`.

**MUMPS fails with out-of-memory.** Drop `stride` one step (4 → 2 → 1), or
switch `solver.linsolve.type` to `"iterative"`.

**Spectrum looks noisy / `nconv < nev`.** Increase `ncv` (try `2*nev`, then
`3*nev`). Tighten `eps_tol` only after `ncv` is settled. See README.md § First
Execution Order for the disciplined order.

**Rankfile hostname mismatch.** `socket0.rankfile` was generated for
`pa-dyf-calc-08.intram.ensam.eu`. On a different host, edit the hostname in
the rankfile or drop `--rankfile` and use `--bind-to core` alone.

---

## 10. Minimum cheat sheet

```bash
# Serial smoke test
.venv/bin/python results/case2/case2.py

# Real run, 10 MPI ranks on one socket, with visibility
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
LINSTAB2D_STAGE_HEARTBEAT_SECONDS=5 PYTHONUNBUFFERED=1 \
PETSC_OPTIONS="-eps_monitor -log_view" \
mpiexec --rankfile results/case2/socket0.rankfile --bind-to core \
  .venv/bin/python results/case2/case2.py 2>&1 | tee run.log

# Campaign
LINSTAB2D_CASE2_PERF_CONFIG=results/case2/evp_stride4_localize.jsonc \
.venv/bin/python results/case2/perf_memory.py

# Force recompute
rm results/case2/output/runs/<your_run>.npz
```
