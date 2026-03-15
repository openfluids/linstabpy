# Song2023 MPI + Order Scaling Plan

## Goal
Benchmark the corrected Song2023 case across:
- finite-difference order: `2`, `4`, `6`
- MPI ranks: `1`, `2`, `4`, `8`

and compare:
- time to solution
- memory footprint
- per-step timings
- spectrum quality versus MATLAB

This must use the real distributed PETSc path, not the current local-only Song matrix runner.

## Important Current State
- The current canonical Song run is **serial**.
- The current `validation/song_2023/song_2023_matrix.jsonc` variants force:
  - `"petsc_matrix_mode": "local"`
- The current matrix runner launches:
  - plain `python -m validation.song_2023.song_2023`
  - not `mpiexec`

So:
- current timing/memory numbers are useful as serial baselines only
- they are **not** MPI scaling evidence

## What Must Be True For This Study
1. Song must run with `mpiexec -n {1,2,4,8}`.
2. Song must request:
   - `"assembly_backend": "petsc"`
   - `"solver": { "petsc_matrix_mode": "distributed", ... }`
3. The run log must prove:
   - `mpi_ranks`
   - resolved `petsc_matrix_mode`
   - `distributed_matrix=True`
   - PETSc matrix type
   - ownership ranges

If any run says otherwise, it does not count as a distributed benchmark.

## Benchmark Definitions

### Time to solution
Use wall-clock time from the slowest rank:
- `total_time_s_max_rank`

### Memory footprint
Do not report rank-0 RSS only.
For each run record:
- `peak_memory_mib_max_rank`
- `peak_memory_mib_sum_ranks`

The first says worst per-rank requirement.
The second says approximate total process memory footprint across MPI ranks.

### Per-step timings
For each stage, record across ranks:
- `time_s_min`
- `time_s_max`
- `time_s_mean`

Primary reported stage time should be:
- `time_s_max`

because that is the real wall-clock stage cost.

### Spectrum quality
For each run compare against MATLAB reference:
- `mean_abs`
- `max_abs`
- `mean_nearest_matlab`
- leading eigenvalue

Keep the current corrected MATLAB-matching procedure:
- transposed Song input
- filtered default branch
- `stability_search_extra_modes = 4`

## Execution Matrix

### Full target matrix
- order 2, ranks 1
- order 2, ranks 2
- order 2, ranks 4
- order 2, ranks 8
- order 4, ranks 1
- order 4, ranks 2
- order 4, ranks 4
- order 4, ranks 8
- order 6, ranks 1
- order 6, ranks 2
- order 6, ranks 4
- order 6, ranks 8

Total:
- `12` full runs

## Why A Staged Rollout Is Still Needed
Even though the target matrix is `12` runs, we should not jump straight into all of them blindly.

Reason:
- `fd_order=4` and `fd_order=6` can change stencil width, memory pressure, and factorization behavior
- the current full Song run is already expensive
- MPI logging still needs to be upgraded from serial-style rank-0 stats to real cross-rank summaries

## Recommended Execution Stages

### Stage 0: MPI instrumentation hardening
Before benchmarking, update Song logging so each run writes:
- `mpi_ranks`
- `petsc_matrix_mode`
- `distributed_matrix`
- `matrix_type`
- `ownership_ranges`
- total time: max across ranks
- peak memory: max and sum across ranks
- per-step time min/max/mean across ranks
- per-step memory max across ranks

Acceptance:
- one 2-rank Song run writes all of the above cleanly

### Stage 1: reduced MPI feasibility sweep
Run reduced Song first for:
- order 2, ranks 1/2/4/8
- order 4, ranks 1/2/4/8
- order 6, ranks 1/2/4/8

using:
- reduced stride settings
- distributed PETSc
- PETSc assembly
- MUMPS

Acceptance:
- all 12 reduced runs complete
- logs prove distributed mode for ranks > 1
- no order/rank combination fails at setup or factorization stage

### Stage 2: full baseline ladder
Run full Song for:
- order 2, rank 1
- order 4, rank 1
- order 6, rank 1

Acceptance:
- full runs complete
- MATLAB comparison remains acceptable
- 4th/6th order cost is quantified before scaling out

### Stage 3: full MPI scaling
Run full Song for:
- order 2, ranks 2/4/8
- order 4, ranks 2/4/8
- order 6, ranks 2/4/8

Acceptance:
- all runs complete
- distributed logs are explicit
- scaling table can be assembled without ambiguity

## Expected Code Changes

### 1. New JSONC benchmark config
Add a dedicated Song benchmark matrix config, for example:
- `validation/song_2023/song_2023_mpi_scaling.jsonc`

This should define:
- order list `[2,4,6]`
- rank list `[1,2,4,8]`
- distributed PETSc mode
- output root
- reduced/full toggles

### 2. New benchmark runner
Add a dedicated runner, for example:
- `validation/song_2023/run_song_2023_mpi_scaling.py`

Responsibilities:
- generate per-run variant JSONC
- launch `mpiexec -n N`
- capture stdout/stderr
- move `.npz/.log/.png`
- extract benchmark metrics into JSON/CSV

Do not overload the current `run_song_2023_matrix.py` with MPI benchmarking concerns unless the interface stays clear.

### 3. Cross-rank instrumentation in `song_2023.py`
The current Song instrumentation already records:
- total time
- peak memory
- per-step timings

But it is rank-local.

Upgrade it so:
- total time is reduced with `max`
- per-step times are gathered/reduced across ranks
- per-rank RSS is gathered

### 4. Optional spectrum-overlay extension
Once all runs exist, extend the Song spectrum figure so it can overlay:
- Python 2nd order
- Python 4th order
- Python 6th order

This should be JSONC-driven and should use the new benchmark outputs as inputs.

## Reporting Outputs

### Benchmark table
One CSV/JSON row per run with:
- order
- ranks
- distributed yes/no
- total time
- peak memory max-rank
- peak memory sum-ranks
- assemble time max-rank
- eigensolve time max-rank
- save time max-rank
- `mean_abs`
- `max_abs`
- `mean_nearest_matlab`
- leading eigenvalue

### Scaling metrics
Compute for each order:
- speedup vs rank-1
- parallel efficiency
- memory per rank trend

### Figure update
Spectrum panel should eventually show:
- MATLAB
- Song original
- Python 2nd order
- Python 4th order
- Python 6th order

Mode panels can remain tied to one chosen reference run unless we decide to add a second figure for order comparison.

## Risks / Failure Modes

### 1. 8-rank local oversubscription
If the current machine does not have enough physical cores or memory, 8-rank timing may be noisy or misleading.

Mitigation:
- still run it if requested, but label it as workstation scaling, not cluster scaling

### 2. MUMPS memory blow-up at higher order
Higher FD order may widen the operator stencil and LU fill-in.

Mitigation:
- reduced sweep first
- stop and reassess if full `fd_order=6` rank-1 is already beyond reasonable memory budget

### 3. Local-vs-distributed confusion
This is the main trap.

Mitigation:
- make distributed mode explicit in JSONC
- write proof fields into the log
- reject runs that do not prove distributed mode

## Recommended Immediate Next Step
1. harden `song_2023.py` instrumentation for real MPI summaries
2. add a dedicated MPI scaling JSONC + runner
3. run the reduced `12`-case feasibility matrix first

Only after that should we launch the full `12`-run order/rank study.
