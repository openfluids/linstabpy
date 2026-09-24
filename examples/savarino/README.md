# Savarino et al. SWBLI resolvent case

## What this reproduces

This directory reproduces the resolvent-analysis validation and speed
figures against the published Savarino et al. shock-wave/boundary-layer
interaction (SWBLI) resolvent dataset, at the 30.8 deg oblique-shock
interaction angle: the manuscript supplement figures S4-S6 (response-mode
comparisons at the lift-up and first-Mack points, and the 200-point gain
map) and the accompanying tables. No new physics is introduced here -- this
is a validation-and-speed reproduction against a published reference, not a
new result.

## Layout

```
examples/savarino/
  configs/                  # point YAMLs, grids, campaign template
  reference/
    baseflow/               # Savarino Tecplot .dat (fetched, not tracked)
    savarino_response_*.npz # built by pack_reference_modes.py (not tracked)
    savarino_grid.csv
    savarino_forcing_mask_measured.csv
  output/
    savarino_map2_200pt.csv # superseded 200-pt campaign, not quoted
    savarino_map317pt.csv   # committed 317-pt campaign run record
    baseflow/               # converted .npy (savarino_to_npy.py)
    compare/                # the two figure points, trimmed (run_compare_points.py rewrites them in full)
    figures/                # plots (compare_modes.py, plot_gain_map.py)
  savarino_map3_200pt.csv        # 200-pt map of record, per-point jobs
  savarino_map3_lines_200pt.csv  # the same 200 points from the line driver
  savarino_cudss_map_200pt.csv   # the same 200 points on the cuDSS GPU backend (gains identical)
  savarino_cudss_farm_fp64.csv   # beta=40/45 subset (47 pts), cuDSS double, per-point jobs
  savarino_cudss_lines_fp64.csv  # the same 47 points from the line driver on cuDSS
  savarino_cudss_farm_fp32.csv   # the same 47 points, single-precision factor; 21 points
                                 #   fail (status != ok) -- kept as the FP32 record
  savarino_gain_map.png          # S6 rendered from savarino_map3_200pt.csv
  savarino_point_gpu.yaml        # GPU (STRUMPACK + FGMRES) variant of configs/mack_f2_b45.yaml
  savarino_line.py, savarino_mapwalk.py, gen_line_campaign.py, gen_mapwalk_campaign.py,
  savarino_line.sbatch.template, savarino_mapwalk.sbatch.template, compare_map_timings.py
```

## Reference data (`reference/`)

| Path | Contents |
|------|----------|
| `savarino_map3_200pt.csv` | Our 200-point gain-map run record (map of record, `plot_gain_map.py` default). |
| `savarino_map3_lines_200pt.csv` | The same 200 points computed by the line driver (in-place refactorization); `rel_err` compares against the farm gains. |
| `savarino_cpu_farm_47pt.csv` / `savarino_cpu_lines_47pt.csv` | Dedicated CPU reference runs of the 47-pt (β=40/45) subset, both execution modes — the CPU side of every GPU comparison. |
| `savarino_strumpack_farm_fp64.csv` / `savarino_strumpack_lines_fp64.csv` | The STRUMPACK+FGMRES GPU campaign on the same 47 points, both modes. |
| `savarino_cudss_*.csv` | The cuDSS GPU campaign: the 200-pt map and the 47-pt (β=40/45) subset in both execution modes reproduce the map-of-record gains to every printed digit; the `fp32` file records the rejected single-precision arm with its failed points. |
| `savarino_mpi_scaling.csv` | MPI strong scaling of the Mack point at 4/8/16 ranks (MUMPS, one idle node). |
| `savarino_campaign_walltimes.csv` | Wall-clock of every campaign arm from Slurm accounting (first job start to last job end), with the Slurm job ranges. |
| `benchmarks/` | Controlled single-stage backend measurements (factor times, FP32 behaviour, tolerance ladder) with their harness scripts — see `benchmarks/README.md`. |
| `output/savarino_map2_200pt.csv` | Earlier 200-point campaign, superseded (see step 6). |
| `reference/baseflow/` | Savarino Tecplot mean-flow `.dat` — downloaded by `fetch_zenodo.py baseflow`, not tracked. |
| `reference/savarino_response_*.npz` | Savarino resolvent response (Mack + lift-up), built locally from the Zenodo modes by `pack_reference_modes.py`; not tracked. |
| `reference/savarino_grid.csv` | Published sample-point grid (metadata). |
| `reference/savarino_forcing_mask_measured.csv` | Forcing-region bounds (metadata). |
| `output/figures/` | Plots from `compare_modes.py` / `plot_gain_map.py`. |
| `output/compare/` | The two points behind the mode figures, kept to what `compare_modes.py` reads: grid, Chu weight, leading response, index maps, gain. Single precision, 5.7 MB each instead of 22 MB; the figure is unchanged and the printed Chu correlation moves by 1e-10. `run_compare_points.py` overwrites them with the full bundles. |

See `reference/README.md` for details.

## How each run record was produced

All compute jobs ran through Slurm on one node (dual Xeon Silver 4210R, 20
cores, MKL BLAS, `I_MPI_PIN=off`; GPU rows add one NVIDIA RTX A6000 per job).
Each CSV row carries its own commit, machine, ranks, and Slurm job id. The
commands, from this directory:

| Record | Command |
|--------|---------|
| `savarino_map3_200pt.csv` | `python gen_campaign.py --config configs/map_base.yaml --nranks 4` (200 jobs, 5 concurrent) |
| `savarino_map3_lines_200pt.csv` | `python gen_mapwalk_campaign.py --config configs/map_base.yaml --nranks 4` per β line (14 line jobs, 5 concurrent) |
| `savarino_cpu_farm_47pt.csv` | `gen_campaign.py --beta 40 45 --config configs/map_base.yaml --nranks 4` |
| `savarino_cpu_lines_47pt.csv` | `gen_mapwalk_campaign.py --beta 40` and `--beta 45`, same config |
| `savarino_strumpack_*.csv` | same two generators with `--config savarino_point_gpu.yaml --nranks 1` (one A6000 per job) |
| `savarino_cudss_*.csv` | same two generators with `--config savarino_point_cudss.yaml` (fp64) or `savarino_point_cudss32.yaml` (fp32), `--nranks 1` |
| `savarino_mpi_scaling.csv` | the (2,45) per-point config from `gen_campaign.py`, run serially with `mpiexec -n {4,8,16}` on an idle node |
| `benchmarks/*.csv` | the `benchmarks/bench_*.py` harnesses on operator dumps from `benchmarks/dump_operator.py` |

To compare the leading response modes of two runs of the same point (for
example a GPU backend against the MUMPS reference):

```
python compare_backend_modes.py reference.npz other.npz
```

## Configs (`configs/`)

| File | Use |
|------|-----|
| `map_base.yaml` | Template for `gen_campaign.py` (200- and 317-pt maps). |
| `mack_f2_b45.yaml` | Mack comparison point (f=2, β=45); base config of every campaign generator. |
| `liftup_f1em4_b163.yaml` | Lift-up comparison point (f=1e-4, β=163). |
| `point.sbatch.template` | Slurm batch template for map campaigns. |
| `savarino_map317_points.csv` | 317-pt (β, f) grid for `--grid dense317`. |

`savarino_point_gpu.yaml` (example root) is the same Mack point with the
GPU-capable solver block (FGMRES right-preconditioned by a STRUMPACK LU).
`savarino_point_cudss.yaml` and `savarino_point_cudss32.yaml` (example root)
are the same point on the cuDSS GPU backend (FGMRES right-preconditioned by a
cuDSS LU; the `32` variant factors in single precision via `factor_precision:
single`). Both run single-rank on one GPU -- see the "cuDSS GPU Backend"
section of `docs/COMPUTE_BACKENDS.md`.

## Data

Zenodo DOI 10.5281/zenodo.18854407. The record is open access but states no
redistribution license, so the repository ships no file from it. Download the
**base-flow Tecplot file** first:

```
python fetch_zenodo.py baseflow
```

The mode comparison (step 4) also needs the two published response modes.
Fetch them and pack them into `reference/savarino_response_*.npz`:

```
python fetch_zenodo.py modes --f 2 --beta 45
python fetch_zenodo.py modes --f 0.0001 --beta 163
python pack_reference_modes.py
```

## Workflow

1. **Fetch and convert** the base flow to linstabpy `.npy` fields:
   ```
   python fetch_zenodo.py baseflow
   python savarino_to_npy.py
   ```
   Downloads `reference/baseflow/state_atcenter_....dat` (md5-checked), then
   converts it to `output/baseflow/`.
   This asserts the file's md5, parses the Tecplot ASCII grid, converts
   conservative to primitive variables, and checks the freestream plateau
   (rho = u = T = 1) before writing anything.
2. **Run the two comparison points** (rewrites `output/compare/`):
   ```
   python run_compare_points.py
   ```
   The repository already carries both points in trimmed form, so step 4
   needs only the Zenodo modes (see Data), not this step.
   A single point by hand, from this directory:
   ```
   LINSTABPY_CASE1_CONFIG=configs/mack_f2_b45.yaml \
   LINSTABPY_CASE1_OUTPUT_NPZ=sav_map_b45p0_f2.npz \
   python -c "from linstabpy.driver.run import run; run()"
   ```
3. **Verify the norm convention** at (f, beta) = (2, 45):
   ```
   python verify_norm_convention.py
   ```
4. **Plot mode comparisons** (S4/S5, Savarino2026 layout):
   ```
   python compare_modes.py
   ```
5. **Run the (beta, f) map.** Two execution modes, same result:
   `gen_campaign.py` submits one job per point (assemble, factor, SLEPc
   gain, from scratch); `gen_mapwalk_campaign.py --n-anchor 99` submits one
   job per beta line (assemble once, symbolic analysis once, numeric
   refactorization in place at every frequency, SLEPc gain). The line
   walker's preconditioned path (`--n-anchor 2`, the `LineWalker` default:
   later frequencies by GMRES with the live factor, taken only when a probe
   beats a fresh factor) never wins on this operator -- GMRES needs 11-32
   iterations at the map's frequency spacings against a break-even below
   11 -- so the map of record was run with every point factorized.
   `savarino_line.py` is a **1-D frequency-line** tool (RSVD / recycle) for
   a single beta, not the published 2-D map.
   ```
   python gen_campaign.py --out campaign --local          # per-point jobs, 200-pt grid
   python gen_campaign.py --out campaign --grid dense317  # per-point jobs, 317-pt grid
   python gen_mapwalk_campaign.py --out lines --beta 45 --n-anchor 99 --driver-out lines/b45
   ```
   Both generators take `--config`, `--cpus-per-task` and `--gpus` for the
   GPU backends: `savarino_point_gpu.yaml` (STRUMPACK LU; same gains, slower
   than the CPU reference on a workstation-class card) and
   `savarino_point_cudss.yaml` / `savarino_point_cudss32.yaml` (cuDSS LU;
   same gains, and on the same card faster than the CPU reference --
   measured numbers in `docs/COMPUTE_BACKENDS.md`).
6. **Collect** the per-point outputs into one run-record CSV:
   ```
   python collect_campaign.py campaign > campaign.csv
   ```
   For an existing NPZ tree (e.g. Savarino2026 dense map, 317 points):
   ```
   python collect_campaign.py --from-npz /path/to/resolvent_map_repo_dense \\
       --points-csv configs/savarino_map317_points.csv \\
       --out output/savarino_map317pt.csv merged
   ```
   The committed `savarino_map3_200pt.csv` is this file from the 200-point
   run of record (per-point jobs); `savarino_map3_lines_200pt.csv` is the
   same 200 points from the line driver (the 14 per-line
   `savarino_mapwalk.csv` files concatenated), agreeing to 4.7e-10.
   `output/savarino_map2_200pt.csv` is the earlier campaign; the line driver's
   `farm_gain` / `rel_err` columns compared against it before 1.0.0rc10 and
   now compare against `savarino_map3_200pt.csv`. Its gains predate the
   filter-placement fix (`6b3ae2a`, <= 4.4e-3 relative) and its wall times
   were measured with reference BLAS, so neither is quoted.
   `output/savarino_map317pt.csv` is the 317-point dense-grid record.
7. **Plot the gain map** (S6, Savarino2026 style; default input
   `savarino_map3_200pt.csv`, default output `output/figures/savarino_gain_map.png`):
   ```
   python plot_gain_map.py
   ```
   For a densified map with sample overlays:
   ```
   python plot_gain_map.py --summary path/to/summary.csv --show-points \\
       --baseline-summary savarino_map3_200pt.csv
   ```

`mode_oracle.py` is a separate diagnostic: it injects the published forcing
directly into the assembled operator and compares the response against the
published mode, independent of the gain and the norm convention (see the
script's own docstring for what it does and does not test).

## The norm convention

The published figure's forcing is unit-norm in a mass-weighted L2 norm over
the conservative momentum components (`rou`, `rov`, `row`); the response
norm is the Chu energy with integration weights (the same norm the
`compute_chu_norm` code path builds). The colorbar label :math:`\sigma_1^2` with :math:`\times 10^4` at the top
means the displayed quantity is :math:`\sigma_1^2 / 10^4`. This was identified, not
assumed -- see `verify_norm_convention.py` for the systematic comparison
across candidate norm choices and the resulting agreement (~0.4% at the
published (2, 45) point).

## Declared settings

- Finite-difference order: fd4.
- Stride: 1 (full resolution, both directions).
- Operator dissipation: epsilon = 0 (rectilinear mesh; the metric
  cross-coupling that requires dissipation on curvilinear meshes is
  negligible here -- see `savarino_to_npy.py` for the mesh construction).
- No sponge layer (matching the reference's treatment).
- Outflow boundary condition: Neumann, versus the reference's supersonic
  extrapolation. This is a declared difference in numerical method, not an
  attempt to match the reference's exact discretization.

## Run record

`savarino_map3_200pt.csv` is the committed 200-point gain map (per-point
jobs): one row per (beta, f) point, with gain, residuals, wall time,
memory, DOF, nnz, and full provenance (git SHA, machine, ranks, backend,
epsilon, forcing region) so each row is self-describing outside the
repository. `savarino_map3_lines_200pt.csv` is the line-driver run of the
same 200 points (gain, per-point factor and solve time, factor count,
memory, ranks, BLAS, `ncv`, git SHA, job). `output/savarino_map2_200pt.csv` is
the superseded earlier campaign (see step 6); `output/savarino_map317pt.csv`
is the 317-point dense-grid record.

The per-point field data (mode shapes, full solution vectors) behind these
rows is not committed here -- it is published as a separate dataset; see
the repository release notes.
