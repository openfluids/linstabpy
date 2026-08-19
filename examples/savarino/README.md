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
    baseflow/               # Savarino Tecplot .dat
    savarino_response_*.npz
    savarino_grid.csv
    savarino_forcing_mask_measured.csv
  output/
    savarino_map2_200pt.csv # superseded 200-pt campaign (line driver's rel_err reference)
    savarino_map317pt.csv   # committed 317-pt campaign run record
    baseflow/               # converted .npy (savarino_to_npy.py)
    compare/                # resolvent .npz (run_compare_points.py)
    figures/                # plots (compare_modes.py, plot_gain_map.py)
  savarino_map3_200pt.csv        # 200-pt map of record, per-point jobs
  savarino_map3_lines_200pt.csv  # the same 200 points from the line driver
  savarino_gain_map.png          # S6 rendered from savarino_map3_200pt.csv
  savarino_point_gpu.yaml        # GPU (STRUMPACK + FGMRES) variant of configs/mack_f2_b45.yaml
  savarino_line.py, savarino_mapwalk.py, gen_line_campaign.py, gen_mapwalk_campaign.py,
  savarino_line.sbatch.template, savarino_mapwalk.sbatch.template, compare_map_timings.py
```

## Reference data (`reference/`)

| Path | Contents |
|------|----------|
| `savarino_map3_200pt.csv` | Our 200-point gain-map run record (map of record, `plot_gain_map.py` default). |
| `output/savarino_map2_200pt.csv` | Earlier 200-point campaign, superseded (see step 6). |
| `reference/baseflow/` | Savarino Tecplot mean-flow `.dat` (published). |
| `reference/savarino_response_*.npz` | Savarino resolvent response (Mack + lift-up). |
| `reference/savarino_grid.csv` | Published sample-point grid (metadata). |
| `reference/savarino_forcing_mask_measured.csv` | Forcing-region bounds (metadata). |
| `output/figures/` | Plots from `compare_modes.py` / `plot_gain_map.py`. |

See `reference/README.md` for details.

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

## Data

Zenodo DOI 10.5281/zenodo.18854407. The **base-flow Tecplot file** is shipped
under `reference/baseflow/`. Use `fetch_zenodo.py modes` only when regenerating
`savarino_response_*.npz` from the Zenodo mode archive.

## Workflow

1. **Convert** the base flow to linstabpy `.npy` fields:
   ```
   python savarino_to_npy.py
   ```
   Reads `reference/baseflow/state_atcenter_....dat` → `output/baseflow/`.
   This asserts the file's md5, parses the Tecplot ASCII grid, converts
   conservative to primitive variables, and checks the freestream plateau
   (rho = u = T = 1) before writing anything.
2. **Run the two comparison points** (writes `output/compare/`):
   ```
   python run_compare_points.py
   ```
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
   GPU-capable backend (`savarino_point_gpu.yaml`); on a workstation-class
   card that path returns the same gains and is slower (see the cost
   section of the supplementary material).
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
   `output/savarino_map2_200pt.csv` is the earlier campaign the line driver's
   `farm_gain` / `rel_err` columns compare against; its gains predate the
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
