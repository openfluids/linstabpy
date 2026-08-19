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

## Data

Reference dataset: Zenodo DOI 10.5281/zenodo.18854407 (base flow, published
resolvent forcing/response modes, and eigenmodes).

`fetch_zenodo.py` downloads what is needed:

- `fetch_zenodo.py baseflow` gets the base-flow file
  (`state_atcenter_mesh300_y300_xend195_angle308_T157_Mach215.dat`).
- `fetch_zenodo.py modes --f F --beta B` gets the four mode files
  (forcing/response, real/imaginary) for one (f, beta) point, extracted
  from the resolvent-modes zip via HTTP range requests (`pip install
  remotezip`), without downloading the whole archive.

Everything lands under `$SAVARINO_DATA` (default: `data/` next to this
script), with mode files under `data/modes/`.

## Workflow

1. **Fetch** the base flow and the mode pairs you need:
   ```
   python fetch_zenodo.py baseflow
   python fetch_zenodo.py modes --f 2 --beta 45
   python fetch_zenodo.py modes --f 0.0001 --beta 163
   ```
2. **Convert** the base flow to linstabpy `.npy` fields:
   ```
   python savarino_to_npy.py data/state_atcenter_mesh300_y300_xend195_angle308_T157_Mach215.dat data/converted
   ```
   This asserts the file's md5, parses the Tecplot ASCII grid, converts
   conservative to primitive variables, and checks the freestream plateau
   (rho = u = T = 1) before writing anything.
3. **Verify the norm convention** at the published (f, beta) = (2, 45)
   point (requires a single-point run, next step, first):
   ```
   python verify_norm_convention.py --point-npz results/campaign/sav_map_b45p0_f2/sav_map_b45p0_f2.npz
   ```
4. **Run a single point** from this directory (edit `paths.case_dir` in
   `savarino_point_f2_b45.yaml` to the converted base-flow directory first):
   ```
   LINSTABPY_CASE1_CONFIG=savarino_point_f2_b45.yaml \
   LINSTABPY_CASE1_OUTPUT_NPZ=sav_map_b45p0_f2.npz \
   python -c "from linstabpy.driver.run import run; run()"
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
   python gen_campaign.py --out campaign --local          # per-point jobs
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
   The committed `savarino_map3_200pt.csv` is this file from the 200-point
   run of record (per-point jobs); `savarino_map3_lines_200pt.csv` is the
   same 200 points from the line driver (the 14 per-line
   `savarino_mapwalk.csv` files concatenated), agreeing to 4.7e-10.
   `savarino_map2_200pt.csv` is the earlier campaign the line driver's
   `farm_gain` / `rel_err` columns compare against; its gains predate the
   filter-placement fix (`6b3ae2a`, <= 4.4e-3 relative) and its wall times
   were measured with reference BLAS, so neither is quoted.
7. **Make the figures**:
   ```
   python plot_gain_map.py savarino_map3_200pt.csv savarino_gain_map.png
   python compare_modes.py --campaign-dir campaign --out-dir .
   ```

`mode_oracle.py` is a separate diagnostic: it injects the published forcing
directly into the assembled operator and compares the response against the
published mode, independent of the gain and the norm convention (see the
script's own docstring for what it does and does not test).

## The norm convention

The published figure's forcing is unit-norm in a mass-weighted L2 norm over
the conservative momentum components (`rou`, `rov`, `row`); the response
norm is the Chu energy with integration weights (the same norm the
`compute_chu_norm` code path builds). The colorbar label "sigma^2 x 10^4"
means the displayed quantity is sigma^2 / 1e4. This was identified, not
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
memory, ranks, BLAS, `ncv`, git SHA, job). `savarino_map2_200pt.csv` is
the superseded earlier campaign (see step 6).
`results/savarino308_backend_parity_2_45.csv` is a CPU/GPU backend parity
check at the (2, 45) point.

The per-point field data (mode shapes, full solution vectors) behind these
rows is not committed here -- it is published as a separate dataset; see
the repository release notes.
