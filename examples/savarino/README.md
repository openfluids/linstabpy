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
    savarino_map2_200pt.csv # committed 200-pt campaign run record
    savarino_map317pt.csv   # committed 317-pt campaign run record
    baseflow/               # converted .npy (savarino_to_npy.py)
    compare/                # resolvent .npz (run_compare_points.py)
    figures/                # plots (compare_modes.py, plot_gain_map.py)
```

## Reference data (`reference/`)

| Path | Contents |
|------|----------|
| `output/savarino_map2_200pt.csv` | Our 200-point gain-map run record (`plot_gain_map.py`). |
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
| `mack_f2_b45.yaml` | Mack comparison point (f=2, β=45). |
| `liftup_f1em4_b163.yaml` | Lift-up comparison point (f=1e-4, β=163). |
| `point.sbatch.template` | Slurm batch template for map campaigns. |
| `savarino_map317_points.csv` | 317-pt (β, f) grid for `--grid dense317`. |

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
2. **Run the two comparison points** (writes `output/compare/`):
   ```
   python run_compare_points.py
   ```
3. **Verify the norm convention** at (f, beta) = (2, 45):
   ```
   python verify_norm_convention.py
   ```
4. **Plot mode comparisons** (S4/S5, Savarino2026 layout):
   ```
   python compare_modes.py
   ```
5. **Run a gain-map campaign** (200- or 317-point grid), through Slurm or by
   printing the commands to run locally:
   ```
   python gen_campaign.py --out results/campaign
   python gen_campaign.py --out results/campaign --grid dense317
   python gen_campaign.py --out results/campaign --local
   ```
6. **Collect** the per-point outputs into one run-record CSV:
   ```
   python collect_campaign.py results/campaign > results/campaign.csv
   ```
   For an existing NPZ tree (e.g. Savarino2026 dense map, 317 points):
   ```
   python collect_campaign.py --from-npz /path/to/resolvent_map_repo_dense \\
       --points-csv configs/savarino_map317_points.csv \\
       --out output/savarino_map317pt.csv merged
   ```
   Committed run records: `output/savarino_map2_200pt.csv`, `output/savarino_map317pt.csv`.
7. **Plot the gain map** (S6, Savarino2026 style):
   ```
   python plot_gain_map.py
   ```
   For a densified map with sample overlays:
   ```
   python plot_gain_map.py --summary path/to/summary.csv --show-points \\
       --baseline-summary output/savarino_map2_200pt.csv
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

`output/savarino_map2_200pt.csv` is the committed 200-point gain map: one
row per (beta, f) point, with gain, residuals, wall time, memory, DOF, nnz,
and full provenance (git SHA, machine, ranks, backend, epsilon, forcing
region) so each row is self-describing outside the repository.

The per-point field data (mode shapes, full solution vectors) behind these
rows is not committed here -- it is published as a separate dataset; see
the repository release notes.
