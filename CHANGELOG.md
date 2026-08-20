# Changelog

All notable changes to linstabpy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Martini compute scripts wrote NPZ and a log only. Plot scripts loaded those
NPZ files and wrote the PNG figures. A conda file installed the CPU stack
from `pyproject.toml`.

### Added
- `environment.yml` installed the package from `pyproject.toml` on a conda-forge complex PETSc stack.

### Changed
- PETSc-only tests skipped collection when mpi4py or petsc4py was missing.
- Auto PETSc matrix layout stayed local when petsc4py was absent.
- README installation named the CPU stack. It stated that no GPU conda file ships.
- Martini `example_*.py` drivers wrote NPZ and a log only.
- `example_*_plots.py` loaded NPZ files and wrote PNG figures.
- `run_all_examples.sh` computed each Martini case, then ran the plot module.
- `benchmark.sh` ran Example 0 as `examples.martini.example_0_curvilinear_validation`.
- README, tutorial, and runner help used `examples.martini.*` module names.

### Removed
- `.pre-commit-config.yaml`. The file-size guard stayed in pytest and CI.
- Unused `tests/fixtures/curvilinear_derivative_parity.json`.
- Campaign leftover `examples/martini/physics_audit_campaign_vs_repo.json`.
- Example 0 write to `artifacts/curvilinear_derivative_parity.json`.

### Verified
- Tests locked `environment.yml` against `pyproject.toml`.
- Tests locked the Martini compute/plot split and the leftover-file ban.
- PETSc-free omit-list coverage rose from 75% (4969/6618) to 83% (5467/6618).
- Full-package coverage without PETSc was 50% (5467/11008). `.coveragerc` left out 4390 solver, linalg, and PETSc-helper statements at 0%.
- The PETSc-free suite collected 913 passed tests. PETSc tests on this CPU box did not collect or did not pass.
- Tests covered driver CLI, sponge, shock cleanup, Floquet shift, and wavemaker inputs.
- Tests covered grid-field load, filters, conservative-momentum forcing, and typed solver config.
- Tests covered mesh create limits, SBP dissipation errors, and empty Dirichlet recipes.

Omit-list files (`.coveragerc` source, solvers and linalg omitted). Hit 5467, miss 1151, stmts 6618.

| File | Hit | Miss | Stmts | Cover |
|---|---:|---:|---:|---:|
| `resolvent/gain.py` | 19 | 100 | 119 | 16% |
| `operators/spatial.py` | 39 | 101 | 140 | 28% |
| `stability/_eigenvalues.py` | 102 | 232 | 334 | 31% |
| `resolvent/operator.py` | 18 | 33 | 51 | 35% |
| `resolvent/analysis.py` | 47 | 78 | 125 | 38% |
| `driver/run.py` | 447 | 213 | 660 | 68% |
| `utils/provenance.py` | 78 | 26 | 104 | 75% |
| `operators/assembly.py` | 519 | 165 | 684 | 76% |
| `boundary_conditions/recipe.py` | 172 | 44 | 216 | 80% |
| `boundary_conditions/dirichlet.py` | 150 | 32 | 182 | 82% |
| `utils/cost_metrics.py` | 42 | 8 | 50 | 84% |
| `boundary_conditions/neumann.py` | 127 | 24 | 151 | 84% |
| `mesh/sbp_generator.py` | 147 | 23 | 170 | 86% |
| `operators/metrics.py` | 178 | 15 | 193 | 92% |
| `mesh/mesh.py` | 615 | 47 | 662 | 93% |
| `driver/io.py` | 55 | 2 | 57 | 96% |
| `config/typed.py` | 296 | 4 | 300 | 99% |
| `baseflow/sponge.py` | 205 | 2 | 207 | 99% |
| `stability/sensitivity.py` | 104 | 1 | 105 | 99% |
| `baseflow/preprocess.py` | 139 | 1 | 140 | 99% |
| `__init__.py` | 28 | 0 | 28 | 100% |
| `baseflow/__init__.py` | 5 | 0 | 5 | 100% |
| `baseflow/baseflow.py` | 21 | 0 | 21 | 100% |
| `baseflow/grid_fields.py` | 63 | 0 | 63 | 100% |
| `baseflow/shock_cleanup.py` | 215 | 0 | 215 | 100% |
| `baseflow/viscosity.py` | 23 | 0 | 23 | 100% |
| `boundary_conditions/__init__.py` | 11 | 0 | 11 | 100% |
| `config/__init__.py` | 4 | 0 | 4 | 100% |
| `config/defaults.py` | 3 | 0 | 3 | 100% |
| `config/schema.py` | 53 | 0 | 53 | 100% |
| `driver/__init__.py` | 1 | 0 | 1 | 100% |
| `driver/cli.py` | 63 | 0 | 63 | 100% |
| `driver/payload.py` | 12 | 0 | 12 | 100% |
| `driver/preflight.py` | 165 | 0 | 165 | 100% |
| `mesh/__init__.py` | 11 | 0 | 11 | 100% |
| `mesh/fdweights.py` | 25 | 0 | 25 | 100% |
| `mesh/filters.py` | 189 | 0 | 189 | 100% |
| `mesh/sbp.py` | 211 | 0 | 211 | 100% |
| `metrics_schema.py` | 38 | 0 | 38 | 100% |
| `mumps_options.py` | 3 | 0 | 3 | 100% |
| `operators/__init__.py` | 4 | 0 | 4 | 100% |
| `operators/coefficients.py` | 453 | 0 | 453 | 100% |
| `operators/energy_norm.py` | 12 | 0 | 12 | 100% |
| `resolvent/__init__.py` | 12 | 0 | 12 | 100% |
| `resolvent/conservative_forcing.py` | 99 | 0 | 99 | 100% |
| `stability/__init__.py` | 20 | 0 | 20 | 100% |
| `stability/acceptance.py` | 93 | 0 | 93 | 100% |
| `utils/__init__.py` | 2 | 0 | 2 | 100% |
| `utils/_solver_cfg.py` | 67 | 0 | 67 | 100% |
| `utils/floquet.py` | 32 | 0 | 32 | 100% |
| `utils/yaml_io.py` | 30 | 0 | 30 | 100% |

Files omitted by `.coveragerc`. Statement count from the coverage parser. Hit 0 on this CPU box.

| File | Stmts | Cover |
|---|---:|---:|
| `solvers/router/stable_sweep.py` | 675 | 0% |
| `solvers/router/resolvent_map.py` | 573 | 0% |
| `linalg/factorize.py` | 452 | 0% |
| `utils/petsc_core.py` | 387 | 0% |
| `solvers/eigenvalue.py` | 320 | 0% |
| `solvers/router/map_walk.py` | 265 | 0% |
| `linalg/backends.py` | 262 | 0% |
| `solvers/router/ram_autoselect.py` | 252 | 0% |
| `solvers/router/anchors.py` | 204 | 0% |
| `solvers/router/recycling.py` | 199 | 0% |
| `solvers/router/certification.py` | 172 | 0% |
| `solvers/router/break_even.py` | 159 | 0% |
| `solvers/spatial_eig.py` | 102 | 0% |
| `solvers/router/band_marking.py` | 96 | 0% |
| `solvers/router/map_record.py` | 78 | 0% |
| `utils/petsc_init.py` | 66 | 0% |
| `solvers/_eps_helpers.py` | 63 | 0% |
| `linalg/shifted.py` | 21 | 0% |
| `solvers/router/__init__.py` | 19 | 0% |
| `linalg/__init__.py` | 13 | 0% |
| `solvers/__init__.py` | 11 | 0% |
| `solvers/petsc_utils.py` | 1 | 0% |

## [1.0.0rc7] - 2026-08-19

Comments in the example scripts and tests were shortened. Ruff formatted
the test tree. The Martini audit-plan notes were removed.

### Changed
- Example and test comments.

### Removed
- Martini audit-plan and parity-summary notes.

## [1.0.0rc6] - 2026-08-19

The supplement examples ship complete: Martini 0-5, Song 32 deg, and
Savarino, with their parity audits, slim reference fixtures, and the
paper-style figures. Two operator corrections came out of that audit.
On a deformed y-symmetric mesh the metric terms now come from the full
eta operators, so the family second derivatives converge at the axis.
The Floquet shift defaults to the continuum chain rule; the LinStab2D
discrete form stays behind `matlab_parity`.

### Added
- Each Savarino map-walk row records its commit, its Slurm job, and its machine.
- `resolvent_analysis` returns the factor, SVD, and residual stage clocks.
- The Savarino line campaign times a recycled sweep against a factor at each frequency.
- **Song (2023) example complete** (`examples/song/`, θ = 32° only): self-contained
  supplement pipeline from Song FV export through LinStabPy validation figures.
  - `prepare_baseflow.py` — Plot3D → nodal `.npy` (601→599 trim, interpolation, nondim).
  - `plot_baseflow.py` — mesh + mean-flow panels (`examples.figure_style` LaTeX fonts).
  - `configs/run_fd4_n30.yaml` — FD4, *n* = 30, β = 19.2, no operator dissipation (ε = 0).
  - `song.py` — EVP driver; default config → `configs/run_fd4_n30.yaml`.
  - `report_matched20.py`, `plot_spectrum_matched20.py`, `plot_eigenmode_lead.py` —
    Hungarian matched-20 spectrum and lead eigenmode vs Song reference.
  - `song_plot_utils.py`, `reference_paths.py`; shipped reference under `reference/`.
  - Shipped validation figures under `figures/` (baseflow mesh/fields, matched-20
    spectrum, lead eigenmode; FD4 *n* = 30, ε = 0).
  - Local-only outputs: `inputs/baseflow/`, `output/fd4_n30/` (gitignored).
  - Validated (FD4 *n* = 30, ε = 0): 20/20 Song modes within |Δω| ≤ 0.01; lead |Δω| ≈ 0.002.
- **Martini (2024) examples 0–5 — presentation + parity closure** (`examples/martini/`):
  audit **CLOSED 2026-08-18** (`AUDIT_PLAN.md`; steps A→E, user sign-off all six cases).
  - Per-case plotters: `example_{0..5}_plots.py` plus shared
    `plot_compare_utils.py`, `split_martini_fixtures.py`,
    `run_all_martini_examples_and_plots.py` — thesis-style mosaics (LinStab2D |
    linstabpy | peak-normalized % error), spectra, and baseflow panels.
  - Slim MATLAB reference fixtures under `examples/martini/LinStab2DResults/`
    (compact u+v / u+v+w, **complex128**); bundled `linstab2D_scripts/` and
    `regenerate_martini_fixtures.sh` for MATLAB → NPZ regen.
  - Bundled baseflow `.mat` files under `examples/martini/baseflows/`
    (`BL_BF.mat`, `Step_BF.mat`, `M09jet.mat`, `Paraboloid_BF.mat`).
  - Shipped presentation PNGs next to each driver (baseflow, spectrum, mode
    mosaics for Ex. 1–5; validation triptych for Ex. 0).
- **Savarino (2026) example — reorganized supplement reproduction** (`examples/savarino/`,
  θ = 30.8° SWBLI resolvent validation, Figs. S4–S6):
  - **Layout:** `configs/` (YAMLs + grids), `reference/` (author baseflow + packed
    response modes), `output/` (run records, figures, local intermediates).
  - **Configs** (`configs/`): `map_base.yaml`, `mack_f2_b45.yaml`,
    `liftup_f1em4_b163.yaml`, `point.sbatch.template`, `savarino_map317_points.csv`
    (317-pt β–f grid for dense-map campaigns).
  - **Shipped reference:** Tecplot base flow under `reference/baseflow/`; packed
    `reference/savarino_response_{mack,liftup}.npz` (primitive ρ′, u′, v′, w′, T′).
  - **Shipped results** (`output/`): `savarino_map2_200pt.csv` (200-pt gain-map run
    record), `savarino_map317pt.csv` (317-pt minimal dense subset from PhD
    `map_minimal_keep_author200`), validation figures under `output/figures/`
    (mode comparisons + 200-/317-pt gain maps).
  - `collect_campaign.py --from-npz` — aggregate NPZ + `.metrics.json` trees into
    the run-record CSV layout (for PhD dense-map campaigns without Slurm logs).
  - `gen_campaign.py --grid {author200,dense317}` — Slurm/local campaign generator
    for 200- or 317-point grids.
  - Plotters aligned with PhD Savarino2026 style:
    - `plot_gain_map.py` — tensor/PCHIP interpolation, σ₁² colorbar with ×10⁴
      on top, Mack / lift-up peak markers (`plot_resolvent_map_savarino_style.py`
      parity); optional `--show-points` / `--baseline-summary` for diagnostics.
    - `compare_modes.py` — Song eigenmode layout, peak-normalised ±1 colourbar,
      ymax 0.06 (Mack) / 0.1 (lift-up) (`plot_savarino_modes_linstabpy_template.py`
      parity).

### Changed
- Recycled GMRES restarts. The restart is a Krylov length. It is never the system size.
- `certified_rsvd` logs every check when one check fails. One seed skips the multi-seed check.
- **Floquet shift default is now `continuum`** (`LINSTABPY_FLOQUET_SHIFT`,
  `numerics.floquet_shift`): `DZ += iα I`, `D2RZ += iα DR`, `D2Z += 2iα DZ − α² I`,
  the chain rule of `(∂z + iα)`. The LinStab2D algebra (`D2RZ += iα I` and `D2Z`
  built from the already-shifted `DZ`, i.e. `−3α² I`) stays available as
  `matlab_parity`; the Martini Ex. 3 YAMLs and `examples/sawtooth` pin it.
  Plane Poiseuille at Re = 10⁴, α = 1 (Orszag 1971): `continuum` converges to
  ω = 0.23752649 + 0.00373967i (|Δω| 3.8e-4 → 1.8e-6 for ny 128 → 512);
  `matlab_parity` stalls at |Δω| ≈ 3.2e-4, a growth rate 8.5 % low, at every
  resolution. Any unpinned Floquet run changes with this default.
- Replaced the legacy Song figure/pack/matrix scripts with the pipeline above;
  see `examples/song/README.md` for run order.
- `song.py` creates the output NPZ parent directory before save and reads
  `numerics.operator_dissipation` from YAML (disabled, ε = 0 for this test).
- `song_plot_utils.load_song_leading_mode_npz` transposes the shipped reference
  eigenmode NPZ to LinStabPy `(ny, nx)` layout for plotting.
- `.gitignore` re-ignores regenerated example outputs (`examples/**/figures/`,
  `examples/**/output/`, Song `inputs/baseflow/`, logs) after the example PNG
  whitelist; Song `figures/` and reference NPZ are shipped; raw Plot3D exports
  under `inputs/raw/` stay local.
- Song plot scripts require LaTeX via `figure_style.check_plot_environment`;
  see `examples/song/README.md` for system packages (`texlive-latex-extra`, …).
- `examples.figure_style`: `check_plot_environment(require_latex=…)` with install
  hint, optional `dpi` / `pad_inches` on `save()`; tests in `tests/test_figure_style.py`.
- Grouped Song (2023) **reference** assets under `examples/song/reference/`:
  `baseflow/song_32deg.npz`, `eigenmode/song_32deg_leading_mode.npz`,
  `spectrum/eigenvalues.dat`, `spectrum/omega_i.csv`. Added `reference_paths.py`
  and updated configs, figure scripts, and tests.
- **Martini fixture layout and git policy:**
  - Shipped NPZ: **`examples/martini/LinStab2DResults/*.npz` only** (slim MATLAB
    reference). Root-level `example_*.npz` from linstabpy drivers are
    **gitignored** (regenerated locally; several exceed the 60 MiB tracked-file
    guard).
  - Raw MATLAB `.mat` exports stay outside Git in `martini_examples/` (gitignored
    scratch; recreated by `regenerate_martini_fixtures.sh`).
  - Run logs under `examples/martini/_run_logs/` are gitignored.
  - Renamed `examples/martini/data/` → `baseflows/`.
- **Martini plot/compare layer** (repo mosaics ≤ **0.49%** peak-norm vs campaign;
  Ex. 4 cavity audit closed — residual tail is MATLAB `eigs` convergence, not
  operator/BC):
  - `plot_compare_utils.py`: `used_ind` 0-based gridding, `matlab_plot_grid_shape`,
    `compact_resolvent_modes_uvw`, peak-normalized mosaic triptychs.
  - `example_4_plots.py` / `split_martini_fixtures.py`: slim MATLAB u+v+w compare
    with joint complex scaling; Example 4 YAML pins `Pr: 0.7` and
    `zero_dmudT: true` for LinStab2D parity.
  - `compare_martini_examples.py` and `tests/test_martini_parity_compare.py`
    updated for `LinStab2DResults/` fixture paths and compact mode layout.
- Moved library validation CLIs from `examples/martini/` to `tests/tools/`
  (`assembly_parity.py`, `assembly_metrics_diff.py`, `lnse_equivalence_harness.py`,
  `matlab_source_parity.py`, `symbolic_compressible_nse.py`,
  `coefficients_reference_base.py`, `NSE_DERIVATION.md`). See
  `tests/tools/README.md` for rationale and owner review checklist.
- Removed unused maintainer-only scripts from `examples/martini/`:
  `check_parity_matrices.py`, `compare_operator_construction.py`,
  `perf_memory_operator_parity.py`, `operator_matrix.json`, `branch_matrix.json`,
  `martini_c64_max_rel.txt`, `martini_examples.sbatch`.
- **Savarino example layout and git policy:**
  - Moved point YAMLs, campaign template, and 317-pt grid into `configs/`; run
    records into `output/` (`savarino_map2_200pt.csv` was at example root).
  - `reference_paths.py`: `GAIN_MAP_CSV`, `GAIN_MAP_317_CSV`, figure paths under
    `output/figures/`; compare outputs under `output/compare/`.
  - `.gitignore` ships **only** curated Savarino outputs: both map CSVs and
    `output/figures/*.png`. **Gitignored:** `output/baseflow/*.npy` (from
    `savarino_to_npy.py`), `output/compare/**/*.npz` (from `run_compare_points.py`),
    `reference/zenodo/`. Whitelist repeated after root `output/` rule.
  - `run_compare_points.py`, `gen_campaign.py`, `collect_campaign.py`, `mode_oracle.py`
    updated for `configs/` paths.
  - See `examples/savarino/README.md` for minimal workflow and maintainer scripts
    (`verify_norm_convention.py`, `mode_oracle.py`, `fetch_zenodo.py`,
    `pack_reference_modes.py`).

### Fixed
- `Mesh.deform` on a y-symmetric mesh with a non-separable map: the `dJ`/`Q`
  metric terms of the symm/asymm family operators are now differentiated with
  the full `Dy`/`D2y` (the coordinate map has no η parity), as LinStab2D
  `DeformMesh.m` does. Before, the family stencils were applied to the
  coordinates themselves and `D2y_asymm`/`D2y_symm` did not converge at the axis
  rows (relative error growing like 1/h under refinement: 7.7 → 61 for
  n = 16 → 128 on a polynomial map; 3.5 → 29 on the Z² map). After: 2.2e-4 → 3.9e-7
  (order 3, asymm) and 2.1e-3 → 3.1e-5 (order 2, symm). Only `y_symmetry=True`
  meshes on the full-2D deform path are affected (Martini Ex. 5 paraboloid;
  the SWBLI cases are untouched). Regression: `test_deform_full_2d_family_metric_terms_use_full_operators`.
- The walk probe uses the last response, not a ones vector.
- A walk GMRES has a four-iteration cap. A dead walk raises and factors this omega.
- `append_row` refuses a row that does not match the header.

### Removed
- Dropped `song_fig{1,2,3}_*.py`, `pack_run.py`, `run_song_2023_*.py`, `song.sh`,
  `song_32deg_spectrum_quoted.csv`, and the old `song_32deg*.yaml` / sbatch configs
  from `examples/song/`.
- Dropped committed LinStabPy **outputs** from the old flat `examples/song/` layout
  (`song_32deg_modes.npz`, `song_32deg_spectrum.csv`, `song_matlab_timing.json`,
  root-level `song_fig*.png`). Large EVP dumps and raw Plot3D inputs stay local;
  curated reference NPZ and validation PNGs now live under `reference/` and
  `figures/` respectively.
- Dropped the 31.6 deg Song example from `examples/song/`: baseflow
  (`song_316deg.npz`, `song_316deg_leading_mode.npz`), reference spectrum
  (`song_316_eigenvalues.dat`), configs (`song_316deg.yaml`, `song_316deg.sbatch`),
  and comparison helpers (`compare_evp_to_song316.py`, `parse_song316_eigenvalues.py`).
  The shipped example is **32 deg only**; 31.6 deg validation stays in the PhD case tree.
- Dropped committed **linstabpy driver NPZ** from `examples/martini/` root
  (`example_*.npz`, legacy `*_values.npz` / `*_modes.npz` / `*_logs.npz` splits).
  Regenerate locally via `./run_all_examples.sh` or per-case drivers; parity
  against MATLAB uses `LinStab2DResults/` only.
- Dropped old debug/support PNGs from `examples/martini/` (per-case
  `*_support_components*.png`, legacy single-panel `example_*_{blasius,jet,...}.png`
  layouts superseded by mosaic figures).
- Dropped `savarino_corner_probe_uv.yaml` and `savarino_corner_probe_wonly.yaml`
  (side experiments; not part of the S4–S6 reproduction path).
- Dropped example-root Savarino YAMLs and Slurm template (now under `configs/`):
  `savarino_point_f2_b45.yaml`, `savarino_point.sbatch.template`; renamed
  `compare_liftup_f1em4_b163.yaml` → `liftup_f1em4_b163.yaml`.
- Dropped committed PNGs from `examples/savarino/reference/` (figures now under
  `output/figures/`).


## [1.0.0rc5] - 2026-08-15

The gain map walks each frequency line. The first two frequencies
pay a MUMPS factor. Later frequencies walk only when a probe beats
that measured cost. Otherwise the line factors again.

### Added
- `LineWalker` walks one beta line. The first two frequencies factor with MUMPS.
- A cold forward and adjoint probe decides whether a walk can beat a new factor.
- `map_record` writes each finished point and resumes unfinished frequencies.

### Changed
- `resolvent_map` walks each beta line with `LineWalker` and SLEPc.
- A walk needs a measured apply count and a cheap probe. Otherwise the line factors.
- GMRES stops when a deadline has passed.
- The map store writes a temp file and then replaces the old store.

### Fixed
- The walk reads GMRES lengths from the walk module, not from the recycle module.

## [1.0.0rc4] - 2026-08-13

The Example 2 jet resolvent matched the MATLAB LinStab2D gains. The axis
coefficient used the true radius. The spatial filter sat inside the
forcing and response masks.

A forcing region composed with the boundary-condition mask.
`input_region.space: conservative_momentum` mapped conservative momentum
forcing onto the primitive residual.

Case files became YAML. One loader read a packed field. The tree dropped
JSONC, import shims, `scripts/`, and the unused Dockerfile.

### Fixed
- `compute_cylindrical_coefficients` no longer floored R at the smallest
  positive spacing. A grid that contains r = 0 raised ValueError.
- `ResolventOperator` applied the spatial filter inside the input and output
  masks, matching MATLAB `resolvent.m`.
- The Example 2 leading gain moved from 411.72 to the MATLAB 413.72 reference.
- A forcing or response region composed with the boundary-condition mask.
- `mask_bc_forcing` stayed true by default. A region no longer forced
  constraint rows.
- A region that selected only constrained rows raised.
- `mask_bc_forcing: false` still forced those rows on purpose.
- The Floquet shift also updated D2RZ. The default mode matched LinStab2D.
- The spatial EVP finite-differenced L in kx, not in the Floquet exponent.
- Example 3 applied spatial Dirichlet after Lc + omega Lw.

### Added
- `input_region.space: conservative_momentum` mapped conservative momentum
  forcing onto the primitive residual with a mass-weighted L2 norm.
- Momentum forcing induced a consistent temperature forcing
  `T' = -kappa(kappa-1) Ma^2 (ubar · u')`.
- `load_config` loaded YAML case files.
- `load_case_fields` read a packed npy directory or one npz.
- `examples/song` and `examples/savarino` shipped the reproduction chains.
- `examples/sawtooth` added the Lui 2022 field on a packed stride-2 300 x 340 mesh.
- Four paper Example 3 YAML files reused `example_3_streaky_bl.py`.
- `examples/martini/matlab_baseflow.py` held the Martini MATLAB field constructors.
- `build_runtime_payload` grouped the runtime NPZ keys before `save_results`.
- `install.sh --diagnose` printed the host preflight and installed nothing.
- `tests/tracked_file_size.py` and `tests/public_provenance.py` replaced `scripts/`.

### Changed
- Example figures probed TeX by writing a PNG. They used usetex when that worked.
- A failed usetex save retried with Computer Modern mathtext.
- A missing TeX stack no longer crashed the figure.
- Case files under `examples/` became YAML. The loader rejected a leftover `.jsonc` suffix.
- Shipped YAML files now comment each key. `validation_cases.yaml` is YAML.
- Comments and the tutorial now say YAML, not JSONC.
- Sponge presets dropped the `case1_` prefix.
- Moved the Martini, Song, and Savarino cases to `examples/martini`,
  `examples/song`, and `examples/savarino`.
- Packed the Song reference fields as float64 npz. The 32 deg field uses
  velocity / c_inf and temperature / T_inf.
- Packed Martini mode arrays as complex64. The largest file is
  `example_3_Streaky_BL_results_temporal_modes.npz`.
- Raised the tracked-file size limit from 50 MiB to 60 MiB for that file.

### Removed
- `load_jsonc`, `load_hamada_case`, `results_runtime`, and the solver import shims.
- The `run/` campaign kit and the broken Docker PETSc CI job.
- The `mumps_icntl_7` boolean alias. Use `mumps_ordering: metis`.
- The unused `Dockerfile` that did not configure PETSc.
- The `scripts/` folder and `diagnose.sh`.

## [1.0.0rc3] - 2026-08-10

### Changed
- **BREAKING (default):** `numerics.operator_dissipation.apply_to_resolvent` now
  defaults to `true` instead of `false`, matching `apply_to_stability`. Previously,
  enabling dissipation gave a damped eigenvalue problem and a **bare** resolvent
  operator — a combination that runs to completion and returns grid-scale sawtooth
  (spurious high-wavenumber streaks, gains orders of magnitude too large) with nothing
  in the log to indicate it. Configs that deliberately want an undamped resolvent must
  now set `apply_to_resolvent: false` explicitly. No config shipped with the project
  changes behaviour: every one that runs a resolvent either disables dissipation or sets
  the switch explicitly. Both defaults now live in one place,
  `config.defaults.DISSIPATION_APPLY_DEFAULTS`, so the driver and the preflight checks
  cannot disagree about them.
- `solver.factor_backend` is now honoured as a fallback for all solver stages
  (`linsolve`, `resolvent`, and the shift-invert eigen path). It was previously read by
  nothing: every resolver receives a *sub*-config, so a key at solver level silently did
  nothing — one symptom being a run intended for GPU that executed entirely on CPU. It
  sits at the bottom of the precedence chain (per-stage key > environment >
  `solver.factor_backend` > built-in default), so no configuration that resolved before
  resolves differently now. Route-specific defaults (for example `strumpack_fgmres`
  requiring `strumpack`) are unaffected.

### Added
- Preflight and post-run advisories (`driver.preflight`), reported in the run log. Each
  advisory names a concrete remedy rather than only a symptom. Preflight covers:
  dissipation enabled but not reaching a stage that is about to run; no dissipation at
  all; dissipation enabled with `epsilon = 0`; a spanwise wavenumber inside the band
  where grid-scale sawtooth dominates while undamped; a base flow that is not normalised
  to the canonical freestream (`rho = u = T = 1`); a shadowed `factor_backend`; and a
  strided (reduced-resolution) mesh. Post-run checks flag a leading gain far above the
  physical scale for these cases and a postcheck residual above tolerance.
- A run **banner** and closing **summary** in the log, reporting *effective* values —
  including the per-stage dissipation epsilon actually applied — so a run's real
  configuration and the trustworthiness of its result are visible without loading the
  output archive.
- Translation of opaque factorization failures into actionable messages. A MUMPS
  workspace exhaustion (`INFOG(1) = -9`) previously surfaced only as
  `SLEPc EPS.setUp failed`; it now names the cause and the remedy (`ICNTL(14)`).

- A **band-marking backend** advisory. `resolvent_map` runs a mandatory band-marking
  eigenvalue problem, and that eigenproblem reads `solver.eigen` — not
  `solver.resolvent`. A configuration that moved only the resolvent onto a scalable
  backend therefore left the eigenproblem on the default direct solver, where at full
  resolution it does not fail but simply never finishes; the run looks like a slow
  resolvent sweep for hours. The advisory names the three ways out (`solver.factor_backend`,
  an explicit `solver.eigen.factor_backend`, or `analysis.skip_band_marking`).
- A source-wide lint (`tests/test_mpi_collective_lint.py`) rejecting MPI collectives
  inside rank-0-only regions, in both the `if rank == 0:` and `if rank != 0: return`
  forms. Such a call deadlocks every multi-rank run while remaining invisible to the
  serial test suite, which runs at size 1. The lint ships with positive controls, since
  a lint never shown to fire cannot be distinguished from one that cannot fire.

### Fixed
- The single-point resolvent path now honours `input_region` and `output_region`, and no
  longer forces the response mask to equal the forcing mask. `driver.run._stage_run_resolvent`
  built its own boundary-condition-only diagonal and then set `C_mat = B_mat.copy()`, so two
  things went wrong at once and neither said anything: a config naming a forcing region ran
  with no region at all, because only `solvers.router.resolvent_map` ever read those keys; and
  even a correct region was copied onto the response, which turns an asymmetric problem
  (restricted forcing, free response) into a different one. Both paths now call the same
  `_forcing_response_mats` helper, so one config key means one thing everywhere, and `C`
  defaults to the boundary-condition mask rather than to `B`. Runs that set no region are
  unchanged: `B` and `C` are still the boundary-condition mask, asserted byte-for-byte by a
  regression test. `scripts/validation/song_2023/song_2023.py` builds its masks separately and
  cannot honour a region, so it now raises instead of ignoring the key.
- MPI is now initialised at `MPI_THREAD_MULTIPLE` on every import path, restoring the
  STRUMPACK factorization backend. Whichever library calls `MPI_Init` first fixes the
  thread level for the process, and it can only be *requested* at init — never raised
  afterwards. `mpi4py` requests `MPI_THREAD_MULTIPLE`, PETSc only
  `MPI_THREAD_FUNNELED`. `utils.petsc_init.init_petsc_from_argv` already imported
  `mpi4py` first, but seven modules imported `petsc4py`/`slepc4py` at *module scope*
  and bypassed it, so whichever of them was imported first decided the level. When
  PETSc won, `MatGetFactor_aij_strumpack()` failed with PETSc error 57 and `SLATE
  requires MPI_THREAD_MULTIPLE`, taking the whole STRUMPACK/GPU route out of service —
  8 of 14 failures in the GPU test venv had this single cause, and none of them
  reported anything about MPI. The guard is a plain `import mpi4py.MPI`, chosen over a
  first-party helper module because isort orders third-party imports above first-party
  ones: a `linstabpy.*` guard could not stay above `petsc4py`, and `ruff --fix` would
  have silently reordered the fix away. `mpi4py` sorts above `petsc4py` on its own.
  `tests/test_mpi_thread_level.py` enforces the ordering as a source-wide AST lint plus
  a runtime thread-level assertion that skips loudly without petsc4py.
- The freestream-normalisation check and the MUMPS error translation were defined but
  never called: both shipped with full unit-test coverage and zero call sites, so
  neither ran in production. They are now invoked from the driver and the eigensolver
  respectively, and a test asserts that every advisory producer is reachable from the
  runtime — unit tests establish that a guard works, not that anything calls it.
- ASM/BJacobi sub-preconditioner settings on the `inexact` shift-invert path are now
  actually applied. They were configured before `eps.setUp()`, where PETSc cannot yet
  materialise sub-solvers (`ierr=58`), and the error was swallowed — so `sub_pc_type`
  and `pc_factor_levels` were discarded. PETSc's default sub-PC is also `ilu`, which
  masked the type being dropped, but the fill level was not masked: a config asking for
  ILU(2) ran with ILU(0). The settings are now applied immediately after `eps.setUp()`,
  failures raise instead of downgrading, and the realized type and fill level are logged
  rather than assumed. This mirrors what `linalg.factorize` already did for the linsolve
  path. Only `st_solver: "inexact"` reached the defect — no shipped configuration selects
  it — so no previously reported result changes.
- MATLAB-parity tests are **opt-in**. This repository ships no MATLAB reference data,
  so the seven tests that compare against it called `pytest.skip` when it was absent —
  which is every machine but the two the data lives on. A run reporting
  "1361 passed, 13 skipped" read as near-total coverage without disclosing that seven
  of those skips could never be satisfied by anything in the repository. They now carry
  a `matlab_parity` marker, deselected by default, and are requested explicitly with
  `LINSTABPY_MATLAB_PARITY=1 pytest -m matlab_parity`. When requested, a missing
  reference is a **failure**, not a skip: a comparison you asked for that quietly does
  not run is indistinguishable from one that passed. A default run now reports
  6 skips, every one an absent optional backend — a true statement about the machine.
- Subprocess timeouts in the test suite are now load-aware everywhere, via a single
  shared helper. One test file already scaled its timeout with load average; its two
  siblings used hard limits of 120 s and 60 s, leaving as little as 1.4x headroom over
  the test's own runtime. On a shared or busy machine those expired and surfaced as
  test *failures*, indistinguishable from a numerical disagreement — a contributor
  running the suite with a browser open would see phantom errors. These timeouts guard
  against a hang; they are not performance assertions, so the bases are deliberately
  generous. The helper documents a limitation found in practice: load average counts
  runnable processes and does not detect memory-bandwidth starvation, so a co-resident
  memory-bound job slowed a test run ~3x while the computed multiplier was 1.17x.
- `production_case_template.jsonc`, the file the documentation tells a new user to copy
  for a production run, shipped with no `operator_dissipation` block at all. A first
  production run made from it was sawtooth-contaminated with nothing in the log to say
  so. It now enables dissipation and documents that `epsilon` must be re-tuned at the
  target resolution rather than inherited.
- The Streaky-BL MATLAB parity test guarded only one of the two references it needs, so
  restoring the parity bundle while the base-flow data directory was still absent turned
  a clean skip into a hard error that resembled a numerical disagreement. The guard now
  checks every reference and announces which one is missing.
- PETSc is now imported lazily by `linalg.backends`, `linalg.factorize`,
  `linalg.shifted`, and `solvers.router.stable_sweep`, matching the existing
  `utils.petsc_core` pattern. Importing these modules — or the
  `solvers.router` package, which re-exports `resolvent_map` — no longer pulls in
  petsc4py or initializes PETSc/SLEPc; initialization now happens on first
  actual solver use. This makes lightweight helpers such as
  `solvers.router.band_marking` usable without starting PETSc. Public APIs and
  numerical behavior are unchanged.

### Fixed
- Restored the guarantee that importing `solvers.router.band_marking` does not
  initialize PETSc (regressed when `resolvent_map` was added to the
  `solvers.router` package exports).
- The CPU/MUMPS parity regression guard no longer fails on freshly built
  environments. Its tolerance was tighter (1e-12) than both the eigensolver's own
  convergence tolerance (`eps_tol=1e-10`) and the measured build-to-build spread
  of the converged eigenvalues (~1e-8 between independently configured
  PETSc/MUMPS/BLAS builds), so it could not be stably satisfied. The guard now
  uses 1e-6 — far above that floor, far below any physically meaningful drift —
  and the frozen reference was regenerated. Repeat runs within one build remain
  bit-identical.

## [1.0.0rc2] - 2026-06-25

GPU-resolvent dual-factorization adjoint heal + amortized stable-sweep wired
into the turnkey gain-map driver. On an NVIDIA A6000, the healed GPU backend
resolvent gain and modes match the MUMPS CPU reference to ~2e-14. Regime-(b)
discounted resolvent and amortized stable-sweep solver acceleration are
validated and wired into the `resolvent_map` entry point. Turnkey gain-map
scripts (`run_gain_map.py` / `fig_gain_map.py`) are now shipped as public
examples.

### Added
- Turnkey gain-map driver scripts: `examples/case_studies/case1_song/scripts/run_gain_map.py`
  (compute (β,f) gain map end-to-end, resumable) and `fig_gain_map.py` (plot and
  annotate the result). Both read the Song 2023 box base flow and are self-contained
  executables shipped as public-facing Case 1 examples.
- Regime-(b) discounted-resolvent wiring into `resolvent_map`: NEAR_POLE band points
  compute the stabilized resolvent via `L0 - (γ + iω)I` (default γ=0.25, user-configurable);
  every regime-b point records `resolvent_discount_gamma` and is hard-gated to a finite gain.
  The regime-a direct path (γ=0) is unchanged and bit-identical to existing runs.
- Amortized stable-sweep solver acceleration into `resolvent_map`: factorizations are
  refactorized across anchor blocks (sparsity reuse), and RSVD probe blocks route through
  `solve_block` (PETSc `MatMatSolve` multi-RHS). Time savings reported in per-run metrics.

### Fixed
- **GPU resolvent adjoint heal via dual factorization:** STRUMPACK's complex transpose-solve
  is not a true adjoint (defect ~3.2e-2). The sweep now engages a dual-factorization heal:
  the operator is factored twice (once forward, once transpose) and the transpose solve
  routes through the transposed factor, recovering the true adjoint to machine precision
  (~2e-14 on an A6000; matching MUMPS). Dual-factor mode is transparent and automatic;
  its engagement is logged. Validated on an A6000 with Song 2023 base flow: leading gain
  and structural-sensitivity modes match the MUMPS reference. This closes the GPU-resolvent
  correctness blocker (bead r3uu.14.17). CPU/MUMPS path unchanged (still single-factor).
- `wire factor_backend` into iterative LU preconditioner: the shift-invert solver now passes
  the selected `factor_backend` (e.g. STRUMPACK) through to the PETSc LU preconditioner
  (`KSP` path), instead of always falling back to MUMPS, so GPU-accelerated FGMRES
  preconditioners route to the GPU backend. Validated on c9 A6000 with complex resolvent:
  residuals match the direct path to ~2e-14.

### Changed
- Case 1 Song base flow is now genericised in the config comment: `song_box_evp.jsonc`
  reads "Song SWBLI box" (removed author name).

## [1.0.0rc1] - 2026-06-18

Physics-campaign freeze release candidate: the turnkey `resolvent_map(base_flow,
beta_range, f_range, cfg)` entry point (base-flow in, (β,f) gain map + cost metrics out)
with region masks on the CPU/MUMPS-validated frozen-numerics path. Seven Appendix-A
oracles verified to <1e-10; full suite green; single frozen SHA for the validation campaign.

### Added
- Turnkey region masks for `resolvent_map`: `input_region`/`output_region` cfg specs (`{"x": [x0,x1], "y": [y0,y1], "vars": [...]}`, any key omittable → full extent / all vars) build the diagonal 0/1 forcing/response operators (`B_mat`/`C_mat`) internally, so a windowed or component-restricted resolvent gain needs no hand-built ndof×ndof matrices and no knowledge of the LNSE state layout. The mask builder honors the layout exactly — active points from `mesh.X/Y.ravel(order="F")[mesh.used_ind]`, variable blocks from `idx['{rho,u,v,w,T}_j']` — and is a sibling of the existing BC-forcing mask helper; an explicit `B_mat`/`C_mat` still takes precedence, then region specs, then the default BC-forcing mask. A full-domain all-vars region is the identity on every dof and reproduces the unmasked resolvent gain and modes to machine precision (~1e-12, phase-aligned); empty/out-of-domain/zero-active-point/unknown-variable specs loud-fail with a diagnostic `ValueError`; per-variable mask coverage (`n_active/n_total` + bounding box) is logged at INFO. Sharp 0/1 masks only (the spec is a dict so smoothed edges can extend it later). A mask changes the *question* to a windowed/component-restricted gain — it is not an approximation of the global gain. (bead resolvent-xc-program-v0-1-0-r3uu.14.15)
- Turnkey resolvent gain-map entry point `resolvent_map(base_flow, beta_range, f_range, cfg) -> RouterResult` (`solvers/router/resolvent_map.py`, re-exported as the public `linstabpy.resolvent_map`): the single user-facing thesis symbol — base-flow in, `(len(beta), len(f))` leading-gain map + intended band labels + CP3 cost-metrics sidecar out. Per β it assembles `L0(β)` once via the existing assembly seam (`build_lnse_system`), asks `mark_unstable_band` for the per-frequency band labels, and computes every `(β,f)` point through the certified/direct `resolvent_analysis` path. The discounted (regime b) and ROM (regime c) speed engines do not exist yet, so the router is honest about it: every point currently runs on regime (a), the band label is recorded as the *intended* route, and the regime-b/c paths are gated off by default (`enable_regime_b`/`enable_regime_c`) — when left off a one-line fallback note is logged, when switched on they raise `NotImplementedError` naming the missing bead rather than silently faking a result. Each point logs index/total, ω, β, leading gain, residual, wall, peak RSS via the existing CP3 cost block + stage-heartbeat env (no new logging system); the sweep is resumable (points already present in the output `.npz` are skipped, same cache-on-existence convention as the examples) and per-point failure-tolerant (a failed solve is caught, recorded with its exception class + last state, marked `NaN`+reason-code, and retried on a resumed run rather than aborting the map). Numerics are reused, not re-implemented; an e2e test asserts a regime-(a) map point matches a standalone `resolvent_analysis` oracle within tolerance. (bead resolvent-xc-program-v0-1-0-r3uu.14.6)
- Frequency-count break-even router `break_even_decision(L0_pattern, n_frequencies, cfg) -> RoutingDecision` (`solvers/router/break_even.py`): decides direct per-frequency factorization vs the anchor-FGMRES sweep from a **measured** cost model, the frequency-count complement to the memory-driven `ram_autoselect` (r3uu.14.13). It times one real SciPy `splu` factorization (`T_fact`) and one representative solve (`T_apply`) on the supplied nonzero pattern (fabricated well-conditioned values; timing/fill depend on pattern not values), then computes the crossover `N_f* = K_anchors / (1 − iters·T_apply/T_fact)`: below it routes `"direct"`, at/above it routes `"fgmres-sweep"`. When `iters·T_apply ≥ T_fact` the sweep can never amortize its anchor factorizations, so the decision stays `"direct"` for every frequency count (crossover `inf`). `RoutingDecision` carries the chosen path, the measured crossover and timings, and a provenance note, so a cost figure can report which path each map used. The crossover is a measured number, not a guessed constant (the test varies only a probe's apply timing and asserts the crossover follows it). Illustrative small-pattern measurement — no production-scale speedup claim — and exposed but not auto-wired into any solver path (`resolvent_map`, r3uu.14.6, does not exist yet). (bead resolvent-xc-program-v0-1-0-r3uu.14.14)
- Stable-sweep solver acceleration (cost-only; numerically exact): the `linalg` factorization seam gained an additive `Factorization.solve_block` (PETSc `MatMatSolve` block multi-RHS, with a single-RHS-loop fallback for iterative/unsupported backends) and `Factorization.refactorize` (reuse the symbolic analysis/fill ordering for a new operator of identical nonzero pattern). `sweep_stable_regime` now factors the first anchor once and refactorizes in place across the remaining anchors — all anchors share K's sparsity (only the diagonal shift differs) — and routes the RSVD probe block through `solve_block`; it falls back to per-anchor factorization if reuse is unavailable or raises. The assembled/solved results are unchanged (block columns equal the single-RHS loop to round-off; symbolic-reuse gains equal recompute within the S7 σ-tolerance). `refactorize` rejects a differing nonzero pattern (collectively across MPI ranks) so reuse can never produce a silent wrong factor. Reductions in analysis time and block-vs-loop solve time are reported in `SweepResult.metrics`; these were demonstrated on a small representative case and carry no production-scale speedup claim. (bead resolvent-xc-program-v0-1-0-r3uu.14.8)
- Opt-in size-based factor-backend auto-routing: `factor_backend="auto"` resolves to STRUMPACK at or above a DOF threshold and to MUMPS below it, chosen at solve time once the operator size is known. The threshold is configurable (`auto_route_dof_threshold` JSONC field / `LINSTABPY_AUTO_ROUTE_DOF_THRESHOLD` env, precedence env > JSONC > default) with a default of 224k DOF — the c9 A6000 measured lower bound below which MUMPS wins; the true crossover up to ~900k is unmeasured, so the default is a tunable approximation, not a universal constant. When STRUMPACK is not available in the build the route falls back to MUMPS (logged), so `"auto"` is safe on CPU-only machines and never applies an absent backend. The default factor backend is unchanged (MUMPS); auto-routing is strictly opt-in, and resolving `"auto"` to MUMPS produces results numerically identical to an explicit MUMPS run. The GPU speedup itself is measured separately and is not exercised by this routing logic. (bead resolvent-xc-program-v0-1-0-r3uu.1.42)
- Documented the NVPL (NVIDIA Performance Libraries) CPU BLAS/LAPACK build recipe for Grace aarch64 in `docs/COMPUTE_BACKENDS.md`: the PETSc `--with-blaslapack-lib` linkage against `libnvpl_blas_*`/`libnvpl_lapack_*` (LP64-sequential pair for the default 32-bit-index complex build, with the ILP64/`_gomp` variants and the MUMPS-rejects-64-bit-BLAS-indices caveat noted), that SLEPc inherits the linkage, the NumPy/SciPy Meson option, and a complex128 (Z) `ldd`/`show_config` verification step. `install.sh` ships no NVPL stack, so it builds manually into an isolated `.venv-nvpl`; on-hardware activation and verification are deferred until Grace aarch64 hardware is available (no speed claim is made). (bead resolvent-xc-program-v0-1-0-r3uu.1.15)
- Adaptive pre-sweep anchor placement `choose_anchors_adaptive(omega_range, band_marks, cfg) -> (anchors, records)` (`solvers/router/anchors.py`): starts from a coarse anchor set and inserts anchors where the S6 EVP distance to the imaginary axis (`BandMarks.distances`) falls below a radius or where measured FGMRES iterations/residual at a sample frequency exceed a threshold, replacing fixed uniform spacing. It is a pure pre-sweep helper — it returns a *fixed* anchor set (plus a per-anchor `AnchorTriggerRecord` provenance trail naming each insertion's trigger) that the caller passes into `sweep_stable_regime`, so the chosen set is fixed per RSVD pass and the S7 determinism/linearity guarantees are unaffected. The FGMRES-iteration signal is supplied by an injectable probe (or measured arrays in cfg), defaulting to a cheap probe, rather than self-measured inside the chooser. On a small case the scheme places more anchors near a hard sub-band and fewer in the easy bulk, and the resulting sweep matches a direct MUMPS oracle within the S7 σ-tolerance; the adaptive-vs-fixed factorization-count comparison is logged as a measurement (no superiority gate — "adaptive beats fixed" stays a hypothesis to measure). Exposed but not auto-wired into a default sweep path. (bead resolvent-xc-program-v0-1-0-r3uu.14.11)
- RAM-calibrated operating-point selector `ram_autoselect(L0_pattern, cfg) -> OperatingPoint` (`solvers/router/ram_autoselect.py`): decides the resolvent solve operating point from the machine, not the user. It reads available RAM portably (injectable for tests; else `/proc/meminfo` `MemAvailable`, else `os.sysconf`), runs a cheap **empirical** sparse-LU analysis probe on the supplied nonzero pattern to measure the real factor fill count and bytes-per-nonzero on *this* build (the §7c formula is only the prior; the probe is the correction, R5), estimates the memory footprint, and picks `direct` when the factor fits the budget or `compressed-sweep` (raising compression / dropping precision / adding ranks) when it does not. `OperatingPoint` returns only resource knobs (backend, reorder, compression level, restart, ranks, precision); user physics choices (grid, fd order, modes, β/f ranges) are never read or altered — selection changes COST, not the numerical answer (the direct end matches a direct MUMPS oracle within the S7 σ-tolerance). The compressed-sweep end's memory ratios are an uncalibrated prior, and its end-to-end oracle check loud-skips where STRUMPACK is unavailable or its complex adjoint is non-certifiable (see r3uu.14.9). The selector is exposed but not auto-wired into `resolvent_map` defaults. (bead resolvent-xc-program-v0-1-0-r3uu.14.13)
- Opt-in GCRO-DR-style Krylov subspace recycling for the stable resolvent sweep (`recycling` JSONC block): carries a *fixed* deflation/recycle subspace from one shifted system to the next within an anchor block and feeds it to the right-preconditioned GMRES, so consecutive shifts reuse spectral structure. The recycle subspace is prepared once per shift solve and never adapted mid-solve, so the solve remains a linear map and the S7 `deterministic_tol_linearity` certification check still passes on the recycled path (measured defects ~1e-12 against a 1e-8 tolerance). `SweepResult.metrics` reports total matvec and GMRES-iteration counts with and without recycling plus the delta. The default path (recycling off) is unchanged. Demonstrated on a small representative case (6×6 operator, 4 frequencies, recycle dim 3): matvecs 1837 → 1513 (−17.6%, including 18 setup matvecs), gains matching an independent direct MUMPS oracle to ~1.5e-13; this is a small-case illustrative measurement and carries no production-scale speedup claim. Recycling and compressed anchors are not combined in this evaluation path. (bead resolvent-xc-program-v0-1-0-r3uu.14.10)
- Opt-in compressed-anchor option for the stable resolvent sweep (`compressed_anchor` JSONC block): routes the anchor factorization through the STRUMPACK backend with BLR/HSS rank compression via the existing `linalg` seam, with the S7 adjoint-of-approximate-factor certification (check #3) wired to the *compressed* factor's own `solve_adjoint` (not the exact-LU assumption), and the measured factor-compression ratio + effective rank reported on `SweepResult.metrics`. RECORDED FINDING (measured on real STRUMPACK): the compressed-anchor path is **not currently usable in the certified sweep** — STRUMPACK's transpose-solve is a non-conjugate transpose for complex operators, so the adjoint-identity defect is ~3.2e-2 (vs ~3e-16 for MUMPS), five orders above the certification tolerance, and it is *identical with BLR compression on and off* — a STRUMPACK complex-adjoint property, not a compression artifact. The sweep therefore correctly refuses (raises) rather than silently using an uncertified factor; the compression-vs-memory hypothesis cannot be evaluated in the certified sweep until this prior backend-adjoint blocker is resolved. The default path (MUMPS-exact) is unchanged. (bead resolvent-xc-program-v0-1-0-r3uu.14.9)

### Changed
- 2-D LNSE block assembly (`_build_block_csr`) now builds each ngp×ngp block CSR-natively instead of materializing a COO `(rows, cols, data)` triplet and letting SciPy `coo_tocsr` sort it: blocks with ≤2 contributions are summed as row-scaled CSR terms, and denser blocks are filled directly into precomputed `indptr`/`indices` (one `cumsum` + per-term position scatter, no sort), eliminating the ~200 MB-per-dense-block COO intermediate. The assembled operator is byte-identical — `‖L0_new − L0_old‖ = 0` with identical sparsity on the Blasius smoke and a Case 2 stride-2 case (510k DOF, 23.2M nnz), and the seven Appendix-A oracles (<1e-10) plus the full suite stay green with no re-baseline — so this is a cost-only change. Block assembly is ~10–20% faster on the Case 2 stride-2 case (single-run, timing-noise-dependent; no production-scale speedup claim). The layout is recomputed per call and deliberately *not* memoized across calls — a module cache keyed on object ids / buffer addresses would be unsafe under allocator address reuse (a freed derivative cache's address reused by a different sparsity pattern could silently return a wrong operator). (bead resolvent-rbi6)
- 2-D LNSE boundary-condition application now runs in a single CSC→CSR→CSC format context across the whole recipe instead of converting the operator per BC rule. `apply_bc_recipe`'s SciPy path zeroes the original Dirichlet columns once in CSC, converts to CSR once, then applies all row-zeroing / Neumann row-insertion / penalty-diagonal writes in CSR before a single final CSC conversion — collapsing the redundant `csr_tocsc` round-trips (9 → 3 on the profiled case). The assembled operator is numerically unchanged — guarded ≤1e-10 by the seven Appendix-A oracles and the equivalence/parity harness (full suite green) — and assembly is ~15–20% faster on a Case 2 stride-2 representative case (510k DOF, 23.2M nnz, single rank: ~3.6–3.9 s → ~3.1 s; no speedup claim at production scale). `apply_dirichlet`/`apply_neumann` keep their public CSC-in/CSC-out contracts for direct callers, and `apply_bc_recipe` no longer mutates its input operator in place. (bead resolvent-rbi6)
- Documented the SuperLU_DIST PETSc build recipe (`--download-superlu_dist` plus the METIS/ParMETIS graph-partitioning downloads; the GPU variant adds `--with-cuda`) in `docs/COMPUTE_BACKENDS.md`, noting that `install.sh` does not yet ship a `superlu_dist` build stack so it must be built manually in an isolated `.venv-superlu_dist`. Clarified that SuperLU_DIST has two distinct roles: a live CPU mixed-precision iterative-refinement path (FP32 factor + FP64 refine via `pc_precision single`) and a complex sparse-direct GPU solve that was rejected on the tested CUDA build (order-one residuals). The previous backend status read as a wholesale "rejected"; the rejection applies only to the complex+CUDA sparse-direct path, not to the live CPU mixed-precision role. (bead resolvent-xc-program-v0-1-0-r3uu.1.16)

### Fixed
- Build requirement corrected to `setuptools >= 77.0` (was `>= 68.0`): the PEP 639 SPDX `license = "GPL-3.0-only"` expression in `pyproject.toml` needs setuptools 77+, so a fresh install on setuptools 68–76 previously failed metadata generation (`invalid pyproject.toml config: project.license`).
- STRUMPACK `EPS.setUp` factorization failures traced to static-pivoting `ZERO_PIVOT` now raise `StrumpackFactorizationFailed` and trigger a one-shot MUMPS fallback, with fallback provenance recorded in solver info; the default MUMPS path and unrelated setup failures remain unchanged. (bead resolvent-xc-program-v0-1-0-r3uu.1.43)
- Factor-backend accuracy probe is now role-aware: a backend used as the preconditioner inside an iterative-outer shift-invert solve (`st_solver` in `iterative`/`krylov`/`slepc`/`fgmres_lu`/`strumpack_fgmres`) is no longer rejected on its bare direct-solve residual. The strict `residual_rel_max` (~1e-8) gate is the right acceptance test for a standalone direct solve, but for a preconditioner the outer Krylov/SLEPc convergence is the real gate and a moderate (~1e-6) single-apply residual is healthy — previously this forced users to disable the probe by hand to run a STRUMPACK-preconditioned route. The genuine direct path (`mumps`/`direct`/`lu`, default-MUMPS exemption) keeps the unchanged strict gate; the preconditioner role applies a relaxed sanity ceiling so a catastrophically ineffective preconditioner still fails fast. The preconditioner-role detection reuses the shared `ITERATIVE_EIGEN_SOLVER_TYPES` set so it cannot drift from the solver-config vocabulary. (bead resolvent-xc-program-v0-1-0-r3uu.1.40)
- `EPS.setUp` failures now surface a useful error message. When petsc4py's `CHKERR` collapses a PETSc/STRUMPACK error into a bare `SystemError: ... returned a result with an exception set`, the raised `RuntimeError` now explains that the real PETSc error was emitted to stderr (and how to capture it via stderr redirection or `-error_output_stderr`) instead of reporting the opaque `SystemError`; a `PETSc.Error` now names its integer error code (`ierr`). All existing diagnostic context (ndof, backend, `st_solver`, MUMPS infog) is preserved. (bead resolvent-xc-program-v0-1-0-r3uu.1.41)

## [0.0.7] - 2026-06-13

GPU and alternative-backend groundwork plus the inexact-resolvent sweep
machinery, all on the CPU/MUMPS-validated default path under the frozen-numerics
invariant — no GPU backend is wired or benchmarked. Adds the MUMPS ordering and
BLR knobs, the mixed-precision factorization config surface, per-stack build
isolation, the parity-vs-MUMPS protocol, and the GPU decision docs (FP64 gate,
cuDSS C-API note); the anchor-preconditioned stable-regime sweep with RSVD
certification and EVP-guided band marking; the wavemaker structural-sensitivity
field; run provenance and a per-run metrics schema. Reconciles the package
version to a single `pyproject.toml` source and gives the band-marking guard a
physical omega-plane default.

### Fixed
- Package version source-of-truth reconciled to a single canonical source. `pyproject.toml` is now the only place the version is written, and `linstabpy.__version__` is derived from `importlib.metadata` instead of a hardcoded string that had drifted to `0.1.0` — the reserved release-candidate tag — while the build metadata said `0.0.6.dev0`. `__version__`, `importlib.metadata.version("linstabpy")`, and `linstabpy --version` now always agree with the installed distribution, guarded by a consistency test that checks every surface against `pyproject.toml`. (bead resolvent-sr3l)

### Changed
- 2-D LNSE assembly now applies the analytical `inv(B)` as a direct CSR row scaling (`data *= per-row factors`) instead of materializing a sparse diagonal and multiplying, and the SciPy raw-assembly path builds CSR natively with a single final `tocsc()` (one fewer format conversion). The assembled operator is numerically unchanged — guarded at ≤1e-10 against the validated block reference, with the seven Appendix-A oracles green — and ~26% faster on a small fd6 Blasius representative case (measured median on a shared box; no speedup claim at production scale). (bead r3uu.14.2)
- Solver module docstrings (`solvers/eigenvalue.py`, `spatial_eig.py`, `resolvent.py`, `linsolve.py`) now document the factor-backend seam: path-accurate resolution order (including that `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE` is not consulted on the reference direct eigen path), probe-based availability with loud failure, conservative backend statuses, and the env-only enforcement status of the experimental precision/BLR/ordering knobs; doc-only, behavior proven unchanged by docstring-stripped AST comparison. (bead r3uu.1.9)
- Phase-zero band-marking default `danger_radius` is now a physical omega-plane scale instead of an unrelated linear-solve residual scale. When no `danger_radius` is configured, the guard now uses `0.5 * median` of the successive spacing of the requested (finite, unique) forcing frequencies — flagging a frequency `near-pole` when a resolvent pole lies within about half a sweep cell — replacing the old fallback to `solver.eigen.backend_probe.residual_rel_max` (~1e-8), which was a residual tolerance that effectively never fired in the omega plane. With fewer than two distinct finite frequencies no sweep scale exists and the default is `0.0` (no `near-pole` labels); NaN/inf frequencies are excluded and non-uniform grids use the median successive spacing. Explicit `danger_radius` config keys still override in the same precedence and are still validated as finite non-negative distances; the least-damped `|Im omega_eig|` basis was rejected because it collapses near a stability threshold, exactly when a near-real pole most needs a band. (bead resolvent-awwc)

### Added
- Hybrid MPI + threaded-BLAS placement guidance (`docs/COMPUTE_BACKENDS.md`): documents the deliberate `OMP_NUM_THREADS`/`OMP_PROC_BIND`/`OMP_PLACES` (+ `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`) recipe as `threads_per_rank = physical_cores // ranks`, the explicit non-oversubscription invariant `ranks * OMP_NUM_THREADS <= physical cores` (SMT/HT siblings excluded), when to pick pure-MPI (the validated MUMPS sparse-direct default) vs hybrid (dense-heavy / threaded vendor BLAS), and a copy-paste `mpirun` sanity check with a clear pass rule. Placement guidance only — no measured speedup claim. (bead r3uu.1.3)
- Line-3 GPU FP64 decision gate (`docs/GPU_FP64_GATE.md`): documents why FP64 peak FLOPS is the wrong axis for the sparse-direct shift-invert workload class, a measurable go/no-go criteria table (parity quality via the acceptance-anchored `PARITY_RECORD` tolerances; factor-memory capacity; sustained bandwidth; wall time at equal mode quality; factorization-reuse economics; transfer overhead — every non-measured threshold tagged `[assumed]`, hardware numbers tagged `[vendor spec]`), and the recorded recommendation: Line 3 commits only as a capacity/bandwidth/mixed-precision play, FP64-FLOPS framing rejected, with explicit reopen conditions. (bead r3uu.1.8)
- Parity-vs-MUMPS validation protocol + CPU harness skeleton (`tests/test_parity_vs_mumps.py`): candidate factor backends are gated against the MUMPS reference on a small Blasius temporal EVP with acceptance-anchored tolerances (postcheck residual <= 1e-6, target distance <= 0.11, conjugate-aware `-conj(omega)` matching, matched-eigenvalue distance bounded by the postcheck gate — no backend-specific epsilon), emitting a structured `PARITY_RECORD` with matrix size, nnz, factor memory, wall times, ranks, exact solver options, and subspace/convergence state. Ships a mumps-vs-mumps self-consistency floor plus the CPU hard-path SuperLU_DIST `-pc_precision single` E2E test, which skips loudly on PETSc builds without superlu_dist; single-precision is verified from PETSc option consumption, never from request state. GPU execution deferred. (bead r3uu.1.6)
- MUMPS fill-reducing ordering knob (off/unset by default): `LINSTABPY_MUMPS_ORDERING` selects `amd`/`pord`/`metis`/`qamd`/`auto` via `-mat_mumps_icntl_7` (JSONC `mumps_ordering` carried, env-only enforced for now); unset expresses no opinion — it neither sets nor clears `ICNTL(7)`, preserving values installed by the legacy `mpi.mumps_icntl_7` gate or user CLI options (pinned by a real-wiring preservation test); when set it overwrites external values for that process. Unknown names fail listing allowed values; any non-MUMPS backend (including the PETSc-default path) fails loud. Real-wiring tests read `ICNTL(7)` back from the factored matrix after `ksp.setUp()`. No performance claim. (bead r3uu.1.2)
- MUMPS BLR low-rank compression knob (experimental, off by default): `LINSTABPY_MUMPS_BLR` / `LINSTABPY_MUMPS_BLR_TOL` env knobs (JSONC `mumps_blr` / `mumps_blr_tol` fields validated and carried, env-only enforced for now, like `factor_precision`) set `-mat_mumps_icntl_35 1` and `-mat_mumps_cntl_7 <tol>` for MUMPS factorizations; options persist in the options DB so MUMPS actually consumes them at factorization setup (verified by real-wiring tests against the factored matrix: ICNTL(35)/CNTL(7) read back after `ksp.setUp()`, including a non-stickiness check that an off-request clears leftovers); enabling BLR with any non-MUMPS backend (including the PETSc-default path) fails loud naming the backend and knob. No performance claim. (bead r3uu.1.1)
- Per-stack build isolation in `install.sh`: `--stack <name>` / `LINSTABPY_BUILD_STACK` select side-by-side virtual environments (`.venv-<stack>`), keeping `.venv` exclusively for the validated CPU/MUMPS reference build; non-default stacks hard-refuse to remove, overwrite, or alias (symlink) the CPU venv, `--force` is scoped to the stack's own directory, and `--dry-run` prints the resolved plan without touching the filesystem. The GPU configure branch (`--with-cuda --download-strumpack`) is documented as design-only in `docs/COMPUTE_BACKENDS.md` — no GPU build is claimed. (bead r3uu.1.7)
- Wavemaker / structural-sensitivity utility (`linstabpy.wavemaker`, `stability/sensitivity.py`): Giannetti–Luchini field `zeta = |q_dir| |q_adj| / |<q_adj, q_dir>_W|` from direct and Chu-weighted adjoint modes — the first user-facing consumer of the free matrix adjoint. Pairs modes by `|lambda_dir - conj(lambda_adj)|` with a tolerance gate that fails loud listing the offending eigenvalues, normalizes with an explicit retained inner-product divisor, reshapes Fortran-order per state component, and logs a per-mode pairing table at INFO. The adjoint convention (`adjoint_eigenvalues` returns `W^-1 L0^H W` modes, no extra transform) is pinned by a discriminating test. (bead r3uu.20)
- Public-boundary input validation: `Mesh.create`, `resolvent_analysis`, `eigenvalues`, `adjoint_eigenvalues`, and `spatial_eigenvalues` now reject malformed arguments at the call boundary with typed errors naming the bad argument (non-square or non-sparse operators, mismatched weight-vector lengths, `n_modes < 1`, non-finite frequencies/targets, grids below the SBP stencil minimum for the requested `fd_order`) instead of failing deep inside assembly; error messages never leak filesystem paths. (bead r3uu.3.2)
- Mixed-precision factorization config surface (experimental, inert by default): `LINSTABPY_FACTOR_PRECISION` env knob and JSONC `factor_precision` field accept `double` (default, zero effect on the options DB) or `single`, which is honored only with the `superlu_dist` factor backend (inserts `pc_precision single`) and fails loud with any other backend; the JSONC field is carried in the typed config and will be consumed by the Phase B.4 iterative-refinement work. (bead r3uu.1.4)
- Anchor-preconditioned stable-regime frequency sweep (`solvers.router.stable_sweep`): `sweep_stable_regime` factors the shifted operator only at caller-given anchor frequencies and sweeps neighboring frequencies with right-preconditioned GMRES reusing the nearest anchor factor; forward and adjoint actions share one factorization per anchor, every leading singular triplet passes the `certified_rsvd` gate, and per-frequency iteration counts and residuals are recorded as measured quantities, never gated. (bead r3uu.14.5)
- Solver-stack reproducibility manifest: runtime provenance now records python/numpy/scipy/petsc4py/slepc4py versions, PETSc arch, MUMPS version, and the MPI implementation (failed probes record `unavailable`, never crash a run); the block is echoed at INFO at save time and `scripts/tools/freeze_env.sh` emits a pip-constraints file for release freezes. (bead r3uu.21)
- Per-run metrics schema gate (`metrics_schema`): a single source-of-truth schema for the cost+quality sidecar with `validate_metrics` reporting every missing or ill-typed key at once; sidecars failing validation are written with a `schema_errors` field and a loud warning so the run is visibly quarantined from the figure pipeline. (bead r3uu.7)
- RSVD certification gate (`solvers.router.certification`): `certified_rsvd` runs five algebraic checks before trusting an inexact resolvent action — σ-accuracy-derived solve tolerance, forward/adjoint consistency with realized-residual enforcement, adjoint-of-approximate-factor verification, deterministic linearity probed on both forward and adjoint actions, and multi-seed RSVD reproducibility — and fails loud naming the tripped check. (bead r3uu.14.4)
- Phase-zero EVP-guided band marking (`solvers.router.band_marking`): labels requested forcing frequencies stable / near-pole / smooth by distance to the resolvent poles (`-omega_eig`) before any sweep. (bead r3uu.14.3)
- Config-selectable FGMRES shift-invert path with direct-LU preconditioning, including STRUMPACK backend selection when available.
- Runtime result archives and metrics sidecars now include git, config, host, compute-mode, and MPI-stack provenance.
- CPU/MUMPS default-path parity fixture on a small Blasius case and standalone regression guard for ≤1e-12 baseline drift. (bead r3uu.1.13)

### Changed
- Runtime cost metrics now use `unknown` as the default hardware tag when no hardware tag is configured.

### Fixed
- `install.sh --dry-run` `planned_remove` now mirrors the real venv lifecycle exactly: it reports a removal only when the venv actually exists and would be removed — under `--force` on an existing venv, or a Python-version mismatch that triggers recreate — and reports `none` for an absent venv or in `--check` mode (which never removes). Previously it keyed off `--force` alone, over-reporting a removal for an absent venv or under `--check`, and under-reporting the mismatch-recreate case. Plan-reporting fidelity only, via a single shared `venv_will_be_removed` predicate; real removals were already correctly guarded by `remove_venv_dir`. (bead resolvent-e4s8)
- `linstabpy --version` and `--help` (`python -m linstabpy.driver.cli`) now work on a base install without the optional PETSc/MPI stack: the `linstabpy.driver` package initializer no longer eagerly re-exports `run` from the PETSc/MPI-backed `linstabpy.driver.run`, so the CLI's existing lazy import is no longer defeated at package-import time. Verified by a test that blocks `petsc4py`/`mpi4py` via `sys.meta_path` and asserts neither is imported on the `--version`/`--help` paths. Import the runtime explicitly (`from linstabpy.driver.run import run`) where it is needed; no consumer change required. (bead resolvent-xs4h)

### Deprecated
- The legacy `mpi.mumps_icntl_7` / `LINSTABPY_PETSC_SET_MUMPS_ICNTL_7` boolean is now a deprecated compatibility alias for `mumps_ordering=metis`; it is still accepted (and still installs `ICNTL(7)=5`) but new configs should use the canonical `mumps_ordering` surface (`mumps_ordering: "metis"` / `LINSTABPY_MUMPS_ORDERING=metis`). The `metis`→`ICNTL(7)=5` value is now drawn from a single shared `MUMPS_ORDERING_TO_ICNTL_7` table (new PETSc-free `linstabpy.mumps_options` module) consumed by both the legacy gate and the ordering knob, so the two surfaces can no longer drift. Numerically identical to the prior behavior for every existing config (real-wiring `ICNTL(7)` read-back: legacy alias → 5, explicit `mumps_ordering` overrides the alias, unset leaves the key absent → MUMPS default 7). (bead resolvent-38ww)

## [0.0.5] - 2026-06-11

Stages 2–6 of the clean-code revamp, completing the refactor program under the
frozen-numerics invariant: the `linalg/`, `resolvent/`, `stability/`, `driver/`,
and `operators/metrics` seams, each shipped with deprecation shims and certified
operator-identical. Final gate on the verification workstation: full suite
1058 passed / 5 skipped, examples 6/6, MATLAB parity unchanged; the assembled
operator is bit-identical to the pre-refactor anchor (v0.0.2) by SHA-256 digest.

### Changed
- **Stage 2 (refactor):** new `linstabpy.linalg` package — `factorize()` /
  `Factorization` (was `make_linear_solver` / `LinearSolver`, with
  `.solve_transpose` renamed `.solve_adjoint`), a typed `Backend` registry
  (`select`/`ensure`/`available`), and `shifted_operator` (was private).
  `solvers.linsolve` / `solvers.backends` remain as deprecation shims for one
  minor version.
- **Stage 3 (refactor):** new `linstabpy.resolvent` package —
  `ResolventOperator` in `resolvent/operator.py` and `WeightedNormalOperator`
  plus the gain driver in `resolvent/gain.py`;
  `resolvent_analysis_with_residuals` folded into
  `resolvent_analysis(residuals=True)`. Numerics frozen: gains and modes
  identical to baseline.
- **Stage 6 (test):** metrics extraction certified operator-identical by the
  assembly-block oracle, MATLAB source parity, metamorphic tests, full suite,
  and Martini example parity. (bead r3uu.6.7.2)
- **Stage 6 (refactor):** assembly metrics, stage timers, and CSR derivative
  cache helpers moved into `linstabpy.operators.metrics`; `assembly.py` now
  imports the plumbing while preserving compatibility re-exports. (bead r3uu.6.7.1)
- **Stage 4 (refactor):** new `linstabpy.stability` package unifies the three
  eigenvalue entry points — `eigenvalues` (was `temporal_eigenvalues`),
  `adjoint_eigenvalues`, `spatial_eigenvalues` — on one shared private
  shift-invert core (`_solve_near_target`, moved verbatim from
  `solvers.eigenvalue._eigs_near_target`). Old names in `solvers.eigenvalue` /
  `solvers.spatial_eig` remain as deprecation shims for one minor version;
  top-level `linstabpy.temporal_eigenvalues` / `adjoint_eigenvalues` are
  unchanged and warning-free. Numerics frozen: full suite and spectra
  identical to baseline. (bead r3uu.6.5.1)
- **Stage 4 (refactor):** `utils.stability_acceptance` moved verbatim to
  `linstabpy.stability.acceptance`, colocating the acceptance gate
  (postcheck residual, target distance, conjugate-aware verdict,
  `reject_reason_code` bitmask — all unchanged) with the eigenvalue core.
  Old path remains a deprecation shim for one minor version. (bead r3uu.6.5.2)
- **Stage 5 (refactor):** new `linstabpy.driver` package; the case-pipeline
  orchestration moved verbatim from `results_runtime.compute` to
  `linstabpy.driver.run` (`from linstabpy.driver import run`).
  `results_runtime.compute` now delegates and remains the supported
  entrypoint; PETSc command-line options are initialised before `petsc4py`
  import on the new path as well. Numerics frozen: full suite and spectra
  identical to baseline. (bead r3uu.6.6.1)
- **Stage 5 (refactor):** runtime `.npz` archive I/O isolated in
  `linstabpy.driver.io` (`save_results` — verbatim move of the writer —
  and `load_results`). The provenance/cost key set is unchanged and now
  guarded by a round-trip test; `results_runtime._stage_write_npz` remains
  as a back-compatible wrapper. (bead r3uu.6.6.2)
- **Stage 5 (refactor):** new CLI front door `linstabpy.driver.cli` with a
  `--version` flag; the cache-aware `main()` moved there verbatim.
  `results_runtime` is now a one-minor-version deprecation shim re-exporting
  its previous public surface (`compute` → `driver.run`, `main` →
  `driver.cli.main`, helpers → their new homes) with `DeprecationWarning`;
  all in-repo importers migrated. (bead r3uu.6.6.3)
- **Stage 5 (test):** driver split certified numerics-inert by the full suite,
  driver provenance round-trip, `--version` smoke, and examples parity gate.
  (bead r3uu.6.6.4)

### Verified
- Full test suite on the verification workstation (Intel-MPI): 1058 passed, 5 skipped.
- Examples: 6/6 PASS; MATLAB (Martini) parity: 5 PASS + 1 known SOFT_FAIL (jet), unchanged.
- No-MPI pure-math lane (dissipation gate): 91 passed.
- Cumulative-drift gate: assembled-operator SHA-256 digests bit-identical
  v0.0.2 → 0.0.5 (32/32 arrays, equivalence-harness cases 2D + axy).
- `ruff check` and `ruff format --check`: clean.

## [0.0.4] - 2026-06-10

Stage 1 of the clean-code revamp: SolverConfig centralization, complete and
verified under the frozen-numerics invariant, plus the Case 1 MATLAB parity
pipeline brought to green on the verification workstation for the first time.

### Added
- `docs/DEVELOPMENT.md`: the development and release workflow — two-branch model,
  the granular `0.0.x` tick cadence (a few beads → verify → tag), the merge gate,
  and CHANGELOG discipline. README gained a short "Branching and releases"
  pointer section.
- `config.typed`: a typed, frozen `Config` tree (`LinSolveCfg`, `ResolventCfg`,
  `EigenCfg`, `SpatialCfg`, `MeshCfg`, `MpiCfg`) with a single `Config.load`
  reader that parses JSONC + env once at the boundary, precedence `JSONC > ENV >
  default`. Defaults are byte-faithful to the legacy `_scfg*` surface, guarded
  by a differential oracle test.
- `utils.petsc_core.mumps_icntl_petsc_options(mpi)`: pure helper mapping the
  typed `MpiCfg` ICNTL fields to PETSc option strings (unit-covered).

### Changed
- **Stage 1 complete:** all 79 `_scfg*` production call sites migrated onto the
  typed `cfg.*` tree (solvers, runtime, diagnostics). `utils/_solver_cfg.py` is
  retained as the low-level parsing-primitive layer that `config.typed` builds
  on; its dead policy surface (`setting*`, `linear_solver_type`,
  `eigen_st_solver`, `root_scatter_enabled`, `_mumps_petsc_options`) was
  removed. Numerics unchanged by construction and by oracle: all seven
  first-principles oracle groups and the Martini example parity are identical
  to the pre-refactor baseline.

### Fixed
- `case1.sh`: repo-root resolution broke after the tree reorganization
  (`../..` → `../../..`); the script aborted with "missing Python runtime" on
  any fresh checkout.
- Case 1 parity stability solve stagnated (EPS `nconv=0` for 60+ restarts):
  the EPS now converges on the shift-invert transformed residual at tol 1e-8,
  matching the MATLAB `eigs` semantics that produced the reference
  `spectrum.mat` (mode quality remains gated by the full-operator postcheck
  and the acceptance bitmask).
- Case 1 stability parity pass now runs sponge-free like the MATLAB reference
  run (whose exported sponge aux is identically zero); the native
  `case1_resolvent` sponge remains active for the resolvent pass. Spectrum
  parity vs MATLAB: max rel. error 2.8e-08 over the 100 closest-to-target
  modes (was O(1) with the sponge mismatch).

### Verified
- Verification workstation, dev @ `800cd5c`: full suite **1034 passed /
  5 skipped / 0 failed**, including — for the first time on that machine —
  the Case 1 MATLAB spectrum-parity and resolvent-parity tests actually
  running (resolvent SVD postcheck residuals ~1e-10). Martini example parity
  table byte-identical to the v0.0.1 baseline across all six examples.

## [0.0.3] - 2026-06-08

First stable, locally verified baseline. Replaces the withdrawn 0.0.2 (see note).

### Fixed
- Single-rank (no-MPI) execution path: the runtime test stubs now provide an
  `allreduce` collective with `SUM`/`MAX`/`MIN` reduction-op sentinels and a
  PETSc `Options` stub, so the cost-metrics MPI reduction works correctly when
  `mpi4py`/`petsc4py` are absent. This restores the pure-math (no-MPI) test lane.

### Changed
- Repository-wide formatting and lint cleanup (`ruff format` plus selected lint
  fixes). A frozen validated-math file keeps a scoped `F841` ignore via config so
  its body is never edited.
- Build: add `g++` to the container image so the PETSc C++ configure step
  succeeds.
- Paused per-push continuous integration and dependency-bump automation while the
  codebase stabilizes; verification is performed locally (full suite plus the
  no-MPI pure-math lane). A leaner CI will be reintroduced at public release.

### Verified
- Full test suite: 749 passed, 14 skipped.
- No-MPI pure-math lane (dissipation gate): 91 passed.
- `ruff check` and `ruff format --check`: clean.

> **Note on 0.0.2.** A `0.0.2` tag was briefly published on a commit whose CI was
> red, then withdrawn. It was never a real release. `0.0.3` replaces it with a
> clean tag that is cut once at the verified baseline and never moved.

## [0.0.1] - 2026-06-04

Initial tagged baseline. It carried known issues that were resolved in later
work — an MPI vector that was replicated across ranks instead of distributed, an
SVD resolvent-postcheck residual that was not energy-norm consistent, and a
relocated run-kit path. Prefer `0.0.3` or later.

[Unreleased]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc7...HEAD
[1.0.0rc7]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc6...v1.0.0rc7
[1.0.0rc6]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc5...v1.0.0rc6
[1.0.0rc5]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc4...v1.0.0rc5
[1.0.0rc4]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc3...v1.0.0rc4
[1.0.0rc3]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc2...v1.0.0rc3
[1.0.0rc2]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc1...v1.0.0rc2
[1.0.0rc1]: https://github.com/openfluids/linstabpy/releases/tag/v1.0.0rc1
[0.0.5]: https://github.com/openfluids/linstabpy/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/openfluids/linstabpy/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/openfluids/linstabpy/compare/v0.0.1...v0.0.3
[0.0.1]: https://github.com/openfluids/linstabpy/releases/tag/v0.0.1
