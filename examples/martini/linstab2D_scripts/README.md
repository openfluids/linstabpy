# LinStab2D parity export scripts (linstabpy)

Edited MATLAB drivers that write **trimmed** `.mat` bundles containing every field
needed by `compare_martini_examples.py` and `example_*_plots.py`. Upstream
LinStab2D solver code lives in `$LINSTAB2D_ROOT` (not duplicated here).

## Regeneration pipeline

From the repo root (requires MATLAB + `LINSTAB2D_ROOT`):

```bash
export LINSTAB2D_ROOT=/path/to/LinStab2D
bash examples/martini/regenerate_martini_fixtures.sh
```

That runs: MATLAB export → extract (staging) → split into `LinStab2DResults/`.
Then refresh Python outputs and compare:

```bash
.venv/bin/python examples/martini/compare_martini_examples.py
bash run_all_examples.sh --fresh
```

Raw `.mat` → `martini_examples/` (gitignored). Fixtures → `LinStab2DResults/`.

## Field contract (what each export must contain)

Keys below are under the `results_*` struct saved in each `.mat` file. After
NPZ extraction they appear as `results_Foo/bar` (or `results_Foo__bar`).

### Example 0 — `example_0_TestCurvilinearDerivative_results`

| Field | Used by |
|-------|---------|
| `x`, `y`, `f`, `df`, `df_an` | parity (`curvilinear_protocol`), `example_0_plots.py` |

### Examples 1, 2, 5 — resolvent (`*_Blasius`, `*_Jet`, `*_Paraboloid`)

| Field | Used by |
|-------|---------|
| `gains` | parity (`compare_resolvent_case`), plot gain bars |
| `responses`, `forces` | parity mode L₂, mode mosaics (first **3** columns) |
| `plot_grid/X`, `plot_grid/Y`, `plot_grid/usedInd` | `matlab_component_grid` in plotters |
| `idx/{u,v,w,T,rho}_{i,j}` | component extraction for mosaics |
| `omega`, `nEig` | metadata |
| `St` (Ex. 2 only) | jet metadata |
| `mode_plot_ylim` (Ex. 1 only) | presentation |

Mode columns saved: **`martini2024_parity_constants().N_RESOLVENT = 3`**
(matches `martini_cases.yaml` `n_compare` and plotter `N_MODES=3`).

### Example 3 — paper campaigns + bundled parity

**Paper stems** (thesis / `example_3_plots.py`):

| `.mat` stem | Fields |
|-------------|--------|
| `example_3_paper_*_results_temporal` | full `omega`; `V` (**50** cols, all `nEig`); `plot_grid`; `idx`; `alpha_tempAn`, `Lx`, `Ly`, `nPeriods`, `floquetExp`, `Nx`, `Ny`, `time_eigsMatlab` |
| `example_3_paper_spatial_Lxpi_nP2_results_spatial` | `alphas`; `alpha_target`, `omega_driver`, `iplot`, `wi`; `plot_grid`; `idx`; `Lx`, `nPeriods`, `floquetExp` |
| `example_3_paper_spatial_Lxpi_nP2_results_temporal` | same temporal fields (from spatial driver run) |

**Bundled parity stem** (`run_ex3_bundled_parity.m` → `compare_martini_examples` Ex. 3):

| `.mat` stem | Fields |
|-------------|--------|
| `example_3_Streaky_BL_results_temporal` | same as paper temporal |
| `example_3_Streaky_BL_results_spatial` | same as paper spatial |

Temporal mode columns: **all 50** (`nEig` in the driver). Parity compares the best **10**
(`martini_cases.yaml` `n_compare=10`); plotters pick the leading **3**. **Not saved:**
`floquet_extX/Y/V` (visualization-only bloat).

### Example 4 — cavity

| `.mat` stem | Fields |
|-------------|--------|
| `example_4_Cavity_results_direct` | full `omega`; `V` (**6** cols); `iplot`, `omega_target`; `plot_grid`; `idx` |
| `example_4_Cavity_results_adjoint` | `omegaAdj`; `VAdj` (**6** cols); `iplot_adj` |

Mode columns: **`N_CAVITY = 6`** (parity `n_compare=6`; plots use 3).

### Intentionally omitted (regenerate from LinStabPy / baseflows instead)

- Full mesh operators (`mesh/Dx`, `W`, `B`, `C`, …)
- Example 3 / 4 `results_*_base` mesh dumps
- Example 3 Floquet extended reconstruction grids
- Log `.mat` files (not read by parity or plotters)

## Requirements

- **MATLAB** with `eigs`
- **`LINSTAB2D_ROOT`** — LinStab2D tree with `aux_matlab/` and `BaseFlows/*.m`
- **`examples/martini/baseflows/`** — `.mat` data (symlinked as `BaseFlows/` at run time)

## Trim policy summary

| Case | Modes kept | Dropped |
|------|------------|---------|
| 0 | all protocol fields | mesh operators |
| 1, 2, 5 | 3 | mesh operators |
| 3 temporal | 50 (+ full ω spectrum) | `floquet_ext*` |
| 3 spatial | all α | mode vectors |
| 4 | 6 | mesh operators |
