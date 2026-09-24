# Song / Robinet SWBLI

Base flows and reference eigenvalues from Z. Song, redistributed with permission.
Cite Song & Hao, Physics of Fluids 35:084121 (2023).

## Layout

```
examples/song/
  reference/              # shipped Song reference data (git)
  inputs/raw/             # place grid.grd + mean_flow.dat (not shipped)
  inputs/baseflow/        # nodal .npy from prepare_baseflow.py (gitignored)
  output/fd4_n30/         # EVP NPZ from song.py (a trimmed copy ships; see step 3)
  figures/                # validation plots (shipped)
  configs/run_fd4_n30.yaml
```

## Reference data (`reference/`)

| Path | Contents |
|------|----------|
| `reference/baseflow/song_32deg.npz` | Packed mean-flow grid (599 × 301) for quick checks. |
| `reference/eigenmode/song_32deg_leading_mode.npz` | Song leading eigenmode fields. |
| `reference/spectrum/eigenvalues.dat` | 20 eigenvalues at β = 19.2. |
| `reference/spectrum/omega_i.csv` | Song β–ωᵢ curve. |

See `reference/README.md` for conventions.

## Pipeline (PhD supp workflow, θ = 32°)

Run from the repo root with the project venv active. **Nothing is executed below
by default** — this is the intended order once inputs and MPI are available.

### 1. Prepare baseflow from Song's FV export

Copy Song's Plot3D files into `inputs/raw/`, then:

```bash
.venv/bin/python -m examples.song.prepare_baseflow
```

Reads `grid.grd` + `mean_flow.dat`, drops the first and last streamwise column
(601 → 599), interpolates cell centres to vertices, non-dimensionalises, and
writes `inputs/baseflow/*.npy` (including zero `sponge.npy`).

### 2. Plot mesh and mean flow

```bash
.venv/bin/python -m examples.song.plot_baseflow
```

### 3. Run EVP (FD4, n = 30, β = 19.2)

No SBP operator dissipation (`numerics.operator_dissipation.epsilon: 0`).

```bash
mpiexec -n 4 python -m examples.song.song --config examples/song/configs/run_fd4_n30.yaml
```

Writes `output/fd4_n30/song_2023.npz` (430 MB: thirty eigenvectors on the full mesh).

The repository ships a trimmed copy of that file at the same path, 6.6 MB: the thirty
eigenvalues, the grid, and the leading eigenvector in single precision, which is what
steps 4 and 5 read. Steps 4 and 5 therefore work from a fresh clone. The run above
overwrites it with the full file. The trimmed mode figure is the same as the full one to
one grey level in a few pixels, and the printed eigenvalues are unchanged.

### 4. Match Song's 20 reference modes

```bash
.venv/bin/python -m examples.song.report_matched20
```

Hungarian assignment on |Δω|; prints per-mode pairing and lead-mode error.

### 5. Plot spectrum and lead eigenmode

```bash
.venv/bin/python -m examples.song.plot_spectrum_matched20
.venv/bin/python -m examples.song.plot_eigenmode_lead
```

Figures land in `figures/validation/`; baseflow figures in `figures/baseflow/`.
Curated PNGs are shipped in git; re-run the plot scripts to regenerate after a new EVP.

### Figure prerequisites (LaTeX)

Song validation figures call `examples.figure_style.check_plot_environment(require_latex=True)`.
Install a TeX stack before plotting, e.g. on Debian/Ubuntu:

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng ghostscript
```

Without LaTeX the plot scripts exit with an install hint (matplotlib mathtext is not used
for this example).

## Convention

The 32° baseflow is velocity / c_inf and temperature / T_inf:
u_max = Ma = 2.15, T_min = 1. The nodal grid is 599 × 301. The raw Plot3D grid
is 601 × 301.

## Default run

```bash
mpiexec -n 4 python -m examples.song.song
```

Uses `configs/run_fd4_n30.yaml` (requires `inputs/baseflow/` from step 1).
