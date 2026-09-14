# Examples: order, data, runner, costs

Run in this order. Each example has a matching `example_N_plots.py`
that re-renders figures from the NPZ without recomputing.

## Example 0: curvilinear validation (run first)

Pure NumPy, single process, no data, under a minute. Validates SBP
derivatives on deformed meshes and proves the import chain.

```bash
.venv/bin/python -m examples.martini.example_0_curvilinear_validation
.venv/bin/python -m examples.martini.example_0_plots   # re-plot anytime
```

If this fails, the problem is the Python env, not MPI or data.

## Example 1: Blasius resolvent (first science run)

Compressible Blasius boundary layer, Re = 6.0e5, Ma = 0.3, SBP order 4,
300x100 grid. Needs the MATLAB base flow `BL_BF.mat` (not bundled):

```bash
git clone https://github.com/LinStab2D/LinStab2D.git ~/linstab2d-reference
export LINSTABPY_BASEFLOW_DATA_DIR=$HOME/linstab2d-reference
.venv/bin/python -m examples.martini.example_1_blasius
.venv/bin/python -m examples.martini.example_1_plots
```

Without the data dir the example raises `FileNotFoundError`. Outputs:
`example_1_blasius.npz` (gains, modes, mesh, Chu weights, provenance),
baseflow PNG (look at this before trusting modes), forcing/response
mosaics per component. Case definition with commented knobs:
`examples/martini/example_1_blasius.yaml`.

## Family map

| # | Module | Teaches | Needs |
|---|---|---|---|
| 0 | `example_0_curvilinear_validation` | SBP derivatives on deformed meshes | Nothing |
| 1 | `example_1_blasius` | Resolvent gains, forcing/response | `BL_BF.mat` via data dir |
| 2 | `example_2_jet` | Resolvent on turbulent jet base flow | `M09jet.mat` via data dir |
| 3 | `example_3_streaky_bl` | Temporal + spatial EVP, Floquet options | MATLAB fixtures for plots |
| 4 | `example_4_cavity` | Direct + adjoint eigenvalues, spectra | `Step_BF.mat` via data dir |
| 5 | `example_5_paraboloid` | Resolvent on 3D body-fitted mesh | `Paraboloid_BF.mat` via data dir |

Beyond Martini (week two): `examples/song/` (SWBLI global stability,
ships packed `song_32deg.npz`), `examples/sawtooth/` (ships packed
`sawtooth.npz`), `examples/savarino/` (gain maps via Slurm; fetch with
`fetch_zenodo.py`), `examples/prescribed_forcing/` (small demos).

## Runner

```bash
bash run_all_examples.sh --list
bash run_all_examples.sh --example 1 --skip-parity   # one example, no MATLAB check
bash run_all_examples.sh                             # everything + parity + plots
bash run_all_examples.sh --mpi -n 10 -m examples.martini.example_3_streaky_bl
bash run_all_examples.sh --mpi -n 1 -m examples.martini.example_1_blasius -- -eps_monitor
```

Without `-n` the runner defaults to half the detected physical cores.
Override the launcher with `LINSTABPY_MPIEXEC`, ranks with
`LINSTABPY_MPI_RANKS`. PETSc options go after `--`. Interpreter:
`LINSTABPY_PYTHON` wins over `LINSTABPY_PYTHON_BIN`, else
`.venv/bin/python`.

Parity against MATLAB references:
`.venv/bin/python examples/martini/compare_martini_examples.py`
(missing bundles skip cleanly outside pytest).

## What things cost (measured anchors)

Savarino case, one c9 node (dual Xeon Silver 4210R, 20 cores, MKL),
Aug 2026. Single resolvent point, 450k unknowns, 23.4M nonzeros, MUMPS:
4 ranks 190.5s/18.6GiB, 8 ranks 154.6s/25.4s, 16 ranks 148.2s/40.6GiB —
gains identical to 10 digits. Factorization dominates; more ranks are
not automatically faster. Full 200-point gain map: 83.6 min as
independent jobs vs 59.7 min via the line driver with anchor reuse.
