# linstab2d

Sparse linear stability and resolvent analysis for compressible viscous flows.

## Motivation

Stability and resolvent analysis of 2D base flows requires solving large sparse linear systems built from the linearized compressible Navier-Stokes equations (LNSE). MATLAB's `backslash` operator uses dense LU factorization — O(n²) memory — which makes problems above ~50k DOFs impractical: a 150k-DOF system needs ~180 GB of RAM, crashing even high-memory workstations. Realistic 2D base flows from LES or DNS data routinely produce 10⁵–10⁶ DOFs.

`linstab2d` is built on sparse matrices from the ground up. It uses SciPy sparse solvers today and targets PETSc/SLEPc for distributed computing on problems with millions of DOFs. No MATLAB license required.

## What it does

`linstab2d` constructs the LNSE operator on structured 2D meshes using summation-by-parts (SBP) finite difference schemes, then solves for resolvent gains, eigenvalues, or optimal forcing/response modes:

- **Resolvent analysis** — optimal gain, forcing, and response mode shapes via singular value decomposition of the resolvent operator
- **Temporal stability** — direct and adjoint eigenvalues of the LNSE operator
- **Spatial stability** — streamwise wavenumber eigenvalues via the companion-matrix generalized eigenvalue problem, with Floquet ansatz for spanwise-periodic flows
- **Coordinates** — Cartesian, cylindrical (axisymmetric), and curvilinear (general mesh deformation with full Jacobian)
- **Boundary treatment** — Dirichlet conditions, sponge layers, mesh masking for complex geometries
- **Energy norm** — Chu compressible energy norm for physically meaningful gain definitions

## Examples

Five validated examples reproduce the cases from Martini & Schmidt (2024):

| # | Example | What it demonstrates |
|---|---------|----------------------|
| 1 | **Blasius boundary layer** | Resolvent analysis, Tollmien-Schlichting wave amplification |
| 2 | **Turbulent jet** | Axisymmetric (cylindrical) resolvent, Kelvin-Helmholtz instability |
| 3 | **Streaky boundary layer** | Temporal + spatial eigenvalues, Floquet ansatz for periodic base flows |
| 4 | **Open cavity** | Mesh masking, direct/adjoint eigenvalues, Rossiter modes |
| 5 | **Paraboloid body** | Curvilinear coordinates, streak receptivity on axisymmetric geometry |

Run any example:

```bash
python linstab2d/examples/example_1_blasius.py
```

Results are cached to `.npz`; subsequent runs plot directly without recomputing.

## Quick start

```bash
# Clone and set up
git clone https://github.com/ricardofrantz/linstab2d.git
cd linstab2d
./install.sh          # creates .venv, installs dependencies via uv

# Activate and run
source .venv/bin/activate
python linstab2d/examples/example_1_blasius.py
```

The Blasius example takes ~11 minutes on a single core (150k × 150k sparse operator, ~30 nnz/row).

### Dependencies

- Python >= 3.10
- NumPy, SciPy, Matplotlib
- scikit-umfpack (optional — faster sparse LU)

All dependencies are installed by `install.sh` via [uv](https://docs.astral.sh/uv/).

### Memory safety guards

For large problems, direct LU factorization can be memory-intensive. Safety guards are enabled by default:

| Variable | Default | Description |
|----------|---------|-------------|
| `LINSTAB2D_LU_SAFE_MODE` | `1` | Block direct LU above configured thresholds |
| `LINSTAB2D_LU_SAFE_MAX_ROWS` | `25000` | Maximum matrix dimension for direct LU |
| `LINSTAB2D_LU_SAFE_MAX_NNZ` | `4000000` | Maximum non-zero entries for direct LU |
| `LINSTAB2D_LU_FORCE_DIRECT` | `0` | Override safety guards |
| `LINSTAB2D_LU_BACKEND` | `auto` | Set to `iterative` for GMRES+ILU path |

## Architecture

```
Mesh → BaseFlow → Operator → BCs → Solver
```

```
linstab2d/
├── mesh/                # SBP operators, structured grids, integration weights, filters
├── baseflow/            # Base flow loaders (Blasius, jet, cavity, ...), Sutherland viscosity
├── operators/           # LNSE coefficient matrices, assembly, spatial operators, Chu energy norm
├── boundary_conditions/ # Dirichlet BC enforcement
├── solvers/             # Resolvent, temporal/spatial eigenvalue solvers, linear solver config
├── examples/            # 5 runnable examples (start here)
└── utils/               # Plotting, Floquet reconstruction
```

Test suite: 270+ tests (`pytest tests/`), covering mesh construction, operators, boundary conditions, solvers, and full integration.

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Core (SciPy) | Done | All 5 examples validated, 270+ tests |
| PETSc/SLEPc | Next | Distributed sparse solvers for O(10⁶) DOF problems |
| Solver optimization | Planned | Push matrix-forming efficiency to its limits |
| Matrix-free Krylov | Future | Time-stepper approach for 3D problems |

## Citing

`linstab2d` implements the theoretical framework described in:

> Martini, E., Schmidt, O. (2024). Linstab2D: stability and resolvent analysis of compressible viscous flows in MATLAB. *Theor. Comput. Fluid Dyn.* [doi:10.1007/s00162-024-00706-0](https://doi.org/10.1007/s00162-024-00706-0)

## License

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)
