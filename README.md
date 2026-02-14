# LinStab2D.py

Python reimplementation of [**LinStab2D**](https://github.com/eduardomartini/LinStab2D), a MATLAB tool for stability and resolvent analysis of compressible viscous flows.

> **Original code by Eduardo Martini and Oliver Schmidt.**
> This Python version is a derivative work of the original MATLAB package,
> licensed under GPL-3.0. All credit for the formulation, algorithms, and
> numerical methodology belongs to the original authors.

## What is LinStab2D?

LinStab2D constructs the linearized compressible Navier-Stokes (LNSE) operator on structured 2D meshes using summation-by-parts (SBP) finite difference schemes. The resulting sparse matrices can then be used for:

- **Resolvent analysis** — optimal forcing/response gains and mode shapes
- **Temporal/global stability** — eigenvalues of the LNSE operator
- **Spatial stability** — streamwise wavenumber eigenvalues at a given frequency

The code handles Cartesian and curvilinear coordinates, Dirichlet boundary conditions, sponge layers, and the Chu energy norm for compressible flows.

## Why a Python version?

The MATLAB original is self-contained and works out of the box. This Python port exists to:

1. **Scale to large problems** — target PETSc/SLEPc for distributed sparse linear algebra on matrices too large for a single workstation (O(10^6) DOFs)
2. **Integrate with the Python ecosystem** — SciPy sparse solvers, NumPy, Matplotlib, and existing post-processing pipelines
3. **Provide a foundation for extensions** — cylindrical coordinates, general mesh deformations, spatial filters, Neumann BCs

## Current status

**Phase 1 is complete.** The Python package reproduces the MATLAB Blasius boundary layer example (resolvent analysis at Re = 6 x 10^5, Ma = 0.3) with validated gains of [128.62, 14.54, 7.49].

| Feature | Status |
|---|---|
| SBP finite difference operators | Done |
| Structured mesh + Hanifi deformation | Done |
| Blasius/jet/cavity/paraboloid base flows | Done |
| Sutherland viscosity law | Done |
| LNSE operator assembly (Cartesian) | Done |
| Dirichlet boundary conditions | Done |
| Chu energy norm | Done |
| Resolvent analysis (SciPy) | Done |
| Temporal/spatial eigenvalue solvers | Done |
| Cylindrical coordinates | Done |
| General mesh deformation (full Jacobian) | Planned |
| PETSc/SLEPc distributed solvers | Planned |
| Mesh masking + Neumann BCs | Planned |
| Spatial filters | Planned |

## Package structure

```
linstab2d/
├── mesh/               # SBP operators, structured grids, integration weights, filters
├── baseflow/           # Base flow loading (Blasius, jet, cavity, ...), Sutherland viscosity
├── operators/          # LNSE coefficient matrices, assembly, Chu energy norm
├── boundary_conditions/# Dirichlet BC enforcement
├── solvers/            # Resolvent, temporal/spatial eigenvalue solvers
├── examples/           # Runnable examples (start here)
└── utils/
```

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/ricardofrantz/LinStab2D.py.git
cd LinStab2D.py
./install.sh          # creates .venv, installs numpy/scipy/matplotlib

# 2. Activate and run
source .venv/bin/activate
python linstab2d/examples/example_1_blasius.py
```

The Blasius example takes ~11 minutes on a single core (150k x 150k sparse operator, ~30 nnz/row). Results are cached to `.npz`; subsequent runs plot directly.

## Dependencies

- Python >= 3.10
- NumPy, SciPy, Matplotlib
- scikit-umfpack (optional, faster sparse LU)

All dependencies are installed by `install.sh` via [uv](https://docs.astral.sh/uv/).

## Citing

If you use this code, please cite the original LinStab2D paper:

> Martini, E., Schmidt, O. Linstab2D: stability and resolvent analysis of compressible viscous flows in MATLAB. *Theor. Comput. Fluid Dyn.* (2024). [https://doi.org/10.1007/s00162-024-00706-0](https://doi.org/10.1007/s00162-024-00706-0)

## License

This project is a derivative of [LinStab2D](https://github.com/eduardomartini/LinStab2D) by Eduardo Martini and Oliver Schmidt, licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).
