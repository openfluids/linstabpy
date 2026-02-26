# linstab2d

Sparse linear stability and resolvent analysis for compressible viscous flows.

## Motivation

Stability and resolvent analysis of 2D base flows requires solving large sparse linear systems built from the linearized compressible Navier-Stokes equations (LNSE). MATLAB's `backslash` operator uses dense LU factorization — O(n^2) memory — which makes problems above ~50k DOFs impractical: a 150k-DOF system needs ~180 GB of RAM, crashing even high-memory workstations. Realistic 2D base flows from LES or DNS data routinely produce 10^5-10^6 DOFs.

`linstab2d` is built on sparse matrices from the ground up, with all solvers backed by PETSc/SLEPc and MUMPS. Python handles matrix assembly and orchestration; the heavy linear algebra runs through compiled C/Fortran libraries optimized for large sparse systems. No MATLAB license required.

## What it does

`linstab2d` constructs the LNSE operator on structured 2D meshes using summation-by-parts (SBP) finite difference schemes, then solves for resolvent gains, eigenvalues, or optimal forcing/response modes:

- **Resolvent analysis** — optimal gain, forcing, and response mode shapes via SVD of the resolvent operator (SLEPc TRLANCZOS)
- **Temporal stability** — direct and adjoint eigenvalues of the LNSE operator (SLEPc Krylov-Schur + shift-invert)
- **Spatial stability** — streamwise wavenumber eigenvalues via the companion-matrix generalized eigenvalue problem, with Floquet ansatz for spanwise-periodic flows
- **Coordinates** — Cartesian, cylindrical (axisymmetric), and curvilinear (general mesh deformation with full Jacobian)
- **Boundary treatment** — Dirichlet conditions, sponge layers, mesh masking for complex geometries
- **Energy norm** — Chu compressible energy norm for physically meaningful gain definitions

## Examples in depth

Five validated examples reproduce the cases from Martini & Schmidt (2024). Each example highlights different numerical techniques — this section explains what happens under the hood and why.

Run any example:

```bash
python linstab2d/examples/example_1_blasius.py
```

Results are cached to `.npz`; subsequent runs plot directly without recomputing.

---

### Example 1: Blasius boundary layer — resolvent analysis

**Physics:** A compressible flat-plate boundary layer (Re = 6 x 10^5, Ma = 0.3) amplifies disturbances through the Tollmien-Schlichting mechanism. Resolvent analysis identifies the frequency and spatial structure of the most amplified perturbations.

**Numerics:**

| Component | Choice | Why |
|-----------|--------|-----|
| Discretization | SBP-FD, order 4, 220 x 70 grid | SBP ensures discrete energy stability; 4th order balances accuracy and stencil width |
| Mesh deformation | Hanifi algebraic y-mapping | Clusters points near the wall where gradients are steepest, stretches to freestream |
| Linear solve | PETSc KSP + MUMPS (direct LU) | Single factorization of `L0 - i*omega*I`, reused for both forward and adjoint resolvent matvecs |
| SVD | SLEPc TRLANCZOS (thick-restart Lanczos bidiagonalization) | Purpose-built for computing a few leading singular triplets of large operators; avoids squaring the condition number |
| Energy norm | Chu norm with quadrature weights | Physically meaningful gains: accounts for compressibility via density and temperature weighting |
| Boundary treatment | Sponge (tanh ramp at inflow/outflow/freestream) + Dirichlet on u, v, w, T | Sponge absorbs reflections without altering the interior spectrum |

**What makes it interesting:** The resolvent operator `H = C inv(L_sigma) B` is never formed as a matrix. Instead, a PETSc MatShell wraps the MUMPS factorization: each SVD iteration applies `H` via a forward solve and `H^H` via a conjugate-transpose solve. This means a 150k x 150k system requires only the memory of one sparse LU factorization, not a dense matrix.

**Key output:** Resolvent gains sigma_1 >> sigma_2 >> sigma_3, confirming rank-1 amplification characteristic of convective instabilities.

---

### Example 2: Turbulent jet — axisymmetric resolvent

**Physics:** A subsonic turbulent jet (St = 0.39) exhibits Kelvin-Helmholtz instability in the shear layer. Resolvent analysis in cylindrical coordinates captures the wavepacket structure responsible for jet noise.

**Numerics:**

| Component | Choice | Why |
|-----------|--------|-----|
| Coordinates | Cylindrical axisymmetric (z, r) | Natural for round jets; azimuthal Fourier decomposition (m = 0) eliminates one spatial dimension |
| Base flow | Interpolated from MATLAB table (LES/experimental data) | Realistic mean profile, not an analytical model |
| Sponge | Loaded from MATLAB `.mat` file and interpolated to mesh | Matches the validated reference configuration exactly |
| Chu norm | Includes radial weighting (r * W_quad) | Correct energy inner product in cylindrical coordinates |
| SVD | Same SLEPc TRLANCZOS as Example 1 | |

**What makes it interesting:** The cylindrical LNSE operator includes 1/r singularity terms at the axis. The y_symmetry boundary condition handles the r = 0 pole, and the Chu norm's radial weighting ensures gains represent physical energy amplification (not just a norm artifact).

---

### Example 3: Streaky boundary layer — temporal and spatial eigenvalues

**Physics:** A boundary layer perturbed by streamwise streaks develops secondary instabilities. Both temporal (fixed alpha, find omega) and spatial (fixed omega, find alpha) eigenvalue problems reveal the instability mechanism.

**Numerics:**

| Component | Choice | Why |
|-----------|--------|-----|
| Periodicity | x-periodic domain with Floquet ansatz | The streaky base flow is periodic in x; Floquet parameter alpha_F sets the subharmonic wavenumber |
| Temporal EVP | SLEPc Krylov-Schur + shift-invert (MUMPS) | Standard approach: factor `(iL0 - sigma*I)`, search near target omega |
| Spatial EVP | SLEPc GNHEP (generalized non-Hermitian) + shift-invert | Companion-matrix linearization: `[Lw, Lc; I, 0] v = alpha * [R, 0; 0, I] v` |
| Sponge | Top boundary only (y-direction tanh ramp) | Periodic x-direction needs no sponge; freestream damping prevents spurious reflections |

**What makes it interesting:** The spatial eigenvalue problem is a *generalized* EVP (two matrices), not a standard one. The companion-matrix formulation converts the quadratic eigenvalue problem in alpha into a linear GNHEP twice the size. SLEPc handles this natively with `setOperators(A, B)` and `setProblemType(GNHEP)`.

**Key output:** Temporal growth rates and spatial amplification rates, directly comparable to published streak instability results.

---

### Example 4: Open cavity — direct and adjoint global modes

**Physics:** Flow over an open cavity (Rossiter modes) exhibits global instability. Computing both direct and adjoint eigenmodes reveals where perturbations grow (direct) and where the flow is most receptive to forcing (adjoint).

**Numerics:**

| Component | Choice | Why |
|-----------|--------|-----|
| Geometry | Mesh masking (remove points outside cavity walls) | No need for body-fitted grids; structured mesh with masked regions handles the L-shaped domain |
| Mesh deformation | Cubic + power-law stretching in both x and y | Concentrates resolution near cavity edges (shear layer) and stretches the far field |
| Direct EVP | SLEPc Krylov-Schur + shift-invert | Standard temporal eigenvalue search near target omega |
| Adjoint EVP | Same solver on `W^{-1} L0^H W` | The W-weighted adjoint operator ensures biorthogonality in the energy norm |
| Dense fallback | NumPy `eig()` for small matrices (n < 2500) | On coarse grids the problem is small enough that dense eigendecomposition is faster than iterative |

**What makes it interesting:** The adjoint eigenmodes are localized upstream of the cavity leading edge — physically, this is where external forcing most efficiently triggers the Rossiter cycle. The eigenvalue relation omega_adj = i * conj(omega_direct) provides a built-in consistency check between direct and adjoint spectra.

---

### Example 5: Paraboloid body — curvilinear resolvent

**Physics:** A paraboloid-shaped body with azimuthal mode m = 5 exhibits streak receptivity. Resolvent analysis in curvilinear coordinates on the body-fitted mesh reveals optimal forcing patterns.

**Numerics:**

| Component | Choice | Why |
|-----------|--------|-----|
| Coordinates | Cylindrical axisymmetric with conformal mapping | Z^2 algebraic mapping (complex plane) maps a rectangular grid onto the paraboloid surface |
| Azimuthal mode | m = 5 (Fourier decomposition) | Higher azimuthal wavenumbers probe streak-like instabilities |
| Singular frequency | omega = 0 with fallback to omega = 10^{-4}i | At omega = 0 the shifted operator `L0 - i*omega*I = L0` may be singular; a small imaginary shift regularizes it |
| SVD | SLEPc TRLANCZOS via MatShell | Same matrix-free approach as Examples 1 and 2 |

**What makes it interesting:** The conformal Z^2 mapping is an elegant way to generate a body-fitted mesh from a simple rectangular grid. The Jacobian of the mapping enters the LNSE coefficients automatically through the curvilinear operator assembly. No special treatment of the body surface is needed beyond the coordinate transformation.

---

### Solver stack summary

All five examples share the same solver backend:

```
SciPy sparse (matrix assembly)  →  PETSc (linear solves via MUMPS)  →  SLEPc (eigenvalue/SVD problems)
```

| Solver task | Library | Algorithm |
|-------------|---------|-----------|
| Sparse LU factorization | PETSc KSP + MUMPS | Multifrontal direct solver with METIS ordering |
| Temporal/adjoint eigenvalues | SLEPc EPS | Krylov-Schur with shift-invert spectral transformation |
| Spatial eigenvalues (GNHEP) | SLEPc EPS | Krylov-Schur on generalized non-Hermitian problem |
| Resolvent SVD | SLEPc SVD | Thick-restart Lanczos bidiagonalization (TRLANCZOS) |
| Implicit filter factorization | PETSc KSP + MUMPS | Direct LU of tridiagonal filter matrices |

**Why PETSc/SLEPc instead of SciPy?**

- **MUMPS** provides fill-reducing orderings (METIS) that dramatically reduce memory for sparse LU compared to SciPy's SuperLU.
- **SLEPc Krylov-Schur** is more robust than ARPACK for non-Hermitian eigenvalue problems near interior targets.
- **SLEPc TRLANCZOS** computes the resolvent SVD directly (no operator squaring), which is numerically more stable than eigendecomposing H^H W H.
- The entire solver chain is MPI-ready for future distributed-memory parallelism.

## Linear-system nature (Hamada Example 6) and solver strategies

For the assembled LNSE operator (after BCs), the current Hamada case has:

- size `127500 x 127500`
- `5,678,509` nonzeros (`0.0349%` density)
- strongly non-symmetric/non-normal behavior
- low global diagonal dominance
- large diagonal scale spread (especially from BC-penalty rows)

This is not a classic incompressible saddle-point matrix (`[A B^T; B 0]`).  
The block system is compressible and mostly coupled in `rho/u/v/T`, with a decoupled `w` block for the 2D (`beta=0`) case.

### Strategy A (robust baseline): MUMPS direct

- `ST=SINVERT`, `KSP=PREONLY`, `PC=LU`, `solver=MUMPS`
- use for medium cases where direct factorization memory fits
- this is the current reliable MATLAB-parity path in this repository
- verified on **2026-02-26** (`example_6_hamada`, `n_modes=30`, `MPI=1`, `threads=6`): converged in `~98s`, parity `pass=True`

### Strategy B (scalable, in progress): PETSc iterative + strong preconditioner

- `ST=SINVERT` or `ST=PRECOND` (JD route), with strict `ksp_error_if_not_converged`
- move from generic ASM to stronger block preconditioning (`PCFIELDSPLIT`) using physics blocks (`rho/u/v/T` and `w`)
- add scaling before solve to reduce BC-penalty conditioning issues
- current tested status on the Hamada case:
- `MPI=1` iterative ASM (`GMRES+ASM+sub-LU`) converges and matches MATLAB, but slower (`~217s`)
- `MPI=2` iterative SINVERT (`GMRES+ASM+sub-LU`) now converges with parity pass (`~185s`, `n_modes=30`) using root-owned matrix transfer path
- `MPI=2` JD+PRECOND (`FGMRES+bjacobi+sub-LU`) did not reach convergence in `~320s` (`nconv=0`) and was terminated
- `MPI=2` `PCFIELDSPLIT` variants are not yet robust on this case and remain experimental

Operational rule:

- if direct factorization fits memory, use Strategy A
- if not, use Strategy B only with explicit convergence reason + parity checks
- reference logs from these tests: `artifacts/logs/example_6_hamada_20260226_085632.log`, `artifacts/logs/example_6_hamada_20260226_085247.log`, `artifacts/logs/example_6_hamada_20260226_092026.log`, `artifacts/logs/example_6_hamada_20260226_115755.log`

### Validated discovery (2026-02-26)

- For the Hamada parity case (`example_6_hamada`, `n_modes=30`), both strategy classes are now validated for temporal stability:
- Strategy A (direct MUMPS) converges and matches MATLAB spectrum.
- Strategy B baseline (iterative SLEPc SINVERT + GMRES/ASM/sub-LU) converges and matches MATLAB spectrum in distributed mode (`MPI=2`), after enabling root-owned SciPy->PETSc row-scatter transfer.
- Matrix-stage parity (pre/post BC assembly stages) is now emitted as an artifact and passes for the parity case.

### Open plan (resolvent path)

- Resolvent is still treated as an open hardening track.
- The new distributed iterative root-scatter path is validated for temporal EVP; equivalent robustness/performance validation for resolvent (gains/modes parity and distributed scalability) remains to be completed before calling Strategy B resolvent-ready.

### Project expansion roadmap

- Extend Strategy B root-scatter assembly from temporal EVP to resolvent pipeline and validate gains/modes parity against MATLAB exports.
- Apply the same scalable transfer pattern to spatial EVP (`Lc`, `R`, `Lw`) to avoid full-matrix per-rank paths in distributed runs.
- Add Linux/Xeon production benchmark profile (`MPI x threads`) with fixed convergence/parity/memory gates and published baseline logs.
- Promote one stable iterative profile to default only after parity + convergence + memory gates pass across temporal, spatial, and resolvent workflows.

## Quick start

```bash
# Clone and set up
git clone --recurse-submodules https://github.com/ricardofrantz/linstab2d.git
cd linstab2d
./install.sh          # creates .venv, installs dependencies via uv
# If submodules were not cloned with --recurse-submodules:
git submodule update --init --recursive

# Activate and run
source .venv/bin/activate
python linstab2d/examples/example_1_blasius.py
```

### Linux one-command install (recommended)

For most Linux users, setup is:

```bash
bash install.sh
```

`install.sh` handles:
- system prerequisites (compiler/MPI/X11 toolchain) via detected package manager (`apt`, `dnf`, `yum`, `pacman`, or `zypper`)
- Python environment creation with `uv` (`.venv`, Python 3.13)
- PETSc/SLEPc + `petsc4py`/`slepc4py` installation and verification
- editable install of `linstab2d` and test tools

If your machine is already provisioned (or you cannot install system packages), skip the system-deps step:

```bash
LINSTAB2D_SKIP_SYS_DEPS=1 bash install.sh
```

### Dependencies

- Python >= 3.10
- NumPy, SciPy (sparse data structures), Matplotlib
- PETSc (complex scalars) + petsc4py
- SLEPc + slepc4py
- MUMPS (bundled with PETSc)

All dependencies are installed by `install.sh` via [uv](https://docs.astral.sh/uv/).

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

## Reference MATLAB source boundary

- **Project implementation:** the Python code in this repository (`linstab2d/`, `tests/`, tooling files) is the source of this package.
- **Reference source:** the MATLAB reference code and `.mat` base-flow files are kept in the git submodule at `linstab2d/data/baseflows`, which points to the upstream MATLAB repository.
- **Local workflow:** `./install.sh` ensures the submodule is initialized so MATLAB files are present on disk for inspection.
- **Boundary rule:** modify/commit only files in this repository unless you intentionally update the submodule itself (rare, with explicit intent).

## Parity status (scoped strict gate)

Current strict pipeline command:

```bash
STRICT_PARITY=1 linstab2d/examples/run_parity.sh
```

Current status:
- examples `0..5`: pass
- parity matrix gate: pass
- symbolic parity gate: pass
- numeric operator parity gate (`--strict`): pass
- perf/memory parity gate (strict ceilings): pass

Scope note:
- strict parity is currently **scoped** by `tools/parity/operator_matrix.json` and `tools/parity/branch_matrix.json`
- branch scope uses explicit `required_exclusions` for combinations not yet required
- to enforce full cross-product branch coverage later, reduce/remove exclusions and validate incrementally

Test suite: 321 tests (`pytest tests/`), covering mesh construction, operators, boundary conditions, solvers, and full integration.

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Core assembly + 5 examples | Done | All 5 examples validated, 321 tests |
| PETSc/SLEPc solvers | Done | MUMPS direct solves, SLEPc eigenvalue/SVD |
| Solver optimization | Planned | Push matrix-forming efficiency to its limits |
| Matrix-free Krylov | Future | Time-stepper approach for 3D problems |

## Citing

`linstab2d` implements the theoretical framework described in:

> Martini, E., Schmidt, O. (2024). Linstab2D: stability and resolvent analysis of compressible viscous flows in MATLAB. *Theor. Comput. Fluid Dyn.* [doi:10.1007/s00162-024-00706-0](https://doi.org/10.1007/s00162-024-00706-0)

## License

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)
