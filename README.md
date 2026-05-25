# LinStabPy

## 1. Project overview

LinStabPy is a Python package for sparse linear stability and resolvent
analysis of compressible viscous flows.

The package targets two-dimensional structured base flows and builds operators
for the linearized compressible Navier-Stokes equations. The public project name
is `LinStabPy`, the repository and distribution name is `linstabpy`, and the
current import package is `linstabpy`.

The motivation is practical memory use. Stability and resolvent analysis of 2D
base flows leads to large sparse linear systems. MATLAB's `backslash` path uses
dense LU-style memory scaling in the workflows that motivated this port. That
is `O(n^2)` storage: a 150k-DOF system needs about 180 GB of RAM, which makes
many LES/DNS-derived base-flow problems unusable on normal workstations.
LinStabPy keeps the matrices sparse and sends the linear algebra through
PETSc/SLEPc and MUMPS.

The package currently supports:

- resolvent analysis: leading gain, forcing modes, and response modes;
- temporal stability: direct and adjoint eigenvalue problems;
- spatial stability: streamwise-wavenumber eigenvalue problems;
- coordinate systems: Cartesian, cylindrical/axisymmetric, and curvilinear
  structured meshes;
- boundary treatment: Dirichlet rows, Neumann rows, sponge damping fields, and
  masking for structured grids with excluded points;
- energy norms: Chu compressible energy norm with quadrature weighting.

The implementation follows the modelling choices of Martini and Schmidt (2024)
where possible, while replacing MATLAB dense workflows with sparse PETSc/SLEPc
solvers.

## 2. Installation

### 2.1 Prerequisites

Use the repository scripts first. They check compilers, MPI, Python bindings,
PETSc/SLEPc, and the runtime stack that will be used by the examples.

Required system tools:

- POSIX shell with `bash`;
- C, C++, and Fortran compilers;
- `make`, `cmake`, `curl`, and `pkg-config`;
- one MPI stack with development headers (`mpicc`, `mpicxx`, `mpifort`,
  `mpiexec`);
- Python environment managed by the repository installer;
- optional HDF5 development files only when MATLAB v7.3 / HDF5 tooling is
  needed.

Do not mix MPI stacks in one environment. PETSc, SLEPc, `mpi4py`, and the
runtime `mpiexec` must agree on the same MPI ABI.

### 2.2 Linux (recommended path)

Clone the repository and run the native checks:

```bash
git clone https://github.com/ricardofrantz/linstabpy.git
cd linstabpy

bash diagnose.sh
bash install.sh
bash install.sh --check
bash benchmark.sh
```

The reference MATLAB base flows (`M09jet.mat`, `BL_BF.mat`, `Step_BF.mat`,
`Paraboloid_BF.mat`) used by the Martini example loaders are not bundled.
See section 2.8 for how to point linstabpy at a local clone of LinStab2D.

`diagnose.sh` is read-only. It checks compilers, MPI launchers, development
headers, dual-MPI conflicts, and whether `mpiexec -n 1 /bin/true` works.

`install.sh` creates `.venv`, builds or installs the Python/PETSc/SLEPc stack,
repairs `mpi4py` when the wheel ABI does not match the selected MPI stack, and
installs this repository in editable mode.

`install.sh --check` verifies that `petsc4py` and `slepc4py` import, that PETSc
uses complex scalars, and that a small SLEPc non-Hermitian Krylov-Schur solve
works from Python.

`benchmark.sh` runs a small smoke benchmark and writes benchmark artifacts under
`benchmark_runs/<run-id>/`.

By default, the installer does not call `sudo` or a package manager. If required
system tools are missing, it prints the commands and stops. On a normal
workstation, opt in to package-manager installation explicitly:

```bash
LINSTAB2D_AUTO_INSTALL_SYS_DEPS=1 bash install.sh
```

On a provisioned workstation or cluster node, skip the system dependency check:

```bash
LINSTAB2D_SKIP_SYS_DEPS=1 bash install.sh
```

### 2.3 macOS

Run the same repository scripts:

```bash
bash diagnose.sh
bash install.sh
bash install.sh --check
bash benchmark.sh
```

For the macOS PETSc/SLEPc path, install Homebrew OpenBLAS and ScaLAPACK before
building when possible:

```bash
brew install openblas scalapack
```

The reason is a macOS runtime-library ordering issue in Python. PETSc's
downloaded ScaLAPACK can resolve LAPACK symbols to Apple Accelerate when loaded
inside Python. The failing path observed in this repository was Python
`slepc4py` `EPS_NHEP` with Krylov-Schur / Arnoldi, aborting inside Accelerate
`ZLADIV`. Native C SLEPc NHEP worked, and Python HEP/LAPACK paths worked, which
pointed to dynamic-library symbol interposition rather than the examples.

The current installer handles this by preferring Homebrew OpenBLAS/ScaLAPACK at
PETSc configure time and by wrapping `.venv/bin/python` so Homebrew OpenBLAS is
loaded before Python imports NumPy, PETSc, or SLEPc. `install.sh --check` now
runs a tiny Python `EPS_NHEP` Krylov-Schur probe, so this crash is detected
before examples are run.

On macOS, do not use `uv run` to run LinStabPy examples after installation. It
can bypass the patched `.venv` and try to rebuild `petsc4py`/`slepc4py` from
source with the wrong runtime layout. Use `.venv/bin/python` directly:

```bash
.venv/bin/python -m examples.example_1_blasius_smoke
```

### 2.4 Clusters / Intel MPI

On clusters, load the compiler/MPI modules before running the installer. The
repository scripts should see the same `mpicc`, `mpicxx`, `mpifort`, and
`mpiexec` that will be used at runtime.

Generic module workflow:

```bash
module load <compiler-module> <mpi-module>
bash diagnose.sh
bash install.sh
bash install.sh --check
```

Intel MPI / oneAPI workflow:

```bash
source /opt/intel/oneapi/setvars.sh --force
export LINSTAB2D_MPI_STACK=intelmpi
export I_MPI_CC=gcc
export I_MPI_CXX=g++
export I_MPI_F90=gfortran
export I_MPI_FC=gfortran

bash diagnose.sh
bash install.sh
bash install.sh --check
```

Use Intel MPI only when `diagnose.sh` shows that the active MPI compiler
wrappers and `mpiexec` come from the Intel MPI environment and the MPI compile
probe passes. Do not use an Intel MPI launcher with Open MPI or MPICH Python
bindings.

### 2.5 Environment variables

| Variable | Default | Use |
|---|---:|---|
| `LINSTAB2D_WITH_H5PY` | `0` | Set to `1` to install `h5py` for MATLAB v7.3 / HDF5 parity tooling. Leave unset for the normal solver install. |
| `LINSTAB2D_MPI_STACK` | auto | Set to `intelmpi` when an Intel MPI environment is already loaded and should be used. Otherwise the scripts infer the active stack. |
| `LINSTAB2D_AUTO_INSTALL_SYS_DEPS` | `0` | Set to `1` to allow `install.sh` to call the system package manager for missing Linux dependencies. |
| `LINSTAB2D_PETSC_MPIEXEC` | auto | Path to the `mpiexec` PETSc/SLEPc configure should use. Set this when the default launcher hangs or is not the runtime launcher you want. |
| `LINSTAB2D_MPI4PY_NO_BINARY` | `0` | Set to `1` to force `mpi4py` to build from source against the active MPI stack. |
| `LINSTAB2D_VERIFY_MPIEXEC` | auto | Path to the `mpiexec` used by `install.sh --check` for MPI import/probe checks. |
| `LINSTAB2D_SKIP_SYS_DEPS` | `0` | Set to `1` to skip Linux system dependency detection and package-manager advice. |

### 2.6 Known-good reference stacks

A healthy Linux install should satisfy this sequence:

```bash
bash diagnose.sh
bash install.sh --check
bash run_all_examples.sh --mpi -n 1 -m examples.example_1_blasius_smoke
bash benchmark.sh
```

One known-good Linux profile used during hardening was:

| Component | Reference value |
|---|---|
| OS family | Ubuntu 24.04 class |
| MPI | Open MPI 4.1.6 class |
| Python | 3.13 class |
| PETSc/SLEPc scalar type | complex |
| Sparse direct solver | MUMPS through PETSc |
| Thread policy | one BLAS/OpenMP thread per MPI rank |

Use this as a comparison shape, not a hard pin. Export a redacted stack
manifest from a working machine with:

```bash
.venv/bin/python scripts/tools/export_known_good_stack.py \
  --output known_good/<machine-label>.json \
  --redact-local-details
```

### 2.7 MPI runtime rules

Use one MPI stack per install and one thread per rank unless you are measuring a
different layout on purpose.

For manual MPI runs:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Prefer the canonical example runner for MPI examples:

```bash
bash run_all_examples.sh --mpi -m examples.example_1_blasius_smoke
bash run_all_examples.sh --mpi -n 10 -m examples.example_3_streaky_bl
bash run_all_examples.sh --mpi -n 1 -m examples.example_1_blasius_smoke -- -eps_monitor
```

If `-n/--ranks` is omitted in `--mpi` mode, the runner starts from half of the
detected physical cores. Override the launcher with `LINSTABPY_MPIEXEC` and the
default rank count with `LINSTABPY_MPI_RANKS` when the target machine needs an
explicit MPI runtime. For sparse direct factorization, more ranks are not
automatically faster; compare rank counts on the target machine.

### 2.8 Fetching reference base flows

The Martini example loaders (`example_1_blasius`, `example_2_jet`,
`example_4_cavity`, `example_5_paraboloid`) consume four `.mat` base
flows from the upstream LinStab2D project: `M09jet.mat`, `BL_BF.mat`,
`Step_BF.mat`, `Paraboloid_BF.mat`. linstabpy does not bundle these
files. Two options to make them available:

1. Clone LinStab2D once and point linstabpy at it with an env var:

   ```bash
   git clone https://github.com/LinStab2D/LinStab2D.git ~/linstab2d-reference
   export LINSTABPY_BASEFLOW_DATA_DIR=$HOME/linstab2d-reference
   ```

2. Copy the four `.mat` files into the gitignored data dir:

   ```bash
   mkdir -p src/linstabpy/data/baseflows/BaseFlows
   cp ~/linstab2d-reference/BaseFlows/*.mat src/linstabpy/data/baseflows/BaseFlows/
   ```

If neither is in place, the Martini examples raise a `FileNotFoundError`
with the exact instruction to set `LINSTABPY_BASEFLOW_DATA_DIR`. All
non-Martini examples (Case 1 Song, Case 2) ship their own base flows
under `examples/case_studies/` and are unaffected.

## 3. Quick start

```bash
git clone https://github.com/ricardofrantz/linstabpy.git
cd linstabpy

bash diagnose.sh
bash install.sh
bash install.sh --check
bash benchmark.sh
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Run the smallest Blasius example:

```bash
.venv/bin/python -m examples.example_1_blasius_smoke
```

Run a full example:

```bash
.venv/bin/python -m examples.example_1_blasius
```

Do not use `uv run` on macOS for PETSc/SLEPc examples. Use the interpreter
installed and patched by `install.sh`:

```bash
.venv/bin/python -m examples.example_1_blasius_smoke
```

Collect the test suite:

```bash
.venv/bin/python -m pytest --collect-only tests
```

The current expected collection size is `535` tests. Run the suite with:

```bash
.venv/bin/python -m pytest tests/
```

### 3.1 Canonical example runner

Use `run_all_examples.sh` as the single example runner for serial acceptance,
cache cleanup, parity checks, and MPI launches. Direct module execution with
`.venv/bin/python -m ...` is still useful for focused debugging, but the runner
is the documented path for full example validation and MPI-safe launches.
For a figure-by-figure walkthrough of the Martini examples, see
`tutorial/tutorial.pdf`, Section 5. That section also explains how to read the
generated baseflow, gain/spectrum, and support-component figures, and records
the plotting improvements proposed for the next figure pass.

| Task | Command |
|---|---|
| List available examples | `bash run_all_examples.sh --list` |
| Run all examples and Martini parity | `bash run_all_examples.sh` |
| Recompute selected examples from cached PNGs/NPZs | `bash run_all_examples.sh --fresh --example 1 --skip-parity` |
| Remove selected generated NPZ/PNG outputs first | `bash run_all_examples.sh --clean --example 1 --skip-parity` |
| Run one example under MPI | `bash run_all_examples.sh --mpi -n 10 -m examples.example_3_streaky_bl` |
| Pass PETSc/SLEPc options to one example | `bash run_all_examples.sh --mpi -n 1 -m examples.example_1_blasius_smoke -- -eps_monitor` |

`--mpi` folds the old one-module MPI wrapper behavior into the same script:
the runner probes a working MPI launcher, defaults to half of the detected
physical cores when `-n/--ranks` is omitted, and caps OpenMP/BLAS thread counts
to one thread per rank unless the caller already set them. Override the runtime
with `LINSTABPY_MPIEXEC`, `LINSTABPY_MPI_RANKS`, or `LINSTABPY_PYTHON` when a
machine needs explicit paths.

### 3.2 Why JSONC configs?

You may not be familiar with JSONC. It is JSON with comments, and this repo uses
it deliberately even though plain JSON is more common. Each example is a
numerical case definition: Reynolds number, Mach number, grid size, boundary
conditions, sponge settings, forcing masks, solver targets, and PETSc/MPI
choices should be visible, diffable, and explained next to the values.

Plain JSON would remove those explanations. Long command-line flags would hide
too much state in shell history or runner scripts. Python config files would mix
data with executable code. JSONC keeps the inputs close to strict JSON while
still letting the case file document why each parameter exists.

## 4. Repository layout

```text
.
|-- src/linstabpy/                 Importable Python package
|   |-- baseflow/                  Base-flow loaders and viscosity helpers
|   |-- boundary_conditions/       Dirichlet/Neumann row replacement and recipes
|   |-- config/                    Runtime validation metadata
|   |-- data/baseflows/            MATLAB reference assets as a git submodule
|   |-- linear_operators/          Operator adapters for resolvent workflows
|   |-- mesh/                      SBP finite differences, grids, filters, deformation
|   |-- operators/                 LNSE coefficients, assembly, spatial operators, Chu norm
|   |-- solvers/                   PETSc/SLEPc resolvent and eigenvalue solvers
|   |-- utils/                     JSONC, plotting, PETSc helpers, acceptance checks
|   `-- results_runtime.py         Package runtime used by case study drivers
|-- examples/                      Runnable Martini examples; generated .npz/.png ignored
|-- studies/                       Active research workflows, not importable library code
|   |-- case1/                     SWBLI parity/performance study
|   |   |-- configs/
|   |   |-- data/baseflow/
|   |   |-- reference/matlab/
|   |   `-- scripts/
|   `-- case2/                     TCFD campaign configs, logs, figures, notes
|       |-- configs/               Base, EVP, resolvent, and perf JSONC configs
|       |-- data/baseflow/         Active mean-flow arrays
|       |-- scripts/               Plotting and campaign helper scripts
|       |-- artifacts/phase1p5/    Curated phase artifacts
|       `-- docs/                  Case-local plan, reference, and tutorial docs
|-- validation/                    Reproducibility harnesses and compact fixtures
|   |-- martini2024/               Martini parity fixtures, tools, and reports
|   |-- operator_fixtures/         Compact operator-parity fixtures and schema
|   `-- song_2023/                 Song 2023 configs, data, reference, and scripts
|-- tools/                         Developer tools
|   `-- debug/                     Platform/debug probes
|-- scripts/                       Repository guardrails and maintenance utilities
|-- tests/                         Pytest suite and shared path helpers
|-- scripts/ci/                Thin install/diagnose/benchmark harness
|-- tutorial/                      Tutorial source and generated PDF
|-- install.sh                     Main installer
|-- diagnose.sh                    Read-only environment preflight
|-- benchmark.sh                   Smoke benchmark
`-- run_all_examples.sh            Serial, parity, and MPI example runner
```

Keep reusable code under `src/linstabpy/`, runnable examples under `examples/`,
case-specific campaigns under `studies/`, and validation reference data under
`validation/`. Generated example outputs, campaign `output/` directories,
runtime caches, and build metadata are ignored rather than represented as
tracked layout.

The reference MATLAB code and `.mat` base-flow files live in the submodule
`src/linstabpy/data/baseflows`. Treat that directory as reference input unless
you intentionally update the submodule.

### 4.1 Data and file-size policy

Tracked GitHub-facing files must stay at or below 50 MiB. Large generated
outputs, raw MATLAB exports, and machine-local benchmark products are kept out
of Git and either regenerated or stored externally. The guardrail is:

```bash
python scripts/check_tracked_file_size.py
```

The same guard is wired into the local `pre-commit` configuration and the
GitHub guardrails workflow. Ignored local outputs above the report threshold are
listed separately so they can be cleaned or archived without blocking normal
source commits.

Compact validation fixtures belong under `scripts/validation/operator_fixtures/` or
`scripts/validation/martini2024/fixtures/`. Raw MATLAB exports such as
`operator_convention.mat` and `coefficients_raw.mat` are intentionally not
tracked when they exceed the 50 MiB policy.

## 5. Numerical setup

### Temporal stability

The current temporal stability path uses:

- SLEPc `EPS`;
- `KRYLOVSCHUR` as the outer eigensolver;
- shift-invert spectral transformation, `ST=SINVERT`;
- direct LU factorization with MUMPS for shifted linear systems in the default
  production path.

The temporal eigenproblem is formed on `i * L0`, so reported temporal
eigenvalues are `omega = eig(i * L0)`.

### Residual bookkeeping

The code records several residual-like quantities:

- SLEPc EPS residuals from `EPS.computeError(...)`;
- the configured EPS convergence test: `abs`, `rel`, or `norm`;
- the selected reported residual type: `absolute`, `relative`, or `backward`;
- an explicit postcheck residual computed outside SLEPc on returned modes;
- boundary-condition leakage diagnostics.

These quantities are related but not interchangeable. `EPSSetTolerances()`
sets the tolerance and maximum iteration count used by the EPS convergence
test. `EPSSetConvergenceTest()` selects how the EPS convergence test is
measured. `EPSComputeError()` reports an error estimate for a returned
eigenpair, but that estimate is not always the same quantity used internally
during the solve.

For shift-invert runs, `true_residual=true` is useful because cheap transformed
residual estimates can be optimistic for the original problem.

### PETSc KSP tolerances

`KSPSetTolerances()` controls a linear Krylov solve: relative tolerance,
absolute tolerance, divergence tolerance, and maximum iteration count. It is
not an eigenpair tolerance.

In the default direct path, shifted systems use:

```text
KSPPREONLY + PCLU + MUMPS
```

In that configuration, iterative `KSP` tolerances are not the main accuracy
knobs. The dominant knobs are EPS settings, Krylov subspace size `ncv`, target
placement, and the conditioning of the shifted operator.

When an iterative linear-solve path is selected, `ksp_rtol`, `ksp_atol`,
`ksp_max_it`, `pc_type`, subdomain settings, and `ksp_error_if_not_converged`
become meaningful.

### Boundary-condition acceptance

Returned stability modes are postchecked against:

- full-operator residual;
- boundary-condition leakage;
- constrained-Dirichlet leakage;
- target distance;
- optional spectral boxes on `Re(omega)` and `Im(omega)`.

This matters because an interior eigenvalue solve can report converged Ritz
pairs while returning modes that are unacceptable under the full operator and
boundary treatment.

### Convergence principles

`eps_tol` alone does not define mode quality. For interior non-Hermitian
problems, tightening `eps_tol` does not guarantee proportional improvement in
accepted modes. Mode quality can be limited by `ncv`, target placement,
operator conditioning, non-normality, and spectral crowding.

`ncv` is a quality-control parameter. If the target neighborhood is crowded or
non-normal, too small an `ncv` can make accepted-mode quality plateau even when
the solve is active.

Eigenvalues can stabilize before eigenvectors. A spectrum that appears stable
is not enough; explicit postcheck residuals and boundary leakage still matter.

Shift-invert quality depends on the shifted operator. If `iL0 - sigma I` is
poorly conditioned, residuals can plateau even with direct LU.

Target choice matters. A target on the real axis can sit in a crowded
neighborhood. Moving the target slightly into the complex plane can isolate a
branch more cleanly. Treat this as a numerical search choice, not a change in
physics.

Branch-localize before scaling `nev`. Start with a small solve, test a few
targets around the branch of interest, judge success from postcheck residual and
target distance, then increase `nev` and `ncv`.

### What to check on every run

Before trusting a spectrum, check:

- requested `nev`;
- actual `nconv`;
- `ncv`;
- EPS tolerance and EPS convergence test;
- whether true residuals were enabled;
- whether shifted solves were direct or iterative;
- explicit postcheck residual counts;
- boundary-condition leakage diagnostics;
- target distance for accepted and rejected modes;
- wall time;
- peak RSS;
- MPI rank count and thread policy;
- the exact JSONC config and git commit that produced the artifact.

### What not to assume

Do not assume:

- `nconv > 0` means the run succeeded;
- passing the EPS convergence test implies a small full-operator residual;
- lowering `eps_tol` is always the cheapest way to improve results;
- a larger `nev` gives more trustworthy leading modes;
- the fastest setup on one machine is the fastest setup on another;
- a mean-flow linearization is physically meaningful only because the operator
  was assembled correctly.

## 6. Nondimensionalization

All flow fields stored in `BaseFlow` are nondimensional. The governing
equations are the compressible Navier-Stokes equations in the primitive
variables `[rho, u, v, w, T]`, linearized around the base flow. The
nondimensionalization follows the standard convective scaling described
below.

### 6.1 Reference scales

| Symbol | Definition | Units | Source in code |
|---|---|---|---|
| `rho_ref` | freestream density | kg/m³ | implicit: `RHO = 1` at freestream (`loaders.py`) |
| `U_ref` | freestream velocity | m/s | implicit: `U = 1` at freestream |
| `L_ref` | problem reference length | m | user-supplied (e.g. displacement thickness `delta*` for Blasius, shock-impingement length `X_sh` for SWBLI) |
| `T_ref` | freestream static temperature | K | `BaseFlow.T_0` (`baseflow.py:40`) |
| `p_ref` | `rho_ref * U_ref**2` | Pa | confirmed by Chu norm weights in `energy_norm.py` |
| `mu_ref` | `mu(T_0)` via Sutherland | Pa·s | `viscosity.py:46`, variable `mu_0` |
| `t_ref` | `L_ref / U_ref` | s | derived |

`L_ref` is not stored inside `BaseFlow`; it is the length scale used when
computing `Re` and when building the mesh coordinate arrays passed to the
loaders. Its physical interpretation depends on the case.

### 6.2 Nondimensional variables

| Variable | Stored as | Nondim form | Freestream value |
|---|---|---|---|
| density | `BaseFlow.RHO` | `rho* = rho / rho_ref` | 1 |
| streamwise velocity | `BaseFlow.U` | `u* = u / U_ref` | 1 |
| wall-normal velocity | `BaseFlow.V` | `v* = v / U_ref` | 0 |
| spanwise velocity | `BaseFlow.W` | `w* = w / U_ref` | 0 |
| temperature | `BaseFlow.T` | `T* = T / T_ref` | 1 |
| viscosity | `BaseFlow.MU` | `mu* = mu / mu(T_0)` | 1 |
| streamwise coordinate | mesh `X` | `x* = x / L_ref` | — |
| wall-normal coordinate | mesh `Y` | `y* = y / L_ref` | — |
| time | operator eigenvalue `omega` | `t* = t * U_ref / L_ref` | — |

Pressure is not stored explicitly. From the ideal gas law nondimensionalized
with `p_ref = rho_ref * U_ref**2`:

```
p* = RHO * T / (kappa * Ma**2)
```

This `Ma**2` factor is absent in acoustic-scaled codes; see §6.5.

### 6.3 Dimensionless numbers

**Reynolds number**

```
Re = rho_ref * U_ref * L_ref / mu(T_0)
```

`mu(T_0)` is Sutherland's law evaluated at the freestream temperature `T_0`.
Enters viscous terms as `1/Re`; for example, `coefficients.py` has
`c_Arr_44 = -MU / Re` (streamwise momentum wall-normal diffusion).

**Mach number**

```
Ma = U_ref / a_ref,  a_ref = sqrt(kappa * R * T_ref)
```

Enters the operator via `c2 = kappa * Ma**2` (pressure-velocity coupling)
and `cv = 1 / (kappa * (kappa - 1) * Ma**2)` (energy equation). Both are
derived in `viscosity.py:34-36` and used throughout `coefficients.py`.

**Prandtl number**

```
Pr = mu(T_0) * cp / k_ref,  cp = kappa * R / (kappa - 1)
```

Enters thermal diffusion as `1 / (Re * Pr)`; for example,
`c_Arr_55 = -kappa * MU / Re / Pr` (temperature Laplacian in energy equation).

**Heat capacity ratio**

```
kappa = cp / cv  (gamma)
```

Stored as `BaseFlow.kappa`. The code uses `kappa` rather than `gamma`; the
`BaseFlow` docstring at `baseflow.py:27` notes "Heat capacity ratio (gamma)"
to keep the physical meaning explicit. Enters the ideal gas relation, the
Chu norm weights, and all compressibility terms.

Re, Ma, Pr, and kappa are user-supplied in the JSONC config under `physics`.
The code uses them as given without internal consistency checks.

### 6.4 Sutherland viscosity

Dynamic viscosity follows Sutherland's law for air. The normalization
reference is `mu_0 = mu(T_0)`, the Sutherland value at the actual freestream
temperature. The relevant section of `viscosity.py`:

```python
    S1: float = 110.4  # Sutherland temperature [K]
    T_ref: float = 280.0  # reference temperature [K]
    mu_ref: float = 1.735e-5  # reference dynamic viscosity [Pa.s]

    Tstar = bf.T * T_0  # dimensional temperature

    # Reference viscosity at T_0
    mu_0 = mu_ref * (T_ref + S1) / (T_0 + S1) * (T_0 / T_ref) ** 1.5

    # Viscosity and its first two derivatives w.r.t. dimensional T
    mu = mu_ref * (T_ref + S1) / (Tstar + S1) * (Tstar / T_ref) ** 1.5
```

`T_ref = 280.0 K` and `mu_ref = 1.735e-5 Pa.s` are anchoring constants in
the Sutherland formula, not the nondimensionalization scale. The operative
nondim scale is `mu_0`, which depends on `T_0`. `MU = mu / mu_0`, so
`MU = 1` at freestream regardless of what `T_0` is.

The temperature derivatives are nondimensionalized consistently:

```
dmudT*  = d(mu/mu_0) / d(T*)  =  (dmu/dT_dim) * T_0 / mu_0
d2mudT2* = (d2mu/dT_dim**2) * T_0**2 / mu_0
```

### 6.5 LES vs linstabpy convention notes

Three specific points where LES codes commonly differ.

**Pressure reference.** Some LES codes use `p_ref = rho_ref * R * T_ref`
(acoustic scaling). linstabpy uses `p_ref = rho_ref * U_ref**2` (dynamic
pressure). The two coincide only at `Ma = 1`. For `Ma != 1`, the nondim
ideal gas law differs by a factor of `kappa * Ma**2`:

| Quantity | linstabpy | acoustic-scaled LES |
|---|---|---|
| `p*` | `p / (rho_ref U_ref**2)` | `p / (rho_ref R T_ref)` |
| ideal gas | `p* = RHO * T / (kappa * Ma**2)` | `p* = RHO * T` |
| conversion | multiply acoustic `p*` by `kappa * Ma**2` | divide linstabpy `p*` by `kappa * Ma**2` |

**Energy equation scaling.** In linstabpy `cv* = 1 / (kappa * (kappa-1) * Ma**2)`.
An LES code that writes `cv* = 1 / (kappa - 1)` is using acoustic-based
nondim. To convert: `cv_LES = cv_linstabpy * kappa * Ma**2`.

**Temperature.** linstabpy stores `T* = T / T_ref` with `T* = 1` in the
freestream. Some LES post-processing outputs temperature as excess
`(T - T_inf) / T_inf` or as a ratio `(T - T_w) / (T_0 - T_w)`. If an
external base-flow file provides temperature already divided by `T_ref`,
load it directly into `BaseFlow.T`. If it provides temperature in Kelvin,
divide by `T_0` before assigning.

## 7. Boundary conditions

Boundary setup is JSONC-driven. Row-replacement boundary conditions are listed
under `bc.recipe`; sponge damping is configured in JSONC as an operator damping
field or loaded from reference case data.

Example:

```jsonc
{
  "physics": {
    "sponge": {
      "enabled": true,
      "preset": "tanh",
      "amplitude": 0.1
    }
  },
  "bc": {
    "penalty": 1e8,
    "recipe": [
      {
        "type": "dirichlet",
        "borders": "lt",
        "variables": "ruvwT"
      },
      {
        "type": "dirichlet",
        "borders": "b",
        "variables": "ruvw"
      },
      {
        "type": "neumann",
        "borders": "b",
        "variables": "T",
        "exclude_corners": true,
        "curvilinear_normals": true
      },
      {
        "type": "neumann",
        "borders": "r",
        "variables": "ruvwT",
        "exclude_corners": true,
        "curvilinear_normals": true
      }
    ]
  },
  "analysis": {
    "stability_acceptance": {
      "postcheck_rel_max": 5e-5,
      "target_distance_max": 0.15,
      "omega_real_abs_max": 1.5,
      "omega_imag_abs_max": 2.0,
      "conjugate_aware": true
    }
  }
}
```

Recipe fields:

| Field | Meaning |
|---|---|
| `type` | `dirichlet` or `neumann` for row-replacement BCs. Sponge damping is configured separately under `physics.sponge` or loaded from case input data. |
| `borders` | Boundary letters. Common values are `l`, `r`, `t`, `b`, or combinations such as `lt`. |
| `variables` | Variable block selector, such as `u`, `T`, `ruvw`, or `ruvwT`. |
| `exclude_corners` | For Neumann rules, remove corner points that would overlap with other boundary rules. |
| `curvilinear_normals` | For Neumann rules, use per-face curvilinear normals instead of axis-aligned normals. |

Use `curvilinear_normals: true` for deformed or body-fitted curvilinear meshes.
The flag makes Neumann rows use per-face normals from the stored mesh Jacobian
entries. Set it to `false` when the base-flow mesh is Cartesian and the
curvilinear normal path is not needed; that preserves the axis-aligned path and
can save a small amount of work. Older parity configurations may spell this
flag explicitly to preserve historical behavior. New configs should spell the
intended value instead of relying on omission.

`analysis.stability_acceptance.conjugate_aware` defaults to `true` in the
current stability-acceptance path. It treats an eigenvalue and its complex
conjugate as equally close to the target:

```text
min(|omega - target|, |omega - conj(target)|)
```

This is the right default for real-valued base flows, where oscillatory modes
come in conjugate pairs. Set `conjugate_aware: false` when you need strict
non-redundant mode selection around one target branch, or when the operator is
not real-valued and conjugate symmetry should not be admitted.

## 8. Parity with MATLAB LinStab2D

### 8.1 Example-level parity table

Reference MATLAB examples:

- `linstabpy/data/baseflows/example_0_TestCurvilinearDerivative.m`
- `linstabpy/data/baseflows/example_1_Blasius.m`
- `linstabpy/data/baseflows/example_2_Jet.m`
- `linstabpy/data/baseflows/example_3_Streaky_BL.m`
- `linstabpy/data/baseflows/example_4_Cavity.m`
- `linstabpy/data/baseflows/example_5_Paraboloid.m`

Python counterparts:

- `examples/example_0_curvilinear_validation.py`
- `examples/example_1_blasius.py`
- `examples/example_2_jet.py`
- `examples/example_3_streaky_bl.py`
- `examples/example_4_cavity.py`
- `examples/example_5_paraboloid.py`

| MATLAB example | Python example | Status | Required artifacts |
|---|---|---|---|
| `example_0_TestCurvilinearDerivative.m` | `example_0_curvilinear_validation.py` | `implemented` | `example_0_curvilinear_validation.npz`, `example_0_curvilinear_validation.png`, `tests/fixtures/curvilinear_derivative_parity.json` |
| `example_1_Blasius.m` | `example_1_blasius.py` | `implemented` | `example_1_blasius.npz`, `example_1_blasius.png` |
| `example_2_Jet.m` | `example_2_jet.py` | `implemented` | `example_2_jet.npz`, `example_2_jet.png` |
| `example_3_Streaky_BL.m` | `example_3_streaky_bl.py` | `implemented` | `example_3_streaky_bl.npz`, `example_3_streaky_bl.png` |
| `example_4_Cavity.m` | `example_4_cavity.py` | `implemented` | `example_4_cavity.npz`, `example_4_cavity.png` |
| `example_5_Paraboloid.m` | `example_5_paraboloid.py` | `implemented` | `example_5_paraboloid.npz`, `example_5_paraboloid.png` |

### 8.2 Operator parity matrix

Canonical machine-readable source: `scripts/validation/martini2024/tools/operator_matrix.json`.

Status semantics:

- `pending`: not implemented yet;
- `implemented`: code exists but parity artifact/signoff is still pending;
- `validated`: implemented and parity-validated by artifacts;
- `deferred`: intentionally postponed;
- `out_of_scope`: intentionally not targeted;
- `blocked`: cannot proceed due to an external blocker.

Every `required: true` entry must reach `status = validated`. Current gate
status is `pass` with `missing_required = 0`.

| ID | MATLAB | Python target | Status |
|---|---|---|---|
| `construct_get_lin_problem` | `ConstructOperators/GetLinProblem.m` | `linstabpy/operators/assembly.py::build_lnse_operator` | `validated` |
| `construct_get_spatial_lin_problem` | `ConstructOperators/GetSpatialLinProblem.m` | `linstabpy/operators/spatial.py::build_spatial_operator` | `validated` |
| `construct_cartesian_coefficients` | `ConstructOperators/getCoeffsLaminar_Cartesian.m` | `linstabpy/operators/coefficients.py::compute_cartesian_coefficients` | `validated` |
| `construct_axisymmetric_coefficients` | `ConstructOperators/getCoeffsLaminar.m` | `linstabpy/operators/coefficients.py::compute_cylindrical_coefficients` | `validated` |
| `construct_create_diff_matrices_axy` | `ConstructOperators/CreateDiffMatrices_Axy.m` | `linstabpy/operators/assembly.py::_create_axy_diff_matrices` | `validated` |
| `construct_lnseoperators_cyl` | `ConstructOperators/LNSEoperators_cyl.m` | `linstabpy/operators/assembly.py::build_lnse_operator(model="axy")` | `validated` |
| `construct_get_comp_energy_norm` | `ConstructOperators/GetCompEnergyNorm.m` | `linstabpy/operators/energy_norm.py::compute_chu_norm` | `validated` |
| `bc_dirichlet` | `BoundaryConditions/BC_Dirichlet.m` | `linstabpy/boundary_conditions/dirichlet.py::apply_dirichlet` | `validated` |
| `bc_dirichlet_spatial` | `BoundaryConditions/BC_Dirichlet_SpatialStability.m` | `linstabpy/boundary_conditions/dirichlet.py::apply_dirichlet_spatial` | `validated` |
| `bc_neumann` | `BoundaryConditions/BC_Neumann.m` | `linstabpy/boundary_conditions/neumann.py::apply_neumann` | `validated` |
| `bc_set_dirbc` | `BoundaryConditions/SetDirBC.m` | `linstabpy/boundary_conditions/dirichlet.py::build_dirichlet_restriction_matrices` | `validated` |
| `bc_set_dirichletbc` | `BoundaryConditions/SetDirichletBC.m` | `linstabpy/boundary_conditions/dirichlet.py::apply_dirichlet_identity_pair` | `validated` |

### 8.3 Branch coverage matrix

Canonical machine-readable source: `scripts/validation/martini2024/tools/branch_matrix.json`.

Target dimensions:

- `model`: `2D`, `axy`;
- `solver`: `temporal`, `spatial`, `adjoint`, `resolvent`;
- `bc`: `Dirichlet`, `DirichletSpatial`, `Neumann`;
- `deformation`: `none`, `x_only`, `y_only`, `full_2d`;
- `backend`: `petsc_slepc`.

Total combinations:

```text
2 * 4 * 3 * 4 * 1 = 96
```

All required combinations must be `validated`. Combinations omitted from
`records` default to `pending`. Scope is controlled by `required_exclusions` in
`scripts/validation/martini2024/tools/branch_matrix.json`. Current scoped gate status is `pass`
with `5` required validated combinations and `91` required exclusions.

| model | solver | bc | deformation | backend | status |
|---|---|---|---|---|---|
| `2D` | `temporal` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `spatial` | `DirichletSpatial` | `none` | `petsc_slepc` | `validated` |
| `2D` | `adjoint` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `resolvent` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `temporal` | `Neumann` | `none` | `petsc_slepc` | `validated` |

### 8.4 Running parity checks

Example-level parity:

```bash
.venv/bin/python scripts/validation/martini2024/tools/compare_martini_examples.py
```

Allow incomplete optional matrix entries while enforcing required scoped gates:

```bash
python scripts/validation/martini2024/tools/check_parity_matrices.py --allow-incomplete
```

Strict CI gate:

```bash
python scripts/validation/martini2024/tools/check_parity_matrices.py
```

## 9. Case studies

### 9.1 Case 1 (SWBLI)

Case 1 is the SWBLI parity harness used to prove that the Python/PETSc/SLEPc
path is equivalent to the MATLAB reference before scaling to larger production
cases.

Objective:

- compare the Python stability and resolvent paths against MATLAB reference
  exports;
- fix operator/source logic when parity fails;
- avoid hardcoding MATLAB outputs as a substitute for correct assembly.

Current status as of 2026-02-28:

- operator-stage parity: `PASS`;
- resolvent parity: `PASS`;
- stability spectrum parity is handled by the case script with awareness of
  which MATLAB export produced the available spectrum.

Key fixes that made parity stable:

- preserve mesh filters during deformation, matching MATLAB behavior;
- load `sponge_aux` from MATLAB `case_inputs_used.mat`;
- use the MATLAB-equivalent `H^H W H` EVP route for resolvent extraction;
- keep canonical parity parameters in the Case 1 fixture directory.

Run from the repository root:

```bash
bash examples/case_studies/case1_song/case1.sh
```

Primary outputs:

- `examples/case_studies/case1_song/case1.npz`;
- `examples/case_studies/case1_song/output/case1_operator_stage_parity.json`;
- `examples/case_studies/case1_song/output/case1_spectrum_parity.json`;
- `examples/case_studies/case1_song/output/case1_resolvent_parity.json`;
- parity figures under `examples/case_studies/case1_song/output/`.

`examples/case_studies/case1_song/` is the canonical Case 1 study path.

Environment overrides:

| Variable | Default | Use |
|---|---:|---|
| `LINSTAB2D_CASE1_CASE_DIR` | `data/baseflow` | Case data directory, relative to `examples/case_studies/case1_song/` unless absolute. |
| `LINSTAB2D_CASE1_RUN_RESOLVENT` | `false` | Enable or disable resolvent computation. |
| `LINSTAB2D_CASE1_RUN_STABILITY` | `true` | Enable or disable temporal stability computation. |
| `LINSTAB2D_CASE1_N_MODES` | `100` | Number of stability modes requested. |
| `LINSTAB2D_CASE1_N_MODES_RESOLVENT` | `10` | Number of resolvent modes requested. |
| `LINSTAB2D_CASE1_STRIDE_KSI` | `1` | Mesh stride in the first computational coordinate. |
| `LINSTAB2D_CASE1_STRIDE_ETA` | `1` | Mesh stride in the second computational coordinate. |
| `LINSTAB2D_CASE1_BC_PENALTY` | `1e8` | Boundary-condition penalty used by the case driver. |

Spectrum comparison from the repository root:

```bash
.venv/bin/python examples/case_studies/case1_song/scripts/check_spectrum.py \
  --python-npz examples/case_studies/case1_song/case1.npz \
  --matlab-spectrum examples/case_studies/case1_song/reference/matlab/MATLAB_results/spectrum.mat
```

Solver strategy A, the robust baseline:

- `ST=SINVERT`;
- `KSP=PREONLY`;
- `PC=LU`;
- PETSc MUMPS factorization;
- main accuracy/selection knobs are `eps_tol`, `conv_test`,
  `true_residual`, `nev`, `ncv`, and target shift;
- use this path first when direct factorization fits memory.

Solver strategy B, the scalable path:

- `ST=SINVERT` or a preconditioned eigensolver route;
- iterative `KSP` with strict `ksp_error_if_not_converged`;
- block Jacobi or ASM with local LU as the current baseline direction;
- stronger physics-block preconditioners remain a tuning track;
- use this path when the direct memory budget is exceeded, and keep parity and
  convergence gates enabled.

Practical rule: validate reduced grids with Strategy A, then move to Strategy B
for cases that do not fit direct factorization.

### 9.2 Case 2

Case 2 has its own active campaign files. Do not duplicate their content in the
top-level README.

- `examples/case_studies/case2_lui/docs/PLAN.md`: current phase plan, acceptance criteria, open
  blockers, and campaign state.
- `examples/case_studies/case2_lui/docs/REFERENCE.md`: stable technical background, governing
  equations, parameter map, physical framing, and references.
- `examples/case_studies/case2_lui/docs/README.md`: operational commands and artifact layout.

Read them in that order before changing Case 2 numerics.

As with Case 1, `examples/case_studies/case2_lui/` is the canonical Case 2 study namespace.

## 10. Troubleshooting

### macOS SLEPc NHEP crash

Symptom:

- Python `slepc4py` `EPS_NHEP` with Krylov-Schur / Arnoldi aborts;
- native C SLEPc NHEP works;
- Python HEP and LAPACK-backed paths work.

Root cause:

- Apple Accelerate LAPACK symbols can be interposed ahead of the LAPACK symbols
  expected by PETSc-downloaded ScaLAPACK inside a Python process.

Fix:

- install Homebrew OpenBLAS and ScaLAPACK;
- run `bash install.sh` so PETSc is configured with those libraries when
  possible;
- use the `.venv/bin/python` wrapper generated by the installer.

Detection:

```bash
bash install.sh --check
```

The check includes a tiny Python SLEPc `EPS_NHEP` Krylov-Schur solve.

### Linux MPI compile failure

If `diagnose.sh` reports an MPI compile failure with `mpi.h` missing, install
the development package for the MPI stack you intend to use. On Ubuntu with
Open MPI, the needed packages are:

```bash
sudo apt install -y libopenmpi-dev openmpi-bin
```

On clusters, load the complete MPI/compiler module instead of installing system
packages.

### Dual-MPI conflict

If Open MPI, MPICH, and/or Intel MPI are all visible, choose one stack, make it
first in `PATH`, and reinstall from a clean `.venv`. Do not let PETSc build
against one MPI while `mpiexec` launches another.

### `mpi4py` ABI mismatch

If `mpi4py` imports fail after switching MPI stacks:

```bash
rm -rf .venv
uv cache clean
LINSTAB2D_MPI4PY_NO_BINARY=1 bash install.sh
```

### PETSc configure hang

If PETSc configure hangs or fails on the wrong launcher:

```bash
LINSTAB2D_PETSC_MPIEXEC=/path/to/working/mpiexec bash install.sh
```

Use the same launcher for verification when needed:

```bash
LINSTAB2D_VERIFY_MPIEXEC=/path/to/working/mpiexec bash install.sh --check
```

### Intel MPI

Load a complete Intel MPI / oneAPI environment before running `diagnose.sh` or
`install.sh`. Do not combine an Intel MPI `mpiexec` with Open MPI or MPICH
Python bindings.

```bash
source /opt/intel/oneapi/setvars.sh --force
export LINSTAB2D_MPI_STACK=intelmpi
bash diagnose.sh
bash install.sh
```

### What to send when a machine is still broken

Send:

- full `diagnose.sh` output;
- full `install.sh` output;
- `benchmark_runs/<run-id>/benchmark_summary.csv`;
- relevant `benchmark_runs/<run-id>/*.log`;
- a redacted stack manifest:

```bash
.venv/bin/python scripts/tools/export_known_good_stack.py \
  --output stack_manifest.json \
  --redact-local-details
```

For deeper OS-specific notes and installer experiments, see
`scripts/ci/`.

## 11. Development

Run tests with:

```bash
.venv/bin/python -m pytest tests/
```

Expected collection size:

```bash
.venv/bin/python -m pytest --collect-only tests
```

Current expected result: `535` collected tests.

Run MATLAB parity checks with:

```bash
.venv/bin/python scripts/validation/martini2024/tools/compare_martini_examples.py
python scripts/validation/martini2024/tools/check_parity_matrices.py --allow-incomplete
python scripts/validation/martini2024/tools/check_parity_matrices.py
```

Packaging:

- build metadata lives in `pyproject.toml`;
- importable code uses the `src/` layout;
- package name is `linstabpy`;
- import package is `linstabpy`;
- optional PETSc/SLEPc dependencies are declared as:

```bash
pip install linstabpy[petsc]
```

For development from a checkout, prefer the repository installer because it
builds and verifies PETSc/SLEPc:

```bash
bash install.sh
.venv/bin/python -m pip install -e .
```

Reference MATLAB source boundary:

- Python implementation: `src/linstabpy/`, `tests/`, `examples/`, `tools/`,
  and repository scripts;
- MATLAB reference source and `.mat` assets:
  `src/linstabpy/data/baseflows`;
- update the submodule only with explicit intent;
- do not edit reference MATLAB files while trying to fix Python parity.

Roadmap:

| Phase | Status | Description |
|---|---|---|
| Core assembly plus Martini examples | Done | Example family implemented and parity-gated. |
| PETSc/SLEPc solvers | Done | MUMPS direct solves and SLEPc eigenvalue/SVD paths. |
| Additional validation cases | In progress | SWBLI validation and production case workflows. |
| Solver optimization | Planned | Improve sparse assembly, transfer, and preconditioner paths. |
| Matrix-free Krylov | Future | Time-stepper-style approach for larger 3D-oriented workflows. |

## 12. References

Martini, E. & Schmidt, O. (2024). Linstab2D: stability and resolvent analysis
of compressible viscous flows in MATLAB. *Theoretical and Computational Fluid
Dynamics*. doi:10.1007/s00162-024-00706-0.

Strand, B. (1994). Summation by parts for finite difference approximations for
d/dx. *Journal of Computational Physics*, 110(1), 47-67.

Mattsson, K. & Nordstrom, J. (2004). Summation by parts operators for finite
difference approximations of second derivatives. *Journal of Computational
Physics*, 199(2), 503-540.

## 13. License

linstabpy is distributed under **GPL-3.0-only**, matching the license of
its upstream MATLAB ancestor LinStab2D (Martini & Schmidt 2024,
https://github.com/LinStab2D/LinStab2D). See `LICENSE` for the full
GPL-3.0 text and `NOTICE` for upstream attribution and modification
history.

When using linstabpy in published work, cite the LinStab2D paper:

> Martini, E. & Schmidt, O. T. LinStab2D: stability and resolvent
> analysis of compressible viscous flows in MATLAB. *Theor. Comput.
> Fluid Dyn.* (2024). doi:10.1007/s00162-024-00706-0

See `CITATION.cff` for the machine-readable citation entry. Once
linstabpy is archived on Zenodo, the software DOI and release version
will be added to that file (placeholders are listed there as TODOs).
