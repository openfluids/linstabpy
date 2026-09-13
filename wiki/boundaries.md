# Boundary interfaces

This page is the reference for how you talk to LinStabPy from outside: CLI commands, YAML cases, Python imports, file inputs, and output files. Math conventions match the rest of the docs: $\omega = \mathrm{eig}(i L_0)$, unstable exactly when $\mathrm{Im}(\omega) > 0$, [resolvent](https://grokipedia.com/page/Resolvent_formalism) $H = C F (L_0 - (\gamma + i\omega) I)^{-1} F B$.

## Command line interface

### linstabpy (driver CLI)

**Description**: Command-line entry for the shared case driver. It takes no case arguments. It reads the case path and output path from environment variables, checks the result cache by config hash and git SHA on rank 0, and either reuses cached results or runs the full driver (preflight, operator assembly, [PETSc](https://en.wikipedia.org/wiki/PETSc) and [SLEPc](https://en.wikipedia.org/wiki/SLEPc) solves, IO, provenance).

**Source file**: `src/linstabpy/driver/cli.py`

**Flags**:

- `--version`: optional. Print the installed package version and exit without initializing PETSc or [MPI](https://grokipedia.com/page/Message_Passing_Interface).

**Environment** (the real CLI surface):

- `LINSTABPY_CASE_CONFIG`: YAML case file. Legacy aliases `LINSTABPY_CASE2_CONFIG`, `LINSTABPY_CASE1_CONFIG`. Default `./configs/parity_config.yaml`.
- `LINSTABPY_CASE_OUTPUT_NPZ`: result archive path. Per-run aliases `LINSTABPY_CASE2_OUTPUT_NPZ`, `LINSTABPY_CASE1_OUTPUT_NPZ`. Default `{prefix}_{config_md5}_{git_sha8}.npz` in the working directory.
- `LINSTABPY_CASE_LOG_PREFIX`: log prefix. Default `linstabpy`.
- `LINSTABPY_FORCE_RECOMPUTE`: set to `1`, `true`, `yes`, or `on` to skip the cache check.
- `LINSTABPY_STAGE_HEARTBEAT_SECONDS`: progress heartbeat interval. Default `30`.

!!! warning "Unknown arguments are silently ignored"
    The parser uses `parse_known_args`, so extra words on the command line do not error. They are dropped and the run uses the env config. There is no `run` subcommand and no `--config` flag. An older revision of this page documented both. They never existed.

**Usage examples**:

```bash
linstabpy --version
```

```bash
export LINSTABPY_CASE_CONFIG=path/to/case.yaml
export LINSTABPY_CASE_OUTPUT_NPZ=results/out.npz
mpiexec -n 8 linstabpy
```

```bash
python -m linstabpy.driver.cli
```

### YAML case configuration interface

**Description**: External configuration boundary for scientific cases. Users interact primarily by editing YAML case files that define baseflow paths, physics, numerics (including [finite-difference](https://grokipedia.com/page/Finite_difference_method) orders), boundary conditions, eigen/resolvent parameters, and solver policy. Files are ingested with a SafeLoader that accepts scientific floats, then structurally validated (TOP_LEVEL_ALLOWED keys such as paths/physics/numerics and related blocks).

**Source File**: `src/linstabpy/utils/yaml_io.py; src/linstabpy/config/schema.py; src/linstabpy/utils/_solver_cfg.py`

**Arguments**:

- `config path` (str | Path): required - Filesystem path passed to load_config(path) returning a dict for driver/solvers.

**Options**:

- `paths`(mapping): optional - Top-level block for input/output paths (baseflow npy/npz dumps, mesh, artifacts).
- `physics`(mapping): optional - Top-level physics parameters for the linearized [compressible](https://grokipedia.com/page/Compressible_flow) [Navier-Stokes](https://grokipedia.com/page/Navier–Stokes_equations) problem.
- `numerics`(mapping): optional - Top-level numerics block including finite-difference order settings traversed by validation helpers.
- `bc`(mapping): optional - Boundary recipe block: ordered `recipe` rules plus `penalty` (Dirichlet diagonal pin and Neumann row scale; driver default `1e8`, library and router default `1e4`).
- `complex / int list fields`(complex | list[complex] | list[int]): optional - Loosely typed YAML scalars/lists parsed into Python complex and int lists (parse_complex, parse_complex_list, parse_int_list) for frequencies, [wavenumbers](https://grokipedia.com/page/Wavenumber), and similar solver inputs.

**Usage Examples**:

```bash
load_config('cases/blasius_resolvent.yaml')
```

```bash
# case.yaml top-level keys constrained by validate_config / TOP_LEVEL_ALLOWED
```

### make (tutorial docs build)

**Description**: Secondary CLI boundary for building the tutorial PDF documentation with pdflatex (not part of the solver runtime).

**Source File**: `docs/tutorial/Makefile`

**Usage Examples**:

```bash
make all
```

```bash
make clean
```

```bash
make tutorial.pdf
```

## Integration Suggestions

### Python library import (lazy public API)

Embed LinStabPy in larger openfluids or research pipelines by importing the package. Public symbols resolve on first attribute access so a plain import stays lightweight and does not eagerly initialize PETSc/MPI. Suitable for stability, [adjoint](https://grokipedia.com/page/Adjoint), [structural sensitivity](core-concepts.md), and resolvent workflows after the caller supplies baseflow data and config.

**Example Code**:

```
import linstabpy
# Heavy solvers load only on first attribute access
print(linstabpy.__version__)  # version without PETSc init
# Configure via YAML dict or SolverConfig, then call driver/stability/resolvent APIs
```

**Best Practices**:

- Rely on lazy __getattr__ so import-time side effects stay minimal
- Call PETSc/MPI initialization explicitly only when entering solver paths
- Pass validated config dicts (validate_config) rather than ad-hoc nested structures
- Prefer SolverConfig / from_config so YAML, env, and defaults share one precedence model

### YAML case file + driver orchestration

Standard user-facing integration for batch and HPC runs: prepare mean-flow fields (npy/npz), write a Martini/Song-style YAML case, validate, and run through the driver CLI under MPI. Outputs are stability/resolvent artifacts, metrics, and provenance payloads.

**Example Code**:

```
# case.yaml defines paths, physics, numerics, eigen/resolvent, solver policy
from linstabpy.utils.yaml_io import load_config
from linstabpy.config.schema import validate_config
cfg = load_config('case.yaml')
validate_config(cfg, source='case.yaml')
# Then invoke driver.run / CLI entry with the merged config
```

**Best Practices**:

- Keep dissipation apply flags aligned across stability and resolvent stages
- Use schema validation before long MPI jobs to fail fast on unknown keys
- Store baseflow and mesh paths relative to a reproducible case directory
- Document expected artifacts similarly to config/validation_cases.yaml catalogs

### PETSc / SLEPc / MUMPS / optional cuDSS backend

External numerical stack integration. Core install uses NumPy/SciPy/PyYAML; extras enable HDF5, parallel PETSc/SLEPc (mpi4py, petsc4py, slepc4py), and optional GPU [cuDSS](https://docs.nvidia.com/cuda/cudss/). [MUMPS](https://en.wikipedia.org/wiki/MUMPS_(software)) ordering names are shared via mumps_options without importing PETSc at module load.

**Example Code**:

```
# Install core + parallel stack
# pip install 'linstabpy[petsc]'  # or conda via environment.yml
# Optional: pip install 'linstabpy[hdf5]' 'linstabpy[cudss]'
# Runtime: export LINSTABPY_CASE_CONFIG=case.yaml && mpiexec -n N linstabpy
```

**Best Practices**:

- Install matching mpi4py/petsc4py/slepc4py builds for the cluster MPI
- Select MUMPS ordering (amd/pord/metis/qamd/auto) consistently in config
- Treat cuDSS as optional GPU factor path shelled through PETSc PC
- Do not import PETSc-dependent modules at package import time; follow lazy init patterns

### CFD baseflow data stores (file I/O)

Ingest on-disk mean-flow fields as npy directories or npz archives (density, velocity components, temperature, mesh, sponge) produced by external nonlinear [CFD](https://grokipedia.com/page/Computational_fluid_dynamics). LinStabPy does not generate baseflows; integration starts at load/preprocess boundaries under baseflow/.

**Example Code**:

```
# paths block in case.yaml points at baseflow archives
# paths:
#   baseflow: ./data/meanflow.npz
#   output: ./artifacts/
```

**Best Practices**:

- Provide structured-mesh-compatible fields expected by grid_fields/preprocess
- Apply sponge and shock-cleanup options via config rather than ad-hoc scripts
- Version baseflow dumps alongside YAML for provenance
