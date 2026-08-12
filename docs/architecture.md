# Architecture Map

This note tells maintainers where numerical policy lives.
Keep policy in the owning module rather than recreating pass-through files.

## Ownership

| Area | Owning module | What belongs there |
|---|---|---|
| Boundary-condition recipes | `linstabpy.boundary_conditions.recipe` | YAML recipe parsing, rule validation, effective rows, column-zeroed Dirichlet rows, overlap reporting, and SciPy/PETSc/spatial dispatch. |
| LNSE assembly | `linstabpy.operators.assembly` | Coefficient preparation, backend choice, operator construction order, boundary-condition application, and the `LNSESystem` result object. |
| Solver policy | `linstabpy.utils._solver_cfg`, `linstabpy.config.typed` | Dict/env resolution, PETSc/SLEPc defaults, MUMPS/KSP/EPS setup, and resolvent linear-solver selection. |
| Runtime pipeline | `linstabpy.driver.run` | Case loading, mesh construction, Chu norm, assembly, stability/resolvent stages, postchecks, acceptance, and `.npz` persistence. |
| Packed-field ingest | `linstabpy.baseflow.grid_fields` | One loader for npz or an npy directory. Filter, boundary sponge, and shock live in `baseflow.preprocess`. |
| Result archive | `linstabpy.driver.payload`, `linstabpy.driver.io` | Payload grouping and NPZ/sidecar write. `io.py` does not import the driver. |
| Mesh state | `linstabpy.mesh.mesh` | Structured geometry, derivative operators, filters, dissipation state, and sponge state. |

## Boundary Conditions

`BCRecipe` is the boundary-condition seam. Public YAML files pass lists of
dictionaries; the recipe layer normalizes them before application.
Effective rows are the rows owned by the final rule after overlap resolution.
Column-zeroed Dirichlet rows record the broader set needed for operator
reduction and provenance.

When changing boundary-condition behavior, update the recipe analysis and every
backend wrapper together. Do not duplicate row-selection logic in assembly,
runtime code, or example scripts.

## LNSE Assembly

`LNSESystem` is the assembly handoff. It carries the assembled operator, index
map, boundary-condition indices, and recipe analysis. The construction order is
part of the numerical contract:

1. prepare coefficients;
2. build the raw operator;
3. apply the inverse mass/metric block;
4. apply sponge and operator dissipation;
5. apply boundary conditions.

Keep this order explicit in `operators.assembly`. Runtime callers should ask
for an assembled system, not rebuild pieces of this sequence.

## Solver Policy

`Config` in `config.typed` reads YAML and environment variables. PETSc/SLEPc
objects are still created inside the solver modules, but the defaults and
fallback choices come from that shared policy layer.

MPI ownership matters: launcher, `mpi4py`, PETSc, and SLEPc must agree on one
MPI stack. Installer checks should report the chosen stack rather than
guessing silently.

## Runtime And NPZ Output

Example scripts set `LINSTABPY_CASE_CONFIG` and call `linstabpy.driver.run`.
Martini generated-mesh examples still build their own fields. Packed-field
cases go through `load_case_fields`.

The compatibility surface for saved results is
`RUNTIME_NPZ_CONTRACT_KEYS` in `linstabpy.driver.payload`. The grouped
explanation is in `docs/runtime_npz_contract.md`. Additive fields are
acceptable. Renames or removals require a migration note, test updates, and
reader updates.

## Mesh State

`Mesh` remains the main structured-grid object. Deformation and masking must
preserve numerical behavior. Geometry-only optimizations are allowed only when
tests prove that derivative operators, filters, weights, and saved outputs
remain equivalent for the relevant cases.

## Packed fields

Every packed case uses the same eight keys: `x`, `y`, `rho`, `u`, `v`, `w`,
`T`, `sponge`. YAML names the file. The loader does not guess a Song filename.
Martini `.mat` constructors live under `examples/martini/matlab_baseflow.py`.
