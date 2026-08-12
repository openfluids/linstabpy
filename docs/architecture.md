# Architecture Map

This note tells maintainers where numerical policy lives after the refactor.
Keep policy in the owning seam rather than recreating pass-through modules.

## Ownership

| Area | Owning module | What belongs there |
|---|---|---|
| Boundary-condition recipes | `linstabpy.boundary_conditions.recipe` | YAML recipe parsing, rule validation, effective rows, column-zeroed Dirichlet rows, overlap reporting, and SciPy/PETSc/spatial dispatch. |
| LNSE assembly | `linstabpy.operators.assembly` | Coefficient preparation, backend choice, operator construction order, boundary-condition application, and the `LNSESystem` result object. |
| Solver policy | `linstabpy.utils._solver_cfg`, `linstabpy.solvers.*` | Dict/env compatibility, PETSc/SLEPc defaults, MUMPS/KSP/EPS setup, root-scatter decisions, and resolvent linear-solver selection. |
| Runtime pipeline | `linstabpy.driver.run` | Case loading, mesh construction, Chu norm, assembly, stability/resolvent stages, postchecks, acceptance, and `.npz` persistence. |
| Mesh state | `linstabpy.mesh.mesh` | Structured geometry, derivative operators, filters, dissipation state, sponge state, and conservative read-only views over those groups. |

## Boundary Conditions

`BCRecipe` is the boundary-condition seam. Public YAML files pass lists of
dictionaries; the recipe layer normalizes them before application.
The important distinction is between effective rows and column-zeroed rows:
effective rows are the rows owned by the final rule after overlap resolution,
while column-zeroed Dirichlet rows record the broader set that must be known for
operator reduction and provenance.

When changing boundary-condition behavior, update the recipe analysis and every
backend wrapper together. Do not duplicate row-selection logic in assembly,
runtime code, or case-study drivers.

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

`SolverConfig` keeps existing dict and environment-variable behavior while
centralizing how solver defaults are interpreted. PETSc/SLEPc objects are still
created inside the solver modules, but the defaults and fallback choices should
come from the shared policy layer. This keeps direct MUMPS, KSP/PC options,
EPS tolerances, and resolvent linear-solver choices inspectable from one path.

MPI ownership matters: launcher, `mpi4py`, PETSc, and SLEPc must agree on one
MPI stack. Installer checks and runtime wrappers should report the chosen stack
rather than guessing silently.

## Runtime And NPZ Output

Example scripts are thin adapters around `linstabpy.driver.run`. The driver
owns stage timing, postcheck logs, acceptance metadata, and the saved `.npz`.

The compatibility surface for saved results is `RUNTIME_NPZ_CONTRACT_KEYS` in
`driver/run.py`. Additive fields are acceptable. Renames or removals update
that tuple, the write test, and every reader.

## Mesh State

`Mesh` remains the main structured-grid object. `MeshGeometry`,
`MeshOperators`, and `MeshRuntimeState` are read-only grouping views over the
same storage; they make ownership clearer without changing callers that still
use the historical attributes directly.

Deformation and masking must preserve numerical behavior. Geometry-only
optimizations are allowed only when tests prove that derivative operators,
filters, weights, and saved outputs remain equivalent for the relevant cases.

