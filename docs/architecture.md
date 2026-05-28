# Architecture Map

This note tells maintainers where numerical policy lives after the refactor.
Keep policy in the owning seam rather than recreating pass-through modules.

## Ownership

| Area | Owning module | What belongs there |
|---|---|---|
| Boundary-condition recipes | `linstabpy.boundary_conditions.recipe` | JSONC-compatible recipe parsing, rule validation, effective rows, column-zeroed Dirichlet rows, overlap reporting, and SciPy/PETSc/spatial dispatch. |
| LNSE assembly | `linstabpy.operators.assembly` | Coefficient preparation, backend choice, operator construction order, boundary-condition application, and the `LNSESystem` result object. |
| Solver policy | `linstabpy.utils._solver_cfg`, `linstabpy.solvers.*` | Dict/env compatibility, PETSc/SLEPc defaults, MUMPS/KSP/EPS setup, root-scatter decisions, and resolvent linear-solver selection. |
| Runtime pipeline | `linstabpy.results_runtime` | Case loading, mesh construction, Chu norm, assembly, stability/resolvent stages, postchecks, acceptance, and `.npz` persistence. |
| Mesh state | `linstabpy.mesh.mesh` | Structured geometry, derivative operators, filters, dissipation state, sponge state, and conservative read-only views over those groups. |

## Boundary Conditions

`BCRecipe` is the boundary-condition seam. Public JSONC files may continue to
pass lists of dictionaries; the recipe layer normalizes them before application.
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

Case-study wrappers remain thin adapters around `linstabpy.results_runtime`.
They install case-specific environment defaults and call `compute()` or
`main()`. The shared runtime owns stage timing, postcheck logs, acceptance
metadata, and the saved `.npz` contract.

The compatibility surface for saved results is
`RUNTIME_NPZ_CONTRACT_KEYS`; the grouped explanation is in
`docs/runtime_npz_contract.md`. Additive fields are acceptable. Renames or
removals require a migration note, test updates, and reader updates.

## Mesh State

`Mesh` remains the main structured-grid object. `MeshGeometry`,
`MeshOperators`, and `MeshRuntimeState` are read-only grouping views over the
same storage; they make ownership clearer without changing callers that still
use the historical attributes directly.

Deformation and masking must preserve numerical behavior. Geometry-only
optimizations are allowed only when tests prove that derivative operators,
filters, weights, and saved outputs remain equivalent for the relevant cases.

## Refactor Safety Gates

Before changing any seam above, run the relevant slice of
`docs/architecture_refactor_test_matrix.md`. For operator or kernel work, also
run the LNSE equivalence harness so coefficient tensors, pre-BC operators,
post-BC operators, boundary-condition index sets, and downstream numerical
checks move together.
