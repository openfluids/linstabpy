# Boundary-condition recipes

Owner: `linstabpy.boundary_conditions.recipe`.

`BCRecipe` is the seam. Public YAML files pass lists of dictionaries. The
recipe layer normalizes them before application.

## Rules

`BCRule.from_mapping` validates type, borders, and variables. Overlap
resolution picks the **effective rows** owned by the last matching rule.
**Column-zeroed Dirichlet rows** are the broader set needed for operator
reduction and provenance.

When you change boundary-condition behavior, update the recipe analysis and
every backend wrapper together. Do not duplicate row-selection logic in
assembly, runtime code, or example scripts.

## Dispatch

| Method | Backend |
|---|---|
| `apply_to` | SciPy CSC temporal operator |
| `apply_to_petsc` | PETSc Mat |
| `apply_to_spatial` | SciPy spatial pair `(Lc, R)` |
| `apply_to_spatial_petsc` | PETSc spatial pair |
| `analyze` | Index sets only, no matrix write |

Dirichlet implementation: `boundary_conditions.dirichlet` (penalty plus
column zeroing). Neumann: `boundary_conditions.neumann`.
