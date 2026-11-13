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

## Rule syntax

Rules are YAML-friendly and applied **sequentially**. Later rules own overlapping rows. That order is part of the numerical contract (there is no separate precedence field). The methods that apply a recipe to each backend are in the dispatch table above.

**Border characters:** `l` left, `r` right, `b` bottom, `t` top, `m` mask/inner
**Variable characters:** `r` ρ, `u`, `v`, `w`, `T`

Example recipe:

```yaml
bc_recipe:
  - type: dirichlet
    borders: "lt"
    variables: "ruvwT"
  - type: neumann
    borders: "b"
    variables: "T"
    exclude_corners: true
```

## `BCRule` and `BCRecipe`

- `BCRule.from_mapping`, normalize and validate one rule
- `BCRecipe.from_config`, list/dict or existing recipe
## Dirichlet implementation (`dirichlet.py`)

Ports `BC_Dirichlet.m`:

1. Collect DOF indices for all (border, variable) pairs.
2. Zero **rows** and **columns** at those DOFs (remove equation and remove constrained variable from interior couplings, MATLAB parity).
3. Set diagonal entries to `penalty` (default `1e4`).
4. `eliminate_zeros` to keep sparsity clean.

Also provides spatial helpers that apply Dirichlet on the doubled EVP after \(L_c + \omega L_w\) is formed, and optional restriction matrices / free-DOF utilities.

## Neumann implementation (`neumann.py`)

Ports `BC_Neumann.m`:

- Supported borders: `b, t, l, r` only (not `m`).
- Replaces boundary rows with scaled normal-derivative stencils from `mesh.Dx` / `mesh.Dy`.
- **Cartesian-mapped normals** from geometry gradients, or **`curvilinear_normals=True`** using mesh metric terms `J11…J22`.
- `exclude_corners` drops shared corner points from a border.
- Ambiguous multi-border corner DOFs raise `ValueError` instead of silently picking one normal.

## Analysis output (`analyze_bc_recipe`)

Useful fields for free-DOF construction and diagnostics:

- `dirichlet_indices` / `neumann_indices`
- `effective_dirichlet_indices` / `effective_neumann_indices` (after sequential ownership)
- `column_zeroed_dirichlet_indices` (all Dirichlet rules that ran, even if a later [Neumann](https://grokipedia.com/page/Neumann_boundary_condition) overwrote the row)
- `overlap_indices`, `total_bc_indices`, per-rule reports

## Lazy import policy

`boundary_conditions/__init__.py` uses PEP 562 `__getattr__` so pure-math entry points do not force `petsc4py` load. PETSc variants import petsc4py inside their modules.
