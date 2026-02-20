# Branch Coverage Matrix

Canonical machine-readable source:
- `tools/parity/branch_matrix.json`

Target dimensions:
- `model`: `2D`, `axy`
- `solver`: `temporal`, `spatial`, `adjoint`, `resolvent`
- `bc`: `Dirichlet`, `DirichletSpatial`, `Neumann`
- `deformation`: `none`, `x_only`, `y_only`, `full_2d`
- `backend`: `petsc_slepc`

Total combinations:
- `2 * 4 * 3 * 4 * 1 = 96`

Required gate:
- All required combinations must be `validated`.
- Combinations omitted from `records` default to `pending`.
- Scope is controlled by `required_exclusions` in `tools/parity/branch_matrix.json`.
- Current scoped gate status: `pass` (required validated combos = `5`, required exclusions = `91`).

Current explicitly tracked combinations:

| model | solver | bc | deformation | backend | status |
|---|---|---|---|---|---|
| `2D` | `temporal` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `spatial` | `DirichletSpatial` | `none` | `petsc_slepc` | `validated` |
| `2D` | `adjoint` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `resolvent` | `Dirichlet` | `none` | `petsc_slepc` | `validated` |
| `2D` | `temporal` | `Neumann` | `none` | `petsc_slepc` | `validated` |

Gate command:
```bash
python tools/parity/check_parity_matrices.py --allow-incomplete
```

Strict CI gate:
```bash
python tools/parity/check_parity_matrices.py
```
