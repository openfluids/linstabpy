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

Current explicitly tracked combinations:

| model | solver | bc | deformation | backend | status |
|---|---|---|---|---|---|
| `2D` | `temporal` | `Dirichlet` | `none` | `petsc_slepc` | `implemented` |
| `2D` | `spatial` | `DirichletSpatial` | `none` | `petsc_slepc` | `implemented` |
| `2D` | `adjoint` | `Dirichlet` | `none` | `petsc_slepc` | `implemented` |
| `2D` | `resolvent` | `Dirichlet` | `none` | `petsc_slepc` | `implemented` |
| `2D` | `temporal` | `Neumann` | `none` | `petsc_slepc` | `implemented` |

Gate command:
```bash
python tools/parity/check_parity_matrices.py --allow-incomplete
```

Strict CI gate:
```bash
python tools/parity/check_parity_matrices.py
```
