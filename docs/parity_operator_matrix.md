# Operator Parity Matrix

Canonical machine-readable source:
- `tools/parity/operator_matrix.json`

Status semantics:
- `pending`: not implemented yet
- `implemented`: code exists but parity artifact/signoff is still pending
- `validated`: implemented and parity-validated by artifact(s)
- `deferred`: intentionally postponed
- `out_of_scope`: intentionally not targeted
- `blocked`: cannot proceed due to external blocker

Required gate:
- Every `required: true` entry must reach `status = validated`.

Current mapped entrypoints (MATLAB -> Python):

| ID | MATLAB | Python target | Status |
|---|---|---|---|
| `construct_get_lin_problem` | `ConstructOperators/GetLinProblem.m` | `linstab2d/operators/assembly.py::build_lnse_operator` | `implemented` |
| `construct_get_spatial_lin_problem` | `ConstructOperators/GetSpatialLinProblem.m` | `linstab2d/operators/spatial.py::build_spatial_operator` | `implemented` |
| `construct_cartesian_coefficients` | `ConstructOperators/getCoeffsLaminar_Cartesian.m` | `linstab2d/operators/coefficients.py::compute_cartesian_coefficients` | `implemented` |
| `construct_axisymmetric_coefficients` | `ConstructOperators/getCoeffsLaminar.m` | `linstab2d/operators/coefficients.py::compute_cylindrical_coefficients` | `implemented` |
| `construct_create_diff_matrices_axy` | `ConstructOperators/CreateDiffMatrices_Axy.m` | `linstab2d/operators/assembly.py::_create_axy_diff_matrices` | `implemented` |
| `construct_lnseoperators_cyl` | `ConstructOperators/LNSEoperators_cyl.m` | `linstab2d/operators/assembly.py::build_lnse_operator(model="axy")` | `implemented` |
| `construct_get_comp_energy_norm` | `ConstructOperators/GetCompEnergyNorm.m` | `linstab2d/operators/energy_norm.py::compute_chu_norm` | `implemented` |
| `bc_dirichlet` | `BoundaryConditions/BC_Dirichlet.m` | `linstab2d/boundary_conditions/dirichlet.py::apply_dirichlet` | `implemented` |
| `bc_dirichlet_spatial` | `BoundaryConditions/BC_Dirichlet_SpatialStability.m` | `linstab2d/boundary_conditions/dirichlet.py::apply_dirichlet_spatial` | `implemented` |
| `bc_neumann` | `BoundaryConditions/BC_Neumann.m` | `linstab2d/boundary_conditions/neumann.py::apply_neumann` | `implemented` |
| `bc_set_dirbc` | `BoundaryConditions/SetDirBC.m` | `linstab2d/boundary_conditions/dirichlet.py::build_dirichlet_restriction_matrices` | `implemented` |
| `bc_set_dirichletbc` | `BoundaryConditions/SetDirichletBC.m` | `linstab2d/boundary_conditions/dirichlet.py::apply_dirichlet_identity_pair` | `implemented` |

Gate command:
```bash
python tools/parity/check_parity_matrices.py --allow-incomplete
```

Strict CI gate:
```bash
python tools/parity/check_parity_matrices.py
```
