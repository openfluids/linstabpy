# LNSE assembly

Owner: `linstabpy.operators.assembly`.

Public names: `build_lnse_system`, `build_lnse_operator`,
`build_lnse_system_result`. State \(q = [\rho, u, v, w, T]\).

## `LNSESystem`

The assembly handoff. It carries the assembled operator, index map,
boundary-condition indices, and recipe analysis. Runtime callers ask for
this object. They do not rebuild pieces of the sequence.

## Construction order

1. Prepare coefficients (`operators.coefficients`, physical-order packing).
2. Build the raw block operator from SBP derivatives.
3. Apply the inverse mass/metric block (`_apply_inv_B`).
4. Apply sponge and operator dissipation.
5. Apply boundary conditions last.

Backends: SciPy CSC, or PETSc Mat. Direct PETSc assembly cannot apply
`dissipation_epsilon`; `auto`/`petsc` may SciPy-assemble then convert.

## Other files

| File | Role |
|---|---|
| `coefficients.py` | Block tensors `A0, Ax, Ay, Axx, Ayy, Axy, B` |
| `energy_norm.py` | Chu compressible energy inner product |
| `metrics.py` | Curvilinear metric terms |
| `spatial.py` | Spatial-EVP left/right operators |

Floquet shift of streamwise derivatives is applied inside assembly when
configured. Do not apply it a second time in the driver.
