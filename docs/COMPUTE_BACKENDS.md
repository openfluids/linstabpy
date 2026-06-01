# Compute Backends

`linstabpy` uses PETSc LU factorization for sparse-direct solver paths. The
default factor backend is `mumps`.

Precedence for the factor backend is:

1. JSONC solver subsection `factor_backend`;
2. existing per-solver environment override, such as
   `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE`;
3. `LINSTABPY_FACTOR_BACKEND`;
4. the call-site default.

The temporal and spatial EVP direct paths intentionally preserve the historical
asymmetry: `st_solver = "mumps"` forces `mumps`, while `st_solver = "direct"`
or `"lu"` leaves PETSc's native LU backend unless a factor backend is explicitly
configured. The linear-solve direct path keeps forcing `mumps` for
`type = "mumps"`, `"direct"`, and `"lu"`.

Audit note for the original hardcoded sites:

- `src/linstabpy/mesh/filters.py`: the alpha-filter solve is a sequential
  mesh-filter LU used when filters are enabled, including default-numerics
  validation paths. It remains pinned to `mumps` and does not follow the global
  factor-backend knob.
- `src/linstabpy/solvers/eigenvalue.py`: temporal ST direct path.
- `src/linstabpy/solvers/spatial_eig.py`: spatial ST direct path.
- `src/linstabpy/solvers/linsolve.py`: reusable KSP/PC direct solve path used
  by resolvent solves.

Requested backends are checked with a tiny PETSc LU setup probe. An unavailable
backend fails loudly instead of falling back silently.
