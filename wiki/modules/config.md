# YAML and typed Config

Owner: `linstabpy.config.typed`. Shared lookup helpers also live in
`linstabpy.utils._solver_cfg`.

`Config` reads YAML and environment variables. PETSc/SLEPc objects are
still created inside the solver modules, but the defaults and fallback
choices come from that shared policy layer.

| File | Role |
|---|---|
| `typed.py` | Typed `Config` |
| `defaults.py` | Numerical defaults, including dissipation policy |
| `schema.py` | YAML shape |

Backend selection for sparse-direct factor (see
[`docs/COMPUTE_BACKENDS.md`](https://github.com/openfluids/linstabpy/blob/main/docs/COMPUTE_BACKENDS.md)):

On the reference direct path (`st_solver` of `mumps`/`direct`/`lu`) the
resolution order is `solver.eigen.factor_backend`, then
`LINSTABPY_FACTOR_BACKEND`, then the `mumps` default. The per-solver
`LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE` override is *not* consulted on
that path. It participates only for the iterative eigensolver with
`pc_type="lu"` and for `st_solver="strumpack_fgmres"` (whose default
is `strumpack`).

An unavailable backend fails a tiny complex PETSc LU probe instead of
falling back silently. Default factor is `mumps`.
