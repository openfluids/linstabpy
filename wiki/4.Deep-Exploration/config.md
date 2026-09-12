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
| `validation_cases.yaml` | Named validation cases |

Backend selection for sparse-direct factor (see
[`docs/COMPUTE_BACKENDS.md`](https://github.com/openfluids/linstabpy/blob/main/docs/COMPUTE_BACKENDS.md)):

1. YAML solver subsection `factor_backend`
2. Per-solver environment override, e.g. `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE`
3. `LINSTABPY_FACTOR_BACKEND`
4. Call-site default

An unavailable backend fails a tiny complex PETSc LU probe instead of
falling back silently. Default factor is `mumps`.
