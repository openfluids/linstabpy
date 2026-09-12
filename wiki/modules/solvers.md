# SLEPc, maps, factorization

Solver policy owner: `utils._solver_cfg` and `config.typed`.
Objects: created in `solvers.*` and `linalg.factorize`.

| File | Role |
|---|---|
| `solvers.eigenvalue` | SLEPc EPS wrapper used by temporal EVP |
| `solvers.resolvent` | Linear solves for resolvent |
| `solvers.spatial_eig` | Spatial EVP |
| `solvers.petsc_utils` | Shared PETSc helpers |
| `solvers.router.resolvent_map` | \(\beta\)–frequency gain map |
| `solvers.router.ram_autoselect` | RAM-aware backend choice |
| `solvers.router.stable_sweep` | 1-D frequency-line tool |

Reference sparse-direct path is MUMPS. GPU SuperLU_DIST complex CUDA was
measured and rejected (order-one residuals). See
[`docs/COMPUTE_BACKENDS.md`](https://github.com/openfluids/linstabpy/blob/main/docs/COMPUTE_BACKENDS.md).
