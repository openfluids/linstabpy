# Runtime driver

Owner: `linstabpy.driver.run`.

The driver package `__init__` exports nothing so `import linstabpy.driver`
never pulls PETSc. `cli.py` handles `--version` without initializing PETSc,
then imports `driver.run`.

## Files

| File | Role |
|---|---|
| `run.py` | Pipeline stages, `run()` |
| `cli.py` | Cache-aware entry, `--version` |
| `payload.py` | `build_runtime_payload`, `RUNTIME_NPZ_CONTRACT_KEYS` |
| `io.py` | NPZ/sidecar write. Does not import the driver |
| `preflight.py` | Non-blocking warnings (dissipation mismatches, shadowed `factor_backend`, band-marking trap, freestream scale, opaque MUMPS INFOG(1)) |

## Dirichlet reduction

Stability may drop Dirichlet-constrained DOFs from the operator and Chu
weights, then scatter eigenvectors back into the full vector. Constrained
entries stay zero. `reduce_dirichlet_set` chooses which index set is
removed.

## Acceptance

`stability.acceptance` applies residual, target-distance, and omega-bound
rules. Conjugate-aware mode admits the conjugate partner of a real-valued
baseflow at the same `target_distance_max`.
