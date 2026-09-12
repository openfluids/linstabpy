# Eigenvalues and acceptance

| File | Role |
|---|---|
| `stability._eigenvalues` | Direct and adjoint temporal EVP (`temporal_eigenvalues`) |
| `stability.sensitivity` | Wavemaker from direct and adjoint modes |
| `stability.acceptance` | Residual, target-distance, omega-bound policy |
| `solvers.spatial_eig` | Spatial (streamwise-wavenumber) EVP |

The driver postchecks returned modes: residual norms, Dirichlet leakage,
distance to `omega_target`. Acceptance may be conjugate-aware for
real-valued baseflows.

SLEPc objects are created in the solver modules. Shift-invert uses the
sparse-direct factor from `linalg.factorize`.
