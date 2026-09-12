# LinStabPy

`linstabpy` builds the linearized compressible Navier–Stokes operators on
two-dimensional structured base flows and solves the eigenvalue and resolvent
problems through PETSc/SLEPc with sparse direct factorization (MUMPS).

State vector: \(q = [\rho, u, v, w, T]\). Spatial discretization is
summation-by-parts finite differences of order 2, 4, 6, or 8. A plain
`import linstabpy` does not start PETSc or MPI.

This site is the code-structure wiki. Numerical policy lives in the owning
modules; the tables here name those owners. Do not treat Litho C4 prose as
the contract.

| Page | What it is |
|---|---|
| [Overview](1.Overview.md) | What the package computes and what it does not |
| [Architecture](2.Architecture.md) | Ownership map and module graph |
| [Workflow](3.Workflow.md) | `driver.run` stages from YAML to NPZ |
| [Deep exploration](4.Deep-Exploration/modules.md) | Per-package notes |
| [Boundaries](5.Boundary-Interfaces.md) | YAML, CLI, environment, NPZ keys |

Human notes that still govern numerics:

- [`docs/architecture.md`](https://github.com/openfluids/linstabpy/blob/main/docs/architecture.md)
- [`docs/COMPUTE_BACKENDS.md`](https://github.com/openfluids/linstabpy/blob/main/docs/COMPUTE_BACKENDS.md)
- [`docs/runtime_npz_contract.md`](https://github.com/openfluids/linstabpy/blob/main/docs/runtime_npz_contract.md)
