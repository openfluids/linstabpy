# Comment And Docstring Audit

This audit is the baseline for the release-facing comment and docstring pass.
It records where comments must protect numerical behavior, and where extra prose
would only add noise. The goal is maintainable scientific code, not decorative
coverage.

## Scope

The current AST inventory command is:

```bash
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --check docstring-inventory
```

Current public-symbol inventory:

| Kind | Total | Required | Missing required |
|---|---:|---:|---:|
| Modules | 42 | 40 | 0 |
| Classes | 8 | 5 | 0 |
| Functions and methods | 265 | 83 | 0 |
| All symbols | 315 | 128 | 0 |

This means public module/class/function docstrings are present. The remaining
work is about quality and invariant coverage in high-risk internals.

## High-Risk Surfaces

| Area | Files | Current risk |
|---|---|---|
| LNSE assembly | `src/linstabpy/operators/assembly.py`, `coefficients.py`, `spatial.py` | The five-stage operator construction order is documented, but many private helpers have no local invariant note. P1/P2/P3 work must add comments where coefficient fusion, block assembly, or derivative materialization changes the order or allocation pattern. |
| Boundary conditions | `src/linstabpy/boundary_conditions/*.py` | Public behavior is documented, but the future BCRecipe seam must make effective rows, column-zeroed rows, overlaps, corners, and curvilinear normals explicit at the seam. |
| Solver policy | `src/linstabpy/solvers/*.py`, `src/linstabpy/utils/_solver_cfg.py` | Module docstrings explain the current SLEPc/MUMPS paths. SolverConfig work must document where defaults are resolved and which choices affect PETSc/SLEPc object construction. |
| Runtime and NPZ provenance | `src/linstabpy/results_runtime.py` | The module has a public runtime docstring, but stage boundaries and NPZ schema ownership are still scattered. The runtime split must document stage outputs and saved-key compatibility. |
| Mesh state | `src/linstabpy/mesh/mesh.py`, `filters.py`, `sbp*.py` | Mesh geometry, operators, filters, sponge, and dissipation live on the same object. Any mutation-reduction work must document which fields are geometry, which are derived operators, and what can be reused after deformation. |
| PETSc/MPI init | `src/linstabpy/utils/petsc_init.py`, `petsc_core.py` | Existing comments explain lazy PETSc access and argv sanitization. Keep future comments focused on import/init ordering and MPI binding assumptions. |
| Case-study wrappers | `examples/case_studies/**`, `studies/case2/**` | Comments should clarify run configuration, cache behavior, and provenance. Machine-specific notes stay outside the public package. |

## Helper Counts In Focus Areas

These counts are a navigation aid for the comment pass, not a coverage quota.

| File | Lines | Functions | Classes | Private nodes without docstrings |
|---|---:|---:|---:|---:|
| `operators/assembly.py` | 1734 | 40 | 3 | 22 |
| `boundary_conditions/recipe.py` | 249 | 4 | 0 | 0 |
| `boundary_conditions/dirichlet.py` | 353 | 10 | 0 | 1 |
| `boundary_conditions/neumann.py` | 285 | 4 | 0 | 0 |
| `solvers/eigenvalue.py` | 1061 | 20 | 0 | 3 |
| `solvers/resolvent.py` | 486 | 12 | 1 | 4 |
| `solvers/linsolve.py` | 345 | 9 | 1 | 5 |
| `solvers/_eps_helpers.py` | 134 | 6 | 0 | 1 |
| `results_runtime.py` | 1335 | 22 | 0 | 2 |
| `mesh/mesh.py` | 1580 | 17 | 1 | 1 |
| `utils/petsc_init.py` | 111 | 4 | 0 | 1 |
| `utils/petsc_core.py` | 762 | 26 | 0 | 2 |

## Comment Rules

Use module docstrings for:

- public purpose and scope;
- numerical convention choices, such as eigenvalue sign and state ordering;
- backend lifecycle assumptions, such as PETSc/SLEPc initialization.

Use function/class docstrings for:

- public APIs;
- compatibility wrappers that preserve old dict/list protocols;
- private helpers selected by `--require-private` because they encode policy.

Use inline invariant comments only when moving or editing the line could break a
numerical contract. Good targets are:

- inv(B), sponge, artificial dissipation, and boundary-condition ordering;
- effective versus column-zeroed boundary rows;
- PETSc local versus distributed matrix ownership;
- root scatter and gathered postcheck paths;
- NPZ key/schema compatibility;
- mesh deformation and derivative-operator reuse.

Do not comment:

- obvious assignments or control flow;
- one-line helper mechanics that tests already make clear;
- generic PETSc, SLEPc, NumPy, or SciPy behavior;
- future plans that belong in tracked issues or architecture documents.

## Terminology

Use these terms consistently:

- `BC recipe`
- `effective rows`
- `column-zeroed rows`
- `overlap rows`
- `LNSE system`
- `PETSc local matrix`
- `PETSc distributed matrix`
- `postcheck residual`
- `target distance`
- `NPZ provenance`

## Release Hygiene

Public code and docs must not mention internal review tools, private planning
files, machine hostnames, or personal paths. When a comment needs provenance,
cite source files, validation commands, artifacts, and commit SHAs instead.
