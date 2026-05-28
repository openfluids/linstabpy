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

Current public-symbol plus required-helper inventory:

| File group | Count |
|---|---:|
| Tracked files | 632 |
| Tracked Python files | 207 |
| `src/linstabpy` Python files | 42 |
| Test Python files | 49 |
| Example Python files | 53 |
| Validation Python files | 58 |
| Documentation Markdown files | 4 |

| Kind | Total | Required | Missing required |
|---|---:|---:|---:|
| Modules | 42 | 40 | 0 |
| Classes | 8 | 5 | 0 |
| Functions and methods | 265 | 90 | 0 |
| All symbols | 315 | 135 | 0 |

This means public module/class/function docstrings are present, and the current
matrix also requires selected private helpers that already own numerical or
runtime invariants. The remaining work is about quality and adding new helper
requirements as the architecture seams move.

## High-Risk Surfaces

| Area | Files | Current risk |
|---|---|---|
| LNSE assembly | `src/linstabpy/operators/assembly.py`, `coefficients.py`, `spatial.py` | The five-stage operator construction order is documented, but many private helpers have no local invariant note. P1/P2/P3 work must add comments where coefficient fusion, block assembly, or derivative materialization changes the order or allocation pattern. |
| Boundary conditions | `src/linstabpy/boundary_conditions/*.py` | `BCRecipe` now owns effective rows, column-zeroed rows, overlaps, corners, curvilinear normals, and backend dispatch. Future comments should protect those distinctions rather than repeat row-loop mechanics. |
| Solver policy | `src/linstabpy/solvers/*.py`, `src/linstabpy/utils/_solver_cfg.py` | Module docstrings explain the current SLEPc/MUMPS paths. `SolverConfig` owns where defaults are resolved and which choices affect PETSc/SLEPc object construction. |
| Runtime and NPZ provenance | `src/linstabpy/results_runtime.py` | The runtime now has explicit stage helpers and a checked NPZ contract. Comments should explain stage outputs, saved-key compatibility, and provenance decisions. |
| Mesh state | `src/linstabpy/mesh/mesh.py`, `filters.py`, `sbp*.py` | Mesh geometry, operators, filters, sponge, and dissipation remain on the same object, with grouped state views for contributor navigation. Comments should say which fields are geometry, which are derived operators, and what can be reused after deformation. |
| PETSc/MPI init | `src/linstabpy/utils/petsc_init.py`, `petsc_core.py` | Existing comments explain lazy PETSc access and argv sanitization. Keep future comments focused on import/init ordering and MPI binding assumptions. |
| Case-study wrappers | `examples/case_studies/**` | Comments should clarify run configuration, cache behavior, and provenance. Machine-specific notes stay outside the public package. |

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

## Required Private Helpers

The documentation gate currently requires docstrings on these private helpers
because they encode behavior that later refactors must preserve:

- `linstabpy.operators.assembly._prepare_coefficients`
- `linstabpy.operators.assembly._assemble_from_prepared`
- `linstabpy.operators.assembly._apply_inv_B`
- `linstabpy.boundary_conditions.recipe._apply_bc_recipe_impl`
- `linstabpy.results_runtime._restrict_operator_for_stability`
- `linstabpy.utils.petsc_init._sanitize_petsc_argv`
- `linstabpy.solvers.eigenvalue._configure_st_for_eigenvalue`

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

## Comment And Docstring Policy

The release standard is selective coverage with explicit reasons. A missing
comment is acceptable when the code is simple and the surrounding tests or type
names already state the contract. A missing comment is not acceptable when the
code owns numerical policy, backend lifecycle, or saved-result compatibility.

### Public API Docstrings

Public modules, classes, functions, and methods under `src/linstabpy` need
docstrings that describe what the caller can rely on. Good public docstrings
state the purpose, important inputs and outputs, numerical conventions, side
effects, and failure modes. They should not mechanically repeat type hints or
list obvious parameter names.

Examples of useful public coverage:

- `BCRecipe` explains effective rows, column-zeroed rows, overlap reporting,
  and backend dispatch because those are public compatibility contracts.
- `LNSESystem` describes the assembled handoff instead of forcing callers to
  infer which sparse matrices and index maps travel together.
- `SolverConfig` documents where dict/env defaults are interpreted before
  PETSc/SLEPc objects are created.

### Private Numerical-Policy Helpers

Private helpers need a docstring or a nearby invariant comment when moving the
helper would change a numerical contract. These helpers should also appear in
the docstring gate's required-private list when they become stable ownership
points.

Require comments around helpers that own:

- boundary-condition ordering, especially effective rows versus
  column-zeroed Dirichlet rows;
- PETSc/MPI initialization, launcher binding, or local/distributed matrix
  ownership;
- solver acceptance, including postcheck residuals, target distance, accepted
  families, and root-gathered diagnostics;
- mesh mutation, deformation, masking, filter reuse, or derivative-operator
  rebuild decisions;
- NPZ schema compatibility and provenance keys;
- sparse-assembly order, coefficient fusion, one-pass block construction, or
  inverse-mass application.

Good invariant comments say why the line is placed where it is. For example:
the inverse mass/metric block must be applied before sponge, dissipation, and
boundary conditions; the postcheck must compare against the reduced operator
used by the solve; and saved NPZ keys may be added but not renamed without a
migration path.

### Simple Private Helpers

Do not add comments to private helpers that are only small adapters, local
formatting utilities, simple array conversions, or one-line path/env lookups.
If a scanner reports one of these, the audit should mark it as intentionally
private/simple rather than forcing a decorative docstring.

Examples that usually should stay uncommented:

- a helper that normalizes a local string before printing;
- a one-line wrapper around `np.asarray`;
- a local path builder whose behavior is already covered by the caller's
  docstring and tests.

### Examples And Scripts

Examples and validation scripts should have module-level or entry-point
docstrings that say what case is being run, which data are optional, whether an
existing cache is loaded or recomputed, and where outputs are written. Inline
comments belong only near non-obvious numerical setup, MPI launch behavior,
optional external assets, or cache/provenance decisions.

Avoid turning examples into tutorials inside the source. Reader-facing
instructions belong in README files or case-study docs; source comments should
protect the run contract.

### Tests

Test names, parametrization IDs, and assertion messages should carry most of
the documentation burden. Add comments only when a tolerance, fixture, skip, or
reference artifact would otherwise look arbitrary. Do not narrate pytest
mechanics or restate the assertion expression in prose.

### Generated And Cache Artifacts

Generated reports, example `.npz` files, cache directories, and regenerated
figures are not hand-commented. Document their regeneration command, commit or
ignore policy, and expected location instead. If a generated artifact becomes
part of a release fixture, record its source command and SHA-bearing provenance
in the relevant release audit.

### Comments To Avoid

Remove or reject comments that only narrate the code:

- "Loop over rows."
- "Set the variable."
- "Calculate the result."
- "Convert to array."

Also avoid generic transition prose and claims that sound detached from the
code path, such as "it is important to note" or "this robustly handles all
cases" without a specific invariant or test. Public comments should cite code,
validation commands, source papers, or artifact contracts rather than internal
planning history.

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
