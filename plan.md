# LinStabPy Simplification Plan

Date: 2026-04-24
Mode: implementation started from the recommended first slice.

## Implementation Log

- 2026-04-24: tutorial commit `0e0ff7e` is preserved on `main`.
- 2026-04-24: default branch migrated from `master` to `main` on GitHub.
- 2026-04-24: implementation branch `cleanup/2026-04-layout` created.
- 2026-04-24: first slice completed: file-size guardrail, path helper
  contract, operator-fixture schema, README guardrail notes, and removal of the
  two existing over-50 MiB raw MATLAB exports from the Git index while keeping
  the local files ignored.

## Goal

Make `linstabpy` a professional scientific Python project without weakening
the numerical validation:

- `README.md` is the canonical project documentation after the Markdown
  consolidation.
- `src/linstab2d` remains the only importable library source.
- Stale source copies are removed.
- Active research workflows move out of the overloaded `results/` namespace.
- MATLAB-backed operator validation is preserved through compact Python-readable
  fixtures.
- No tracked GitHub-facing file exceeds 50 MB.
- Numerical behavior changes happen only behind tests or parity checks.

## Baseline From This Checkout

Repository state:

- The worktree is dirty. Current changes include deleted old Markdown files,
  deleted `paper/` files, modified `README.md`, modified parity reports,
  untracked Case 2 configs, untracked audit tools, and this plan.
- The branch is still `master...origin/master`; repo policy says the default
  branch should be `main`. Branch migration happens as the first commit of
  Phase 1 (see below).
- `README.md` is now the documentation consolidation target. Do not recreate a
  broad `docs/` tree unless a specific long-form reference needs it.

Size and structure:

- Tracked file counts are dominated by non-package material: `validation` 258,
  `results` 184, `install_tester` 71, `src` 52, `tests` 39, `tools` 18,
  `examples` 17.
- Working-tree disk usage is concentrated outside `src`: `validation` ~1.2 GB,
  `results` ~822 MB, `examples` ~426 MB, while `src` is ~10 MB.
- Tracked MATLAB/NumPy-style data: 202 `.mat`, `.npy`, `.npz`, or `.dat` files,
  about 1.9 GB total.
- Two tracked files currently violate the 50 MB GitHub constraint:
  `results/case1/MATLAB_exports/operator_convention.mat` (~88 MB) and
  `validation/song_2023/matlab_code/SolverMartini/MATLAB_exports_Ziming_Internal/coefficients_raw.mat`
  (~74 MB).
- Generated local example artifacts are also large, but ignored: for example
  `examples/example_3_streaky_bl.npz` is ~254 MB and
  `examples/example_4_cavity.npz` is ~120 MB.

Code layout:

- `pyproject.toml` already uses `src` layout, but package data only lists
  `linstab2d = ["data/baseflows/*.mat"]` while the actual upstream baseflow
  submodule lives under `src/linstab2d/data/baseflows/BaseFlows`.
- Heaviest package modules by line count are:
  - `src/linstab2d/operators/assembly.py` (~1535 lines)
  - `src/linstab2d/mesh/mesh.py` (~1343 lines)
  - `src/linstab2d/results_runtime.py` (~1169 lines)
  - `src/linstab2d/operators/coefficients.py` (~1115 lines)
  - `src/linstab2d/solvers/eigenvalue.py` (~869 lines)
  - `src/linstab2d/utils/petsc_core.py` (~711 lines)
- Most of the large numerical modules are real complexity. The clearest
  simplification target is `results_runtime.py`, which is package code for
  `results/case*` wrappers rather than a clean library API
  (`src/linstab2d/results_runtime.py:1-4` and `compute()` at line 404).
- `src/linstab2d/baseflow/case1.py` is study-specific Hamada/Case 1-2 loading
  inside the library, and even references `results/case2/PLAN.md` in a code
  comment (`src/linstab2d/baseflow/case1.py:97-107`).
- `src/linstab2d/solvers/petsc_utils.py` is a compatibility re-export of
  `linstab2d.utils.petsc_core`, so it is not urgent but should eventually
  disappear after import cleanup.

Path coupling:

- `src/linstab2d/config/validation_cases.yaml` still points to
  `results.case1.case1` and `results/case1/case1.npz`.
- `README.md` still contains many `results/case*` instructions.
- Tests hardcode multiple incompatible Case 1 locations:
  `tests/test_case1_parity.py` and `tests/test_case1_resolvent_parity.py`
  use `results/case1`, `tests/test_case1_native_sponge.py` uses
  `results/case1`, while `tests/test_case1_operator_parity.py` now uses
  root-level `case1`.
- `examples/_helpers.py` points MATLAB parity references at
  `tools/parity/npz`, but that directory is absent; parity reports currently
  record missing MATLAB NPZ bundles.
- Several validation and study scripts inject repo paths into `sys.path`.
  That is acceptable for research scripts, but package tests should not depend
  on repo-root import accidents.

## Target Structure

```text
.
  README.md
  LICENSE
  pyproject.toml
  uv.lock
  plan.md
  src/linstab2d/
  tests/
  examples/
  scripts/
  studies/
    case1/
    case2/
  validation/
    operator_fixtures/
    martini2024/
    song_2023/
```

Directory meaning:

- `src/linstab2d/`: library code only. No generated outputs, no committed
  study outputs, no install-tester source copy.
- `examples/`: runnable examples and `.jsonc` configs. Local `.npz` and `.png`
  outputs remain ignored.
- `tests/`: unit and integration tests. Tests should load small fixtures through
  a central helper, not hardcoded `results/case*` paths.
- `scripts/`: repo operations: install, diagnose, data conversion, size checks,
  and bundle generation.
- `studies/`: active research/campaign workflows. This replaces the current
  overloaded `results/case*` role.
- `validation/operator_fixtures/`: curated MATLAB-generated operator references
  converted to `.npz` or SciPy sparse `.npz`.
- `validation/martini2024/`: Martini example parity bundles and manifests.
  Fixtures live under `validation/martini2024/fixtures/`.
- `validation/song_2023/`: Song validation harness and only the data that passes
  the fixture/data policy.

## Non-Goals

- Do not rename the import package from `linstab2d` to `linstabpy` in this
  cleanup.
- Do not split `operators/assembly.py`, `mesh/mesh.py`, or
  `operators/coefficients.py` until operator fixtures and tests make that safe.
  See "Follow-up Work" at the end of this file.
- Do not change PETSc/SLEPc settings, matrix assembly, boundary conditions, or
  acceptance thresholds.
- Do not delete MATLAB reference data before preserving the operator-validation
  subset.
- Do not keep tracked GitHub-facing files over 50 MB.

## Simplification Strategy

The repo should be simplified from the outside inward:

1. **Path and fixture contracts first.** Establish where studies and validation
   fixtures live before moving many files.
2. **Folder moves second.** Move `results/case*` only after tests and scripts
   have a stable way to find fixtures.
3. **Code movement third.** Move or split `results_runtime.py` only after the
   study layout exists.
4. **Numerical refactors last.** Do not restructure the core operator/mesh
   modules until MATLAB-to-NPZ fixtures protect behavior.

This is more conservative than moving directories immediately, but it avoids
breaking tests by path churn.

## Phase ordering note

Phase 5 (install_tester dedup) and Phase 7 step 4 (absolute-path purge) have
no dependency on Phases 2-4 and can run in parallel once Phase 1 is green.

## Phase 0 - Freeze And Classify

Purpose: avoid mixing simplification with unrelated side-session work.

Actions:

1. Re-check `git status --short --branch`.
2. Classify current dirty changes:
   - accepted `README.md` consolidation;
   - deleted Markdown/docs files that should remain deleted;
   - deleted `paper/` files (permanently removed from this repo);
   - generated outputs to ignore/untrack;
   - active scientific campaign files to preserve;
   - unrelated side-session files.
3. Keep only this `plan.md` as the active plan.
4. All work in this simplification lands on a `cleanup/2026-04-layout` branch.
   One commit per phase-gate. Squash-merge to `main` once the full sequence is
   green.

Gate:

- No structural moves until the dirty tree is classified.

## Phase 1 - Metadata, README, And Guardrails

Purpose: make the current checkout coherent before deeper changes.

Actions:

1. Migrate the default branch from `master` to `main`. This is the first commit
   of this phase; all subsequent phase work targets `main`.
2. Fix the stale gitlink/submodule state:
   - remove `docs/resolvent4py` from the index if `docs/` is retired;
   - keep only valid submodules in `.gitmodules`.
3. Update `README.md` as the only broad Markdown documentation:
   - current top-level layout;
   - install/run/test commands;
   - optional data policy;
   - GitHub 50 MB file-size policy;
   - `LinStabPy` / `linstabpy` / `linstab2d` naming note.
4. Add `scripts/check_tracked_file_size.py` wired into a `pre-commit` hook and
   CI. The script fails if any tracked file exceeds 50 MB, and reports ignored
   local outputs over 50 MB separately.

Validation:

- `git submodule status` must not fail.
- README links must point to existing files.
- No package behavior changes.
- `scripts/check_tracked_file_size.py` exits non-zero when a tracked file
  exceeds 50 MB.

Rollback: revert the phase commit; state returns to the prior gate.

## Phase 2 - Path And Fixture Contracts

Purpose: make future moves mechanical instead of fragile.

Actions:

1. Define canonical path roots in docs/tests before moving files:
   - `studies/case1`
   - `studies/case2`
   - `validation/operator_fixtures`
   - `validation/martini2024`
2. Add `tests/_paths.py` exposing `STUDIES_ROOT`, `VALIDATION_ROOT`, and
   `case_fixture(name)` so tests do not each invent their own
   `ROOT / "results/case1"` path.
3. Add `scripts/check_tracked_file_size.py` (wired in Phase 1; ensure it is
   importable from tests if needed).
4. Add `validation/operator_fixtures/SCHEMA.md` (one page): `.npz` vs sparse
   `.npz` conventions, required `.jsonc` keys, ordering convention, tolerance.
5. Decide the MATLAB-to-NPZ fixture schema:
   - dense `.npz` for small arrays;
   - SciPy sparse `.npz` for sparse matrices;
   - `.jsonc` metadata with source script/commit, physical parameters, shape,
     dtype, ordering convention, tolerance, and generation date.
6. Identify the minimum canonical fixtures:
   - derivative matrices;
   - quadrature/Chu-norm weights;
   - curvilinear metric/Jacobian arrays;
   - pre-BC and post-BC operators;
   - boundary-condition rows or masks;
   - selected sparse blocks;
   - selected eigen/resolvent reference outputs.
7. Choose fixture sizes that stay below 10 MB per file and still catch ordering,
   sign, scaling, metric, and boundary-condition regressions.

Validation:

- Tests can refer to future fixture locations without editing each test again.
- Missing optional fixtures skip with precise messages.

Rollback: revert the phase commit; state returns to the prior gate.

## Phase 3 - Preserve MATLAB Validation, Then Reduce Raw Data

Purpose: keep validation value while shrinking and clarifying data ownership.

Actions:

1. Convert the selected MATLAB operator references to `.npz` / SciPy sparse
   `.npz`.
2. Put these under `validation/operator_fixtures/` with metadata.
3. Update parity tests to load these fixtures instead of raw `.mat` files where
   practical.
4. For raw `.mat` files over 50 MB: move to external storage or regeneration
   instructions. A tracked fixture must be <= 10 MB and sufficient to catch the
   regression it names. Anything larger than 10 MB becomes a regeneration script
   under `scripts/regenerate_<name>.py`, not a tracked artifact.
5. Move Martini example parity bundles from the missing `tools/parity/npz`
   convention to `validation/martini2024/fixtures/`.
6. Keep `tools/parity/` as conversion/comparison code only, not as a data home.

Validation:

- Operator parity runs from Python without MATLAB.
- `pytest -m "not slow"` does not require large optional data.
- The tracked-file size check passes.

Rollback: revert the phase commit; state returns to the prior gate.

## Phase 4 - Move Studies Out Of `results/`

Purpose: stop using `results/` as both source and output.

Path migration table (canonical reference for `rg`-driven rewrites):

| Old | New |
|---|---|
| `results/case1/` | `studies/case1/` |
| `results/case2/` | `studies/case2/` |
| `tools/parity/npz/` (missing) | `validation/martini2024/fixtures/` |
| `results/case1/MATLAB_exports/operator_convention.mat` | `validation/operator_fixtures/operator_convention.{npz,jsonc}` |

Actions:

1. Move curated Case 1 material to `studies/case1/`.
2. Move curated Case 2 material to `studies/case2/`.
3. Keep generated campaign outputs under ignored `output/` directories inside
   each study.
4. Update `README.md`, scripts, configs, tests, and comments from
   `results/case*` to `studies/case*` using the table above as the rewrite map.
5. Preserve active Case 2 plan/worklog content. Do not discard scientific
   state during a layout cleanup.
6. Delete superseded configs only when the Case 2 plan/worklog confirms they
   are inactive.

Validation:

- `rg -n "results/case"` should only find migration notes or no hits.
- Study commands run from repo root.
- Case tests still locate fixtures or skip honestly.

Rollback: revert the phase commit; state returns to the prior gate.

## Phase 5 - Remove Source Duplication

Purpose: make `src/linstab2d` the only package source.

Actions:

1. Replace `install_tester/src/linstab2d` and `install_tester/examples` with
   one of:
   - preferred: install tester installs the real package from the repo or local
     wheel/sdist;
   - fallback: `scripts/build_install_tester_bundle.py` creates an export bundle
     outside committed source.
2. Keep only install harness, machine notes, diagnostics, and benchmark scripts
   that are truly install-tester-specific.
3. Update install tester docs to state whether it is online, offline,
   cluster-only, or general installation.

Validation:

- `diff -qr src/linstab2d install_tester/src/linstab2d` is no longer relevant
  because the duplicate tree is gone.
- The install tester imports the installed package, not a stale local copy.

Rollback: revert the phase commit; state returns to the prior gate.

## Phase 6 - Code Boundary Cleanup

Purpose: simplify package boundaries after folders and fixtures are stable.

Actions:

1. Split `src/linstab2d/results_runtime.py`:
   - keep reusable, tested utilities in package code;
   - move case/study orchestration into `studies/common/` or study-local code;
   - keep a temporary compatibility shim only if existing tests need it.
2. Rename `src/linstab2d/baseflow/case1.py` to a physical/domain name such as
   `hamada.py` only after path moves settle. Keep a compatibility import for one
   transition commit if needed.
3. Remove `results/case*` references from library comments and configs.
4. Keep `solvers/petsc_utils.py` as a compatibility shim until imports are
   updated; remove it only in a small cleanup commit.
5. Do not split core numerical modules yet. Candidates for a later pass are
   listed in "Follow-up Work" at the end of this file.

Validation:

- Public imports used by examples and tests still work.
- Case/study orchestration can run without importing research paths into the
  library package.
- No numerical output changes except path/provenance metadata.

Rollback: revert the phase commit; state returns to the prior gate.

## Phase 7 - Scripts And Commands

Purpose: make operations discoverable and avoid one-off path hacks.

Actions:

1. Move support utilities into `scripts/` unless they are intentionally
   top-level entry points.
2. Keep root-level shell entry points only if documented in `README.md`, for
   example:
   - `install.sh`;
   - `diagnose.sh`;
   - `run_all_examples.sh`;
   - maybe `benchmark.sh`.
3. Remove obsolete wrappers after `rg` confirms no references remain.
4. Replace absolute local paths in ad hoc tools with paths from `__file__`.
5. Keep campaign parameters in `.jsonc` files, not CLI flag sprawl.

Validation:

- README command blocks work as written.
- No hardcoded absolute local user paths remain in committed tools.

Rollback: revert the phase commit; state returns to the prior gate.

## Phase 8 - Verification And Commit Strategy

Verification order:

1. `git status --short --branch`
2. `git submodule status`
3. tracked-file size check: no tracked file over 50 MB
4. `.venv/bin/python -m pytest -m "not slow"`
5. targeted operator-parity tests loading curated MATLAB `.npz` fixtures
6. targeted path-rewrite tests for moved study/case fixtures
7. `bash diagnose.sh`
8. `bash run_all_examples.sh` only when example paths or data handling changed

Commit order:

1. Plan only.
2. Metadata, submodule, README, and size-guard coherence.
3. Path/fixture helper contract.
4. MATLAB-to-NPZ operator fixtures.
5. Study layout/path rewrite.
6. Install tester de-duplication.
7. `results_runtime.py` / case-loader boundary cleanup.
8. Data untracking or externalization.
9. Final README polish.

Stage named paths only. Do not use `git add .` or `git add -A`.

## Open Decisions

3. Which MATLAB-generated operator stages are canonical validation fixtures?
4. What is the smallest mesh/case that still catches operator regressions?
6. After curated `.npz` fixtures exist, which raw `.mat`/debug exports remain in
   git, and which move to external storage or regeneration instructions?

## Recommended First Implementation Slice

Start with the lowest-risk slice:

1. classify the dirty tree;
2. fix stale `docs/resolvent4py` gitlink/submodule state after confirming
   `docs/` is retired;
3. update `README.md` links to the consolidated documentation and current
   layout;
4. add `scripts/check_tracked_file_size.py` wired into a `pre-commit` hook and
   CI for the 50 MB GitHub limit;
5. add `tests/_paths.py` with `STUDIES_ROOT`, `VALIDATION_ROOT`, and
   `case_fixture(name)` so later folder moves do not require editing every test
   by hand.

Done when: `git status --short` is clean on the `cleanup/2026-04-layout` branch,
`scripts/check_tracked_file_size.py` passes, `pytest -m "not slow"` passes, and
README relative links resolve (`scripts/check_readme_links.py` or
`markdown-link-check`).

## Follow-up Work (Out Of Scope)

The items below are candidates for a separate pass once operator fixtures exist
and the layout is stable. Do not act on these during the current simplification.

- `operators/assembly.py`: possible split into metrics/backend orchestration vs
  block assembly.
- `mesh/mesh.py`: possible split into geometry deformation vs operator
  construction.
- `operators/coefficients.py`: possible split into Cartesian vs cylindrical
  coefficient paths.
