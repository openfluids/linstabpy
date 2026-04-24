# LinStabPy Repository Cleanup Plan

Date: 2026-04-24
Branch: `main`
Status: active plan before using `GEMINI.md`

## Purpose

Make this repository easier to review, clone, test, and release without
weakening numerical validation.

The cleanup must preserve these invariants:

- `src/linstab2d/` remains the only importable library source.
- `README.md` remains the canonical broad documentation.
- MATLAB-backed validation is preserved through explicit fixtures, scripts, or
  regeneration instructions.
- No tracked GitHub-facing file exceeds 50 MiB.
- Numerical behavior changes require tests or parity checks.
- `GEMINI.md` is draft review input for a later pass. Do not treat it as
  canonical until this plan is complete and current.

## Current State

Verified on 2026-04-24:

- Local and remote branches are consolidated on `main`.
- Old cleanup branches were merged into `main` and deleted locally/remotely.
- Last completed cleanup commit before this plan refresh was
  `d1f5d31 Add layout cleanup guardrails`.
- `GEMINI.md` is untracked and unfinished.
- `git submodule status` succeeds for `src/linstab2d/data/baseflows`.
- `python scripts/check_tracked_file_size.py` passes.
- The only ignored files over 50 MiB currently reported by the guard are local
  generated example outputs:
  - `examples/example_3_streaky_bl.npz`
  - `examples/example_4_cavity.npz`

Completed on `main`:

- Default branch migrated from `master` to `main`.
- Tutorial commit `0e0ff7e` preserved.
- File-size guardrail added:
  - `scripts/check_tracked_file_size.py`
  - `.pre-commit-config.yaml`
  - `.github/workflows/guardrails.yml`
- Two over-50 MiB raw MATLAB exports were removed from Git tracking. The paths
  remain ignored if the files are restored locally:
  - `results/case1/MATLAB_exports/operator_convention.mat`
  - `validation/song_2023/matlab_code/SolverMartini/MATLAB_exports_Ziming_Internal/coefficients_raw.mat`
- Shared test path helper added:
  - `tests/_paths.py`
- Operator fixture schema added:
  - `validation/operator_fixtures/SCHEMA.md`

Still open:

- `results/case1` and `results/case2` are still active tracked study
  directories.
- `install_tester/src/linstab2d` and `install_tester/examples` still duplicate
  source/examples.
- `tools/parity/martini_manifest.json` still contains absolute local paths.
- `tools/parity/npz` is still the documented parity bundle location, but the
  directory is absent.
- `src/linstab2d/results_runtime.py` still contains study orchestration in
  package code.
- `src/linstab2d/baseflow/case1.py` still contains study-specific Hamada/Case 1
  loading inside the library.

## Target Layout

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
      fixtures/
    song_2023/
```

Directory rules:

- `src/linstab2d/`: library code only.
- `examples/`: runnable examples and `.jsonc` configs; generated `.npz` and
  `.png` outputs remain ignored.
- `studies/`: active research/campaign workflows.
- `validation/operator_fixtures/`: compact MATLAB-derived operator fixtures and
  metadata.
- `validation/martini2024/fixtures/`: compact Martini parity bundles.
- `tools/parity/`: conversion and comparison tools, not data storage.
- `install_tester/`: install harness, notes, and scripts only; no duplicate
  package or example tree.

## Non-Goals For This Cleanup

- Do not rename the import package from `linstab2d` to `linstabpy`.
- Do not split core numerical modules such as `operators/assembly.py`,
  `mesh/mesh.py`, or `operators/coefficients.py` in this cleanup.
- Do not change PETSc/SLEPc solver defaults, matrix assembly, boundary
  conditions, viscosity formulas, or acceptance thresholds without a separate
  test-first numerical task.
- Do not treat `GEMINI.md` findings as accepted until they are reviewed against
  the code and this plan.

## Phase 1 - Finish Layout Cleanup

Goal: stop using `results/` as the source namespace for active studies.

Actions:

1. Move curated Case 1 material:
   - from `results/case1/`
   - to `studies/case1/`
2. Move curated Case 2 material:
   - from `results/case2/`
   - to `studies/case2/`
3. Preserve Case 2 scientific state:
   - `PLAN.md`
   - `README.md`
   - `REFERENCE.md`
   - `worklog.md`
   - current `.jsonc` campaign configs
4. Keep generated campaign outputs under ignored `output/` directories inside
   each study.
5. Update references from `results/case*` to `studies/case*` in:
   - `README.md`
   - `src/linstab2d/config/validation_cases.yaml`
   - `src/linstab2d/baseflow/case1.py` comments
   - tests
   - study scripts
6. Use `tests/_paths.py` as the compatibility layer during the move.

Validation:

- `rg -n "results/case"` should show only migration notes or no hits.
- `python scripts/check_tracked_file_size.py`
- `PYTHONPATH=src .venv/bin/python -m pytest -m "not slow"`
- Targeted Case 1/Case 2 tests either pass or skip with precise missing-data
  messages.

## Phase 2 - Remove Install Tester Duplication

Goal: make `src/linstab2d` the only library source tree.

Actions:

1. Remove the duplicate package tree:
   - `install_tester/src/linstab2d`
2. Remove duplicated examples:
   - `install_tester/examples`
3. Update `install_tester/run_install_tester.sh` so it installs/tests the real
   package from the repository checkout or a generated bundle.
4. Keep install-tester-specific files:
   - `install_tester/README.md`
   - `install_tester/install.sh`
   - `install_tester/diagnose.sh`
   - `install_tester/benchmark.sh`
   - `install_tester/installation_learnings.md`
   - `install_tester/lovelace_requirements.md`
5. Update `install_tester/README.md` to explain the new source-of-truth model.

Validation:

- `rg -n "install_tester/src|install_tester/examples"` should show only
  historical notes or no hits.
- `bash -n install_tester/run_install_tester.sh`
- The top-level non-slow pytest suite still passes.

## Phase 3 - Sanitize Martini Parity Metadata

Goal: remove local-machine paths and define the future parity fixture home.

Actions:

1. Replace absolute paths in `tools/parity/martini_manifest.json` with
   repository-relative paths.
2. Update `tools/parity/martini_cases.jsonc` comments from `tools/parity/npz`
   to `validation/martini2024/fixtures`.
3. Update parity tools so the default bundle directory is
   `validation/martini2024/fixtures`, with compatibility for old local
   `tools/parity/npz` if needed.
4. Move existing tracked parity reports only if their path references are
   regenerated or clearly marked historical.

Validation:

- `rg -n "/Users/rfrantz" tools validation README.md src tests` returns no
  tracked source references.
- Existing parity-tool unit tests pass.
- Missing optional parity bundles skip with precise messages.

## Phase 4 - Preserve Operator Validation With Compact Fixtures

Goal: keep MATLAB validation useful without tracking large raw exports.

Actions:

1. Decide the minimum operator fixtures before converting data:
   - derivative matrices
   - quadrature / Chu-norm weights
   - metric and Jacobian arrays
   - pre-BC and post-BC operators
   - selected boundary-condition rows or masks
   - selected eigen/resolvent reference outputs
2. Convert selected raw MATLAB exports into compact `.npz` or SciPy sparse
   `.npz` files under `validation/operator_fixtures/`.
3. Add `.jsonc` metadata for each fixture following
   `validation/operator_fixtures/SCHEMA.md`.
4. Keep each tracked fixture below 10 MiB where possible.
5. For any larger artifact, add a regeneration script under `scripts/` instead
   of tracking the artifact.
6. Update operator parity tests to use compact fixtures where practical.

Validation:

- Operator parity tests run without MATLAB for the selected fixtures.
- `python scripts/check_tracked_file_size.py`
- `PYTHONPATH=src .venv/bin/python -m pytest -m "not slow"`

## Phase 5 - Separate Study Runtime From Library Runtime

Goal: keep package code reusable and move study orchestration out of the
library.

Actions:

1. Split `src/linstab2d/results_runtime.py`:
   - keep reusable utilities in package code;
   - move Case 1/Case 2 orchestration into `studies/common/` or study-local
     code.
2. Keep temporary compatibility wrappers only where tests or study scripts need
   a transition path.
3. Remove package comments and configs that point to old `results/case*`
   locations.
4. Keep `src/linstab2d/solvers/petsc_utils.py` as a compatibility re-export for
   now; remove it only in a separate small commit after import cleanup.

Validation:

- Public imports used by examples and tests still work.
- Study workflows run from repository root.
- No numerical output changes except path/provenance metadata.

## Phase 6 - Final Documentation And Plan Closure

Goal: leave the repository with one current README and no stale cleanup plan.

Actions:

1. Update `README.md` with the final layout and commands.
2. Decide whether to keep `plan.md`:
   - keep it if more cleanup phases remain;
   - delete it if all actions above are complete and the README is current.
3. If deleted, make sure remaining release or review notes live in README,
   issues, or a named review ledger.
4. Commit and push the final clean state on `main`.

Validation:

- `git status --short --branch` is clean except intentionally untracked drafts.
- `python scripts/check_tracked_file_size.py`
- `git submodule status`
- `git diff --check`
- `PYTHONPATH=src .venv/bin/python -m pytest -m "not slow"`

## GEMINI.md Handling

`GEMINI.md` is an unfinished review draft. Work on it only after this plan is
current.

When ready:

1. Treat it as input, not truth.
2. Classify each finding as:
   - accepted now;
   - needs investigation;
   - rejected with evidence;
   - future roadmap.
3. Do easy hygiene findings first:
   - install-tester duplication;
   - absolute manifest paths;
   - documentation/layout issues.
4. Do numerical findings only with focused tests and evidence:
   - Sutherland non-dimensionalization;
   - axisymmetric `1/r` handling;
   - MUMPS default changes;
   - energy-equation row audits.
5. Commit `GEMINI.md` only when it is either:
   - a polished review ledger; or
   - replaced by a cleaner tracked plan/review artifact.

## Immediate Next Step

Start with Phase 1: move `results/case1` and `results/case2` to `studies/`,
then update references and run the non-slow test suite.
