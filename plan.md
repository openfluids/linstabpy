# LinStabPy Folder Reorganization Plan

Date: 2026-04-24
Branch: `main`
Status: implemented on `main`

## Purpose

Reduce visual and operational clutter without weakening numerical validation.
The goal is a repository where a new reader can immediately distinguish:

- reusable library code;
- runnable examples;
- active research campaigns;
- validation/reference data;
- developer and installation tools;
- generated outputs and caches.

This plan replaces the older cleanup plan as the current `plan.md`. Open
non-folder numerical/release items from that plan are preserved in the final
section so they are not lost.

## Implementation Ledger

- Phase 0 preflight: completed.
- Phase 1 study-folder cleanup: completed in `f8b1fd8`.
- Phase 2 validation-folder consolidation: completed in `3b67ba1`.
- Phase 3 legacy-layout fallback removal: completed in `ad341b8`.
- Phase 4 study runtime split: completed in `d362fff`.
- Phase 5 documentation refresh: completed in the final documentation commit.
- Corrective pass: completed. The Phase 4 runtime split crossed the package
  boundary: `linstab2d.results_runtime` is installed from `src/`, but the
  implementation had been moved to top-level `studies/`, which is repo-local.
  Fixed by moving runtime implementation back into `src/linstab2d/`, making
  studies wrappers depend on the package, deleting the duplicate
  `studies/common/` module, and repairing direct-entry/config defaults.

Verification notes:

- Path-sensitive compile, shell-syntax, direct import/config, tracked-size, and
  diff-whitespace checks were run during the implementation.
- Pytest collection is blocked in this local checkout because `petsc4py` is not
  installed; collection stops in `tests/conftest.py` before reaching the
  changed tests.

Corrective pass validation:

- `python -m compileall -q src studies validation tests`
- `bash -n run_all_examples.sh validation/song_2023/song_2023.sh studies/case1/case1.sh studies/case2/run_case_once.sh studies/case2/run_perf_queue_once.sh`
- `git diff --check`
- `python scripts/check_tracked_file_size.py`
- `make -C tutorial`
- `PYTHONPATH=src:. python -c "from studies.case2 import perf_memory; ..."`:
  default Case 2 campaign config exists and loads one case.
- `PYTHONPATH=src python -c "import linstab2d.results_runtime"` and direct
  Case 1/Case 2 entrypoint checks now reach `petsc4py`; this local interpreter
  still lacks `petsc4py`, so full PETSc-backed entrypoint execution remains
  blocked here by environment rather than by repository layout.

## Current Evidence

The tracked top-level structure is already mostly reasonable:

```text
src/linstab2d/        library package
tests/                pytest suite
examples/             Martini examples
studies/              active Case 1 / Case 2 workflows
validation/           reproducibility and reference-data workflows
tools/                developer utilities
scripts/              guardrails
install_tester/       installer harness
tutorial/             tutorial source and PDF
```

The main clutter is not duplicated top-level folders anymore. It is inside:

- `studies/case1/`: baseflow arrays, MATLAB exports, scripts, configs, and
  outputs all live near the top of the case directory.
- `studies/case2/`: many EVP/resolvent configs, active scripts, rankfiles,
  phase artifacts, and documentation share one directory.
- `validation/song_2023/`: Python workflow code, JSONC configs, baseflow data,
  MATLAB source, MATLAB reference outputs, and run scripts are mixed.
- `tools/parity/` and `validation/martini2024/`: Martini validation logic and
  Martini fixtures are split across two roots.
- `tests/_paths.py` still has legacy fallback paths for old `results/case*`
  and root-level `case*` layouts.

## Invariants

These must remain true throughout the reorganization:

- `src/linstab2d/` remains the only importable library source tree.
- Root commands stay easy to find:
  - `install.sh`
  - `diagnose.sh`
  - `benchmark.sh`
  - `run_all_examples.sh`
- `run_all_examples.sh` remains the canonical example/parity/MPI runner.
- `tutorial/` stays available unless a later explicit decision moves docs under
  `docs/`.
- Generated `.npz`, `.png`, logs, caches, and campaign `output/` directories
  stay ignored unless explicitly curated as paper artifacts.
- Numerical behavior does not change during path-only moves.
- Every move is followed by reference rewrites and targeted verification.
- No broad staging: stage only named files.

## Non-Goals

- Do not rename the import package from `linstab2d` to `linstabpy`.
- Do not change PETSc/SLEPc defaults, boundary conditions, matrix assembly,
  viscosity formulas, or acceptance thresholds.
- Do not move root onboarding scripts into a hidden tools directory.
- Do not split core numerical modules as part of this folder cleanup.
- Do not delete validation data unless a regeneration path and consumer audit
  are documented first.

## Target Layout

Preferred final shape:

```text
.
|-- src/linstab2d/
|-- tests/
|-- examples/
|-- studies/
|   |-- case1/
|   |   |-- configs/
|   |   |-- data/baseflow/
|   |   |-- reference/matlab/
|   |   |-- scripts/
|   |   `-- artifacts/
|   `-- case2/
|       |-- configs/
|       |   |-- evp/
|       |   |-- resolvent/
|       |   `-- perf/
|       |-- data/baseflow/
|       |-- scripts/
|       |-- artifacts/phase1p5/
|       `-- docs/
|-- validation/
|   |-- martini2024/
|   |   |-- fixtures/
|   |   |-- reports/
|   |   `-- tools/
|   |-- operator_fixtures/
|   `-- song_2023/
|       |-- configs/
|       |-- data/baseflow/
|       |-- reference/matlab/
|       `-- scripts/
|-- tools/
|   `-- debug/
|-- scripts/
|-- install_tester/
|-- tutorial/
|-- install.sh
|-- diagnose.sh
|-- benchmark.sh
`-- run_all_examples.sh
```

Decision notes:

- Keep `install_tester/` and `tutorial/` at the top level for now. They are
  user-facing entrypoints, not accidental duplication.
- Move `tools/parity/` only after the case-study cleanup, because
  `run_all_examples.sh`, tests, and README currently point to it.
- Keep case entrypoints at the case root during the first move:
  - `studies/case1/case1.py`
  - `studies/case1/case1.sh`
  - `studies/case2/case2.py`
  - `studies/case2/perf_memory.py`

## Phase 0 - Preflight

Goal: make sure moves are mechanical and reversible.

Actions:

1. Confirm clean state:
   - `git status --short --branch`
2. Record current tracked path counts:
   - `git ls-files | awk -F/ '{count[$1]++} END {for (n in count) print count[n], n}'`
3. Find path consumers before each move:
   - `rg -n "studies/case1|studies/case2|validation/song_2023|tools/parity"`
4. Create one small commit per phase.

Validation:

- `git diff --check`
- no unexpected untracked source files

## Phase 1 - Reorganize Study Internals

Goal: make `studies/` readable without changing public study entrypoints.

### Case 1

Proposed moves:

```text
studies/case1/{T,rho,u,v,w,x,y,sponge}.npy
  -> studies/case1/data/baseflow/

studies/case1/MATLAB_exports/
studies/case1/MATLAB_exports_resolvent/
studies/case1/MATLAB_results/
  -> studies/case1/reference/matlab/

studies/case1/perf_memory_cases*.jsonc
studies/case1/production_case_template.jsonc
studies/case1/resolvent_smoke_fd4_socket1.jsonc
studies/case1/parity_config.jsonc
  -> studies/case1/configs/

studies/case1/plot_*.py
studies/case1/check_*.py
studies/case1/build_reference_inventory.py
studies/case1/generate_figures.py
studies/case1/perf_memory.py
  -> studies/case1/scripts/

studies/case1/workflow/
  -> studies/case1/reference/matlab/workflow/
```

Keep at case root initially:

- `case1.py`
- `case1.sh`
- `case1.md`
- `test_case1.py`
- `parity_reference_inventory.json` until consumers are checked

### Case 2

Proposed moves:

```text
studies/case2/baseflow/
  -> studies/case2/data/baseflow/

studies/case2/evp_*.jsonc
  -> studies/case2/configs/evp/

studies/case2/resolvent_*.jsonc
  -> studies/case2/configs/resolvent/

studies/case2/_defaults.jsonc
studies/case2/parity_config.jsonc
  -> studies/case2/configs/

studies/case2/phase1p5_*.py
studies/case2/plot_saved_*.py
  -> studies/case2/scripts/

studies/case2/phase1p5/
  -> studies/case2/artifacts/phase1p5/

studies/case2/{PLAN,README,REFERENCE,QFILT_COMPARISON_PROTOCOL}.md
studies/case2/tutorial.md
studies/case2/worklog.md
  -> studies/case2/docs/
```

Keep at case root initially:

- `case2.py`
- `perf_memory.py`
- `run_case_once.sh`
- `run_perf_queue_once.sh`
- `socket*.rankfile`

Validation:

- `rg -n "studies/case1|studies/case2" README.md tutorial tests src scripts`
- `bash -n studies/case1/case1.sh studies/case2/run_case_once.sh studies/case2/run_perf_queue_once.sh`
- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_case2_perf_memory_include.py`
- targeted Case 1 tests either pass or skip with precise missing-data messages

## Phase 2 - Consolidate Validation Layouts

Goal: make validation workflows self-contained by case/source.

### Martini 2024

Proposed moves:

```text
tools/parity/*.py
tools/parity/*.json
tools/parity/*.jsonc
  -> validation/martini2024/tools/

tools/parity/reports/
  -> validation/martini2024/reports/
```

Update:

- `run_all_examples.sh` parity script path
- README parity commands
- `tests/test_martini_parity_compare.py`
- any JSON report path references

### Song 2023

Proposed moves:

```text
validation/song_2023/baseflow/
  -> validation/song_2023/data/baseflow/

validation/song_2023/*.jsonc
  -> validation/song_2023/configs/

validation/song_2023/plot_*.py
validation/song_2023/run_*.py
validation/song_2023/compare_*.py
validation/song_2023/scan_preprocess.py
validation/song_2023/populate_scaling_table.py
  -> validation/song_2023/scripts/

validation/song_2023/matlab_code/
validation/song_2023/matlab_reference/
validation/song_2023/ref_results/
  -> validation/song_2023/reference/matlab/
```

Keep at validation root initially:

- `validation/song_2023/song_2023.py`
- `validation/song_2023/song_2023.sh`
- `validation/song_2023/baseflow_loader.py`
- `validation/song_2023/bc_setup.py`
- `validation/song_2023/plotter.py`

Validation:

- `bash -n validation/song_2023/song_2023.sh`
- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_martini_parity_compare.py`
- `bash run_all_examples.sh --list`
- parity reports still write to the documented location

## Phase 3 - Remove Legacy Compatibility Paths

Goal: stop carrying old layout assumptions after the move is stable.

Actions:

1. Remove fallback paths from `tests/_paths.py`:
   - `results/case*`
   - root `case*`
2. Remove stale README references to migration/legacy locations.
3. Search for old locations:
   - `rg -n "results/case|tools/parity|validation/song_2023/baseflow|MATLAB_exports"`
4. Keep intentional historical notes only if they say they are historical.

Validation:

- `PYTHONPATH=src .venv/bin/python -m pytest -m "not slow"`
- `python scripts/check_tracked_file_size.py`
- `git diff --check`

## Phase 4 - Keep Runtime On The Package Boundary

Goal: keep importable runtime code inside the installed package while making
case-study entrypoints thin and local.

Actions:

1. Keep `src/linstab2d/results_runtime.py` as the single runtime
   implementation used by Case 1 and Case 2.
2. Keep `studies/case1/case1.py` and `studies/case2/case2.py` as thin wrappers
   that set case-specific defaults before calling `linstab2d.results_runtime`.
3. Do not keep a duplicate `studies/common/` runtime package; `studies/` is
   repo-local workflow/data/docs, not installable library code.
4. Review `src/linstab2d/baseflow/case1.py`:
   - keep reusable loader primitives if general;
   - move study-specific Hamada/Case 1 assumptions under `studies/case1/` if
     they are not library-level behavior.

Validation:

- package imports used by tests and examples still resolve from `src/linstab2d`
- Case 1 and Case 2 workflows run from repository root
- no numerical output changes except path/provenance metadata

## Phase 5 - Documentation And Tutorial Refresh

Goal: make the new structure obvious to humans.

Actions:

1. Update the README folder diagram.
2. Update the tutorial "Where files belong" section.
3. Update Case 1 and Case 2 local docs after their paths move.
4. Update `install_tester/README.md` only if root paths changed.

Validation:

- `make -C tutorial`
- `pdftotext tutorial/tutorial.pdf - | rg "Where files belong|run_all_examples"`
- `git diff --check`

## Commit Strategy

Use small commits:

1. `Plan folder reorganization`
2. `Reorganize study case folders`
3. `Consolidate validation folders`
4. `Remove legacy layout fallbacks`
5. `Separate study runtime helpers`
6. `Refresh layout documentation`

Do not use broad staging. Stage explicit paths in each commit.

## Acceptance Criteria

The reorganization is done when:

- top-level layout matches README and tutorial;
- Case 1 and Case 2 root directories no longer contain a flat mix of configs,
  scripts, baseflow arrays, MATLAB references, and paper artifacts;
- Martini parity has one validation namespace;
- Song 2023 has clear `configs/`, `data/`, `reference/`, and `scripts/`
  subdirectories;
- legacy fallback paths are removed from tests;
- `git status --short --branch` is clean;
- `python scripts/check_tracked_file_size.py` passes;
- targeted path-sensitive tests pass;
- generated outputs remain ignored.

## Preserved Non-Folder Backlog

These items came from the previous plan and should remain separate from the
folder cleanup:

- compact operator fixtures under `validation/operator_fixtures/`;
- live PETSc/MUMPS factorization verification for current defaults;
- solver doctor behavior;
- energy-equation row audit;
- axis singularity regularization;
- release tuple / immutable public-code reference:
  - commit;
  - release tag;
  - Zenodo DOI or explicit blocker.

Do not mix those numerical/release tasks into a path-only reorganization commit.
