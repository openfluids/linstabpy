# Architecture Refactor Test Matrix

This matrix defines the checks that guard the BCRecipe, LNSESystem,
SolverConfig, runtime-pipeline, Mesh-state, and kernel-performance refactors.
It is a release-engineering artifact: every phase must record the git SHA,
commands, artifacts, and review note in the relevant bead before the next phase
starts.

The matrix has two rules.

1. Phase 0 baselines are immutable references. If a later refactor changes an
   output, the change must be explained by a deliberate numerical fix, not by a
   tolerance change.
2. Documentation is part of the safety case. New public seams and high-risk
   private numerical helpers must have docstrings or invariant comments before
   the phase that introduces them can close.

## Runner

Use the matrix runner from the repository root:

```bash
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --list
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group phase0-local
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group phase0-expanded-local
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group phase0-external-assets
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group documentation-gates
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group phase0-local --skip-long
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group phase0-golden-local
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group phase0-golden-all
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group safety-infrastructure
```

The runner writes a JSON report with the current git SHA, selected checks,
commands, timeout budgets, return codes, artifact paths, and artifact
inventories. File artifacts include size and SHA-256. Directory artifacts
include file count, total size, and a stable SHA-256 computed from each relative
file path, file size, and file digest. If an artifact cannot be read after the
check finishes, the runner records an inventory error instead of discarding the
whole report. Artifact inventory hashing has its own 60 second cap so report
post-processing cannot make a timed check run indefinitely. A normal group run
includes every check in that group. Use `--skip-long` only when you are doing a
quick local run that is not a Phase 0 signoff.

The `phase0-golden-local` virtual group runs `phase0-local`,
`phase0-expanded-local`, and `documentation-gates` as one local baseline suite.
The `phase0-golden-all` virtual group also includes
`phase0-external-assets`; it is the intended Phase 0 signoff suite once the
optional Martini assets and fixture bundles are present.
The `safety-infrastructure` virtual group runs the equivalence gates and the
documentation gate that protect later seam and kernel work.
The compatibility and shim policy for those seam migrations lives in
`docs/migration_strategy.md`.

Non-long checks have a 15 minute timeout. Checks marked `long` have a 2 hour
timeout. A timeout is reported as its own check status in the JSON report and
must be triaged like a failed validation gate.

## Required Phases

| Phase | Blocking evidence | Required before |
|---|---|---|
| Phase 0 characterization | Full local pytest, focused operator/BC/solver tests, example runs, parity matrix status, operator perf/memory baseline, BC behavior tests, selected generated artifacts with SHA | Any BCRecipe, LNSESystem, SolverConfig, runtime, Mesh, or kernel implementation |
| Safety infrastructure | Test matrix runner, numerical equivalence harness, dual-path migration plan, docstring gate, invariant-comment policy | Any destructive migration from old path to new seam |
| BCRecipe | BC index parity across SciPy, PETSc, and spatial paths; compatibility wrappers; no public JSONC breakage | LNSE assembly seam migration |
| LNSESystem and kernel work | Full operator numerical diffs, coefficient baseline diffs, PETSc/SciPy backend parity, perf/memory sidecars | Solver policy centralization |
| SolverConfig | Solver init/config characterization, PETSc/SLEPc construction parity, MPI checks | Runtime pipeline split |
| Runtime and Mesh split | NPZ output contract diff, Case 1/2 runs, example family runs, MPI multi-rank runs | Final c8 gate |
| Final c8 gate | Fresh pull on c8, full tests, canonical examples, Martini parity, Song validation, Case 1/2 checks, MPI runs, representative perf baselines | Goal completion |

## Phase 0 Local Matrix

These checks are intended to run on the normal development checkout before any
production refactor.

| ID | Command | Artifact |
|---|---|---|
| `pytest-full` | `.venv/bin/python -m pytest tests/ -q` | pytest console log |
| `pytest-focused` | `.venv/bin/python -m pytest tests/test_boundary_conditions.py tests/test_operators.py tests/test_solvers.py tests/test_stability_acceptance.py -q` | pytest console log |
| (removed 2026-08-12; reduced example deleted) | – | – |
| `parity-matrix-status` | `.venv/bin/python scripts/validation/martini2024/tools/check_parity_matrices.py --allow-incomplete --artifacts-dir artifacts/architecture_refactor/architecture_refactor_phase0` | `artifacts/architecture_refactor/architecture_refactor_phase0/parity_gate_summary.json` |
| `operator-perf-memory` | `.venv/bin/python scripts/validation/martini2024/tools/perf_memory_operator_parity.py --meshes 24x16,32x24 --output artifacts/architecture_refactor/architecture_refactor_phase0/operator_perf_memory.json` | `artifacts/architecture_refactor/architecture_refactor_phase0/operator_perf_memory.json` |

## Phase 0 Expanded Local Matrix

These checks produce the first local "golden" artifacts used by later dual-path
and kernel-equivalence work. They are still local-workstation checks; c8 jobs
remain a later gate.

| ID | Command | Artifact |
|---|---|---|
| `phase0-characterization` | `.venv/bin/python scripts/validation/phase0_characterization.py --output artifacts/architecture_refactor/architecture_refactor_phase0/phase0_characterization.json` | BC index sets, coefficient tensor hashes, and coefficient/assembly timing sidecars |
| `operator-construction-parity` | `.venv/bin/python scripts/validation/martini2024/tools/compare_operator_construction.py --strict --output artifacts/architecture_refactor/architecture_refactor_phase0/operator_numeric_parity.json` | operator rebuild and PETSc matvec parity report |
| `assembly-pair-2d` | `.venv/bin/python scripts/validation/martini2024/tools/assembly_parity.py pair --model 2D --backend-before scipy --backend-after petsc --skip-eigs --artifact-before artifacts/architecture_refactor/architecture_refactor_phase0/assembly_pair_2d_scipy.npz --artifact-after artifacts/architecture_refactor/architecture_refactor_phase0/assembly_pair_2d_petsc.npz --output artifacts/architecture_refactor/architecture_refactor_phase0/assembly_pair_2d.json` | deterministic 2-D SciPy/PETSc sparse parity |
| `assembly-pair-axy` | `.venv/bin/python scripts/validation/martini2024/tools/assembly_parity.py pair --model axy --backend-before scipy --backend-after petsc --skip-eigs --artifact-before artifacts/architecture_refactor/architecture_refactor_phase0/assembly_pair_axy_scipy.npz --artifact-after artifacts/architecture_refactor/architecture_refactor_phase0/assembly_pair_axy_petsc.npz --output artifacts/architecture_refactor/architecture_refactor_phase0/assembly_pair_axy.json` | deterministic axisymmetric SciPy/PETSc sparse parity |
| `curvilinear-example0` | `bash run_all_examples.sh --example 0 --skip-parity --fresh` | `examples/example_0_curvilinear_validation.npz` |
| `high-mode-local-baseline` | `.venv/bin/python scripts/validation/phase0_characterization.py --output artifacts/architecture_refactor/architecture_refactor_phase0/phase0_characterization.json --high-mode-output artifacts/architecture_refactor/architecture_refactor_phase0/phase0_high_mode_baseline.json` | 100-mode stability baseline for one interior target and one boundary-influenced target |
| `coefficient-tensor-snapshot` | `.venv/bin/python scripts/validation/phase0_characterization.py --output artifacts/architecture_refactor/architecture_refactor_phase0/phase0_characterization.json --coefficient-snapshot-output artifacts/architecture_refactor/architecture_refactor_phase0/phase0_coefficient_tensors.npz` | exact selected coefficient tensors for P1 before/after equivalence checks |
| `case1-local-assembly-baseline` | `LINSTABPY_CASE1_PERF_CONFIG=examples/case_studies/case1_song/configs/perf/phase0_local_assembly_baseline.jsonc LINSTABPY_STAGE_HEARTBEAT_SECONDS=60 .venv/bin/python examples/case_studies/case1_song/scripts/perf_memory.py` | bounded Case 1 assembly-only timing/RSS report for P8 |
| `case2-local-assembly-baseline` | `LINSTABPY_CASE2_PERF_CONFIG=examples/case_studies/case2_lui/configs/perf/phase0_local_assembly_baseline.jsonc LINSTABPY_STAGE_HEARTBEAT_SECONDS=60 .venv/bin/python examples/case_studies/case2_lui/perf_memory.py` | bounded Case 2 assembly-only timing/RSS report for P1/P2/P8 |

## Phase 0 External-Asset Matrix

These checks require optional upstream Martini/LinStab2D assets that are not
committed to the package repository. Provide them through
`LINSTABPY_BASEFLOW_DATA_DIR` or `src/linstabpy/data/baseflows`, and restore or
extract the MATLAB fixture NPZ bundles under
`scripts/validation/martini2024/fixtures`. The fixture runbook in
`scripts/validation/martini2024/fixtures/README.md` lists the required raw
exports and the commit policy for compact bundles.

| ID | Command | Artifact |
|---|---|---|
| `martini-fixture-preflight` | `.venv/bin/python scripts/validation/martini2024/tools/extract_martini_to_npz.py --check-fixtures` | explicit inventory of missing raw MATLAB exports and extracted NPZ fixtures |
| `martini-full-examples` | `bash run_all_examples.sh --fresh` | regenerated `examples/*.npz` files plus `scripts/validation/martini2024/reports/martini_parity_latest.json` |

## Expanded Baseline Matrix

These checks are required before deleting old paths or accepting kernel
rewrites. Some are workstation or c8 jobs because they are expensive.

| ID | Scope | Evidence |
|---|---|---|
| `operator-full-martini` | Martini operator-level parity for all implemented examples | `compare_martini_examples.py` report with SHA |
| `curvilinear-baseline` | Example 0 curvilinear derivative and normal handling | regenerated NPZ plus parity fixture report |
| `case1-operator-spectrum-resolvent` | Song/Robinet Case 1 operator, spectrum, and resolvent parity | Case 1 parity JSON files |
| `case2-key-config` | Case 2 representative run and one medium stride config | perf/memory JSON and NPZ output |
| `high-mode-baseline` | Stability/resolvent runs with high mode count on representative targets | spectra, gains, residuals, and postcheck arrays |
| `bc-index-baseline` | Effective and column-zeroed BC indices, overlap, corners, curvilinear normals across backends | machine-readable JSON from characterization tests |
| `coefficient-tensor-baseline` | Selected coefficient tensors before P1 fusion | JSON/NPZ hashes and max-norm summaries |
| `npz-contract-baseline` | Full output contract for at least one Case 2 config | before/after key inventory and value diffs |

## Equivalence Harness Requirements

The numerical equivalence harness must compare the old and new paths on the
same inputs and fail on unapproved drift:

- assembled operators before and after boundary conditions;
- nonzero count, sparse pattern, and value differences;
- BC effective and column-zeroed index sets;
- coefficient tensor hashes or norms for P1;
- final eigenvalues, gains, residuals, accepted/rejected mode indices, and NPZ
  provenance fields;
- wall-time and peak-memory sidecars for coefficient preparation and raw
  assembly.

The default tolerance is the existing parity tolerance for that artifact. Any
new tolerance must be documented in the bead review note with the failing
baseline, observed candidate, and reason.

The first harness slice is:

| ID | Command | Artifact |
|---|---|---|
| `lnse-equivalence-harness` | `.venv/bin/python scripts/validation/lnse_equivalence_harness.py --case 2D --case axy --output artifacts/architecture_refactor/safety_infrastructure/lnse_equivalence_harness.json` | coefficient tensor, raw-operator, pre-BC operator, post-BC operator, BC-index, and eigen-sanity comparison report |
| `lnse-dual-path-env` | `LINSTABPY_DUAL_PATH=1 LINSTABPY_DUAL_PATH_REPORT=artifacts/architecture_refactor/safety_infrastructure/lnse_dual_path_env.json .venv/bin/python scripts/validation/lnse_equivalence_harness.py --case 2D --case axy` | same comparison report, with `dual_path_enabled=true` |

Before the P1/P2/P3 candidate paths exist, this command runs the current path as
both baseline and candidate. That establishes the report schema and the exact
zero-drift contract future paths must satisfy.

## Documentation Gate

The docstring/comment gate is required for the safety infrastructure phase and
then for every later implementation phase. It must cover:

- all new public modules, classes, and functions under `src/linstabpy`;
- public compatibility wrappers for old dict/list-based APIs;
- private helpers that encode numerical ordering or invariants, including BC
  ordering, coefficient fusion, one-pass assembly, PETSc/SLEPc init policy, and
  NPZ schema handling.

Comments should explain maintained invariants and numerical assumptions. Avoid
line-by-line narration or generated-looking filler.

The matrix runner passes `--require-private` for the current high-risk helpers
that encode construction ordering, BC application, stability reduction, PETSc
argv handling, and SLEPc shift configuration. New BCRecipe, LNSESystem,
SolverConfig, runtime, or kernel-fusion helpers must be added to that required
set when they become the owner of an invariant.

| ID | Command | Artifact |
|---|---|---|
| `docstring-inventory` | `.venv/bin/python scripts/validation/check_docstrings.py --root src/linstabpy --require-private <selected invariant helpers> --output artifacts/architecture_refactor/documentation/docstring_inventory.json` | AST inventory of public modules/classes/functions and selected private invariant helpers |

## Bead Review Note Template

Each major bead boundary should include a review note in this form:

```text
SHA: <git sha>
Scope: <bead ids and touched files>
Matrix slice: <check ids>
Commands:
- <command> -> <pass/fail, artifact>
Numerical result: <identical / accepted drift with reason>
Documentation gate: <pass/fail and covered seams/helpers>
Remaining risk: <short note>
```
