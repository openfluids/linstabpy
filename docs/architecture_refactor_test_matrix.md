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
.venv/bin/python scripts/validation/architecture_refactor_matrix.py --group phase0-local --skip-long
```

The runner writes a JSON report with the current git SHA, selected checks,
commands, return codes, and artifact paths. A normal group run includes every
check in that group. Use `--skip-long` only when you are doing a quick local
smoke run that is not a Phase 0 signoff.

## Required Phases

| Phase | Blocking evidence | Required before |
|---|---|---|
| Phase 0 characterization | Full local pytest, focused operator/BC/solver tests, example smoke, parity matrix status, operator perf/memory baseline, BC behavior tests, selected generated artifacts with SHA | Any BCRecipe, LNSESystem, SolverConfig, runtime, Mesh, or kernel implementation |
| Safety infrastructure | Test matrix runner, numerical equivalence harness, dual-path migration plan, docstring gate, invariant-comment policy | Any destructive migration from old path to new seam |
| BCRecipe | BC index parity across SciPy, PETSc, and spatial paths; compatibility wrappers; no public JSONC breakage | LNSE assembly seam migration |
| LNSESystem and kernel work | Full operator numerical diffs, coefficient baseline diffs, PETSc/SciPy backend parity, perf/memory sidecars | Solver policy centralization |
| SolverConfig | Solver init/config characterization, PETSc/SLEPc construction parity, MPI checks | Runtime pipeline split |
| Runtime and Mesh split | NPZ output contract diff, Case 1/2 smoke, example family smoke, MPI multi-rank smoke | Final c8 gate |
| Final c8 gate | Fresh pull on c8, full tests, canonical examples, Martini parity, Song validation, Case 1/2 checks, MPI runs, representative perf baselines | Goal completion |

## Phase 0 Local Matrix

These checks are intended to run on the normal development checkout before any
production refactor.

| ID | Command | Artifact |
|---|---|---|
| `pytest-full` | `.venv/bin/python -m pytest tests/ -q` | pytest console log |
| `pytest-focused` | `.venv/bin/python -m pytest tests/test_boundary_conditions.py tests/test_operators.py tests/test_solvers.py tests/test_stability_acceptance.py -q` | pytest console log |
| `example-blasius-smoke` | `bash run_all_examples.sh --example 1s --skip-parity --fresh` | `examples/example_1_blasius_smoke.npz` |
| `parity-matrix-status` | `.venv/bin/python scripts/validation/martini2024/tools/check_parity_matrices.py --allow-incomplete --artifacts-dir artifacts/architecture_refactor/architecture_refactor_phase0` | `artifacts/architecture_refactor/architecture_refactor_phase0/parity_gate_summary.json` |
| `operator-perf-memory` | `.venv/bin/python scripts/validation/martini2024/tools/perf_memory_operator_parity.py --meshes 24x16,32x24 --output artifacts/architecture_refactor/architecture_refactor_phase0/operator_perf_memory.json` | `artifacts/architecture_refactor/architecture_refactor_phase0/operator_perf_memory.json` |

## Expanded Baseline Matrix

These checks are required before deleting old paths or accepting kernel
rewrites. Some are workstation or c8 jobs because they are expensive.

| ID | Scope | Evidence |
|---|---|---|
| `operator-full-martini` | Martini operator-level parity for all implemented examples | `compare_martini_examples.py` report with SHA |
| `curvilinear-baseline` | Example 0 curvilinear derivative and normal handling | regenerated NPZ plus parity fixture report |
| `case1-operator-spectrum-resolvent` | Song/Robinet Case 1 operator, spectrum, and resolvent parity | Case 1 parity JSON files |
| `case2-key-config` | Case 2 representative smoke and one medium stride config | perf/memory JSON and NPZ output |
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
