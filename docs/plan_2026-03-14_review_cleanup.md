# Review Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the 5 findings from the MPI Option B code review: extract duplicate test helpers, use PETSc-native int type for row indices, add TODO comments to relaxed tolerances, add `destroy()` to `LinearSolver`, and improve `_unexpected_fallback` sentinel messages.

**Architecture:** All changes are localized refactors — no behavioral changes, no new features. The test suite must pass identically before and after.

**Tech Stack:** Python, pytest, petsc4py.

---

## File Structure

| File | Action | What changes |
|------|--------|-------------|
| `tests/mpi_helpers.py` | Create | Shared MPI subprocess helpers extracted from test files |
| `tests/test_operators.py` | Modify | Remove duplicate helpers, import from `mpi_helpers` |
| `tests/test_solvers.py` | Modify | Remove duplicate helpers, import from `mpi_helpers` |
| `tests/test_utils.py` | Modify | Import `_run_mpi_python` from `mpi_helpers` (only uses that one) |
| `src/linstab2d/solvers/linsolve.py` | Modify | Add `destroy()` method to `LinearSolver`; use `PETSc.IntType` for row arange |
| `src/linstab2d/operators/assembly.py` | Modify | Use `PETSc.IntType` for row arange in `_apply_inv_B_petsc` |
| `tests/test_integration.py` | Modify | Add TODO comments to relaxed tolerances |

---

## Chunk 1: Extract duplicate MPI test helpers

### Task 1: Create shared `tests/mpi_helpers.py`

**Files:**
- Create: `tests/mpi_helpers.py`

The three duplicated functions are:
- `_run_mpi_python(n_ranks, script)` — identical in `test_operators.py:99` and `test_solvers.py:49`
- `_parse_mpi_reports(stdout)` — identical in `test_operators.py:108` and `test_solvers.py:58`
- `_assert_split_ownership(reports, n_ranks)` — identical in `test_operators.py:112` and `test_solvers.py:62`

`test_utils.py` has its own inline `subprocess.run` call (not using the helper), but its MPI test could also benefit from the shared helper.

- [ ] **Step 1: Write `tests/mpi_helpers.py`**

```python
"""Shared MPI subprocess helpers for test harnesses."""
from __future__ import annotations

import json
import subprocess
import sys


def run_mpi_python(n_ranks: int, script: str) -> subprocess.CompletedProcess[str]:
    """Launch a Python script under mpiexec and return the completed process."""
    return subprocess.run(
        ["mpiexec", "-n", str(n_ranks), sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_mpi_reports(stdout: str) -> list[dict]:
    """Extract JSON report lines emitted by MPI rank scripts."""
    return [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]


def assert_split_ownership(reports: list[dict], n_ranks: int) -> None:
    """Verify that MPI ownership ranges partition the matrix correctly."""
    assert len(reports) == n_ranks, reports
    reports = sorted(reports, key=lambda item: item["rank"])
    assert {report["rank"] for report in reports} == set(range(n_ranks))

    ranges = [tuple(report["ownership_range"]) for report in reports]
    assert len(set(ranges)) == n_ranks
    assert ranges[0][0] == 0
    for prev, curr in zip(ranges, ranges[1:]):
        assert prev[1] == curr[0], ranges

    nrows = reports[0]["nrows"]
    assert ranges[-1][1] == nrows
    assert sum(r1 - r0 for r0, r1 in ranges) == nrows
    assert any(r0 > 0 for r0, _ in ranges)
```

Note: public names (no leading underscore) since this is a shared module.

- [ ] **Step 2: Verify the module imports cleanly**

Run:
```bash
.venv/bin/python -c "from tests.mpi_helpers import run_mpi_python, parse_mpi_reports, assert_split_ownership; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add tests/mpi_helpers.py
git commit -m "tests: extract shared MPI subprocess helpers into mpi_helpers.py"
```

### Task 2: Replace duplicates in test_operators.py

**Files:**
- Modify: `tests/test_operators.py:1-11` (imports), `tests/test_operators.py:99-131` (delete helpers)

- [ ] **Step 1: Add import and remove the three duplicate functions**

Add to the import block:
```python
from tests.mpi_helpers import run_mpi_python, parse_mpi_reports, assert_split_ownership
```

Delete the local definitions of `_run_mpi_python`, `_parse_mpi_reports`, `_assert_split_ownership` (lines 99-131 approximately).

- [ ] **Step 2: Update all call sites from `_run_mpi_python` to `run_mpi_python` etc.**

The calls are in:
- `test_build_lnse_system_petsc_real_mpi_assembly_matches_owned_scipy_rows`
- `test_build_lnse_system_petsc_real_mpi_dirichlet_matches_owned_scipy_rows`
- `test_build_lnse_system_petsc_real_mpi_neumann_matches_owned_scipy_rows`

Each uses `_run_mpi_python(...)`, `_parse_mpi_reports(...)`, `_assert_split_ownership(...)` — drop the leading underscore.

- [ ] **Step 3: Run the affected tests**

```bash
.venv/bin/python -m pytest tests/test_operators.py -v -x --timeout=120
```

Expected: all tests pass, same behavior as before.

- [ ] **Step 4: Commit**

```bash
git add tests/test_operators.py
git commit -m "tests: use shared mpi_helpers in test_operators"
```

### Task 3: Replace duplicates in test_solvers.py

**Files:**
- Modify: `tests/test_solvers.py:3-10` (imports), `tests/test_solvers.py:49-82` (delete helpers)

- [ ] **Step 1: Add import and remove the three duplicate functions**

Same pattern as Task 2. Add import, delete local defs, rename calls.

- [ ] **Step 2: Run the affected tests**

```bash
.venv/bin/python -m pytest tests/test_solvers.py -v -x --timeout=120
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_solvers.py
git commit -m "tests: use shared mpi_helpers in test_solvers"
```

### Task 4: Simplify MPI subprocess call in test_utils.py

**Files:**
- Modify: `tests/test_utils.py`

`test_utils.py` has an inline `subprocess.run(["mpiexec", ...])` call in `test_petsc_owned_rows_matches_owned_csr_blocks_under_mpi`. It doesn't use the helper pattern but could benefit from `run_mpi_python` and `parse_mpi_reports`.

- [ ] **Step 1: Add import**

```python
from tests.mpi_helpers import run_mpi_python, parse_mpi_reports
```

- [ ] **Step 2: Replace the inline subprocess.run with `run_mpi_python`**

Replace:
```python
proc = subprocess.run(
    ["mpiexec", "-n", "2", sys.executable, "-c", script],
    check=False,
    capture_output=True,
    text=True,
)
```

With:
```python
proc = run_mpi_python(2, script)
```

Replace the inline JSON parsing:
```python
reports = [
    json.loads(line)
    for line in proc.stdout.splitlines()
    if line.strip().startswith("{")
]
```

With:
```python
reports = parse_mpi_reports(proc.stdout)
```

- [ ] **Step 3: Remove now-unused imports**

`json`, `subprocess`, `sys` may no longer be needed in `test_utils.py` — check and remove if unused.

- [ ] **Step 4: Run the affected tests**

```bash
.venv/bin/python -m pytest tests/test_utils.py -v -x --timeout=120
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_utils.py
git commit -m "tests: use shared mpi_helpers in test_utils"
```

---

## Chunk 2: PETSc-native int type for row indices

### Task 5: Replace `dtype=np.int32` with `PETSc.IntType` in linsolve.py

**Files:**
- Modify: `src/linstab2d/solvers/linsolve.py:200`

- [ ] **Step 1: Change the dtype**

At line 200:
```python
# Before:
rows = np.arange(r0, r1, dtype=np.int32)
# After:
rows = np.arange(r0, r1, dtype=PETSc.IntType)
```

`PETSc` is already imported in this file.

- [ ] **Step 2: Run solver tests**

```bash
.venv/bin/python -m pytest tests/test_solvers.py -v -x --timeout=120
```

- [ ] **Step 3: Commit**

```bash
git add src/linstab2d/solvers/linsolve.py
git commit -m "fix: use PETSc.IntType for row indices in linear solver vec creation"
```

### Task 6: Replace `dtype=np.int32` with `PETSc.IntType` in assembly.py

**Files:**
- Modify: `src/linstab2d/operators/assembly.py:505,907`

- [ ] **Step 1: Change the dtype at both locations**

Line 505 (in `_build_lnse_direct_petsc`):
```python
# Before:
local_rows = np.arange(r0, r1, dtype=np.int32)
# After:
local_rows = np.arange(r0, r1, dtype=PETSc.IntType)
```

Line 907 (in `_apply_inv_B_petsc`):
```python
# Before:
rows = np.arange(r0, r1, dtype=np.int32)
# After:
rows = np.arange(r0, r1, dtype=PETSc.IntType)
```

Verify `PETSc` is already imported in this file (it is, via `from petsc4py import PETSc`).

- [ ] **Step 2: Run operator tests**

```bash
.venv/bin/python -m pytest tests/test_operators.py -v -x --timeout=120
```

- [ ] **Step 3: Commit**

```bash
git add src/linstab2d/operators/assembly.py
git commit -m "fix: use PETSc.IntType for row indices in assembly diagonal scaling"
```

**Note:** There are more `dtype=np.int32` occurrences in `dirichlet.py`, `neumann.py`, `eigenvalue.py`, and `spatial.py`. Those use int32 for PETSc row/column index arrays passed to `zeroRows`, `zeroRowsColumns`, `setValues`, etc. They should also be `PETSc.IntType` for consistency, but those files are not part of the current diff. Flag for a follow-up pass if desired — do not expand scope here.

---

## Chunk 3: Tolerance tracking and LinearSolver cleanup

### Task 7: Add TODO comments to relaxed tolerances

**Files:**
- Modify: `tests/test_integration.py:136,219`

- [ ] **Step 1: Add tracking comments**

At line 136 (resolvent gain threshold):
```python
# TODO: tighten to > 5.0 once BCs use the full non-reflecting treatment
# instead of the current `lrbt` Dirichlet-everywhere constraint.
assert gains[0] > 1.0, (
```

At line 219 (cylindrical eigenvalue rtol):
```python
# TODO: tighten to rtol=1e-6 once the axy cylindrical path is
# validated against a higher-resolution reference.
np.testing.assert_allclose(
    closest.imag, omega_ref_imag, rtol=5e-3,
```

- [ ] **Step 2: Run integration tests to confirm no accidental changes**

```bash
.venv/bin/python -m pytest tests/test_integration.py -v -x --timeout=300
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "tests: add TODO comments tracking relaxed tolerance thresholds"
```

### Task 8: Add `destroy()` method to `LinearSolver`

**Files:**
- Modify: `src/linstab2d/solvers/linsolve.py:35-43` (dataclass), `src/linstab2d/solvers/linsolve.py:248-254` (return statement)

The current `LinearSolver` captures `ksp` and `A` (a PETSc Mat copy) in closures but never exposes cleanup. In MPI contexts, leaked PETSc objects can cause hangs at finalize.

- [ ] **Step 1: Change `LinearSolver` from frozen dataclass to regular dataclass with a destroy method**

```python
@dataclass
class LinearSolver:
    """Callable wrapper for forward and adjoint solves."""

    backend: str
    solve: Callable[[np.ndarray], np.ndarray]
    solve_transpose: Callable[[np.ndarray], np.ndarray]
    solve_vec: Callable[[PETSc.Vec], PETSc.Vec]
    solve_transpose_vec: Callable[[PETSc.Vec], PETSc.Vec]
    _ksp: PETSc.KSP | None = None
    _mat: PETSc.Mat | None = None

    def destroy(self) -> None:
        """Release PETSc KSP and matrix resources."""
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        if self._mat is not None:
            self._mat.destroy()
            self._mat = None
```

Note: drop `frozen=True` since `destroy()` needs to mutate. The callables are still set once at construction and not reassigned in practice.

- [ ] **Step 2: Pass `ksp` and `A` to the dataclass at construction**

In `make_linear_solver`, change the return to:

```python
return LinearSolver(
    backend=backend,
    solve=solve,
    solve_transpose=solve_transpose,
    solve_vec=solve_vec,
    solve_transpose_vec=solve_transpose_vec,
    _ksp=ksp,
    _mat=A,
)
```

- [ ] **Step 3: Write a test for `destroy()`**

In `tests/test_solvers.py`, add a small test in the existing `TestDistributedPetscFallbackSolverInputs` class or as a standalone:

```python
def test_linear_solver_destroy_releases_petsc_objects(self):
    """LinearSolver.destroy() should release KSP and Mat without error."""
    n = 50
    diag = np.arange(1, n + 1, dtype=complex) + 0.1j
    A_sp = sp.diags(diag, format="csc")
    solver = make_linear_solver(A_sp)
    assert solver._ksp is not None
    assert solver._mat is not None
    solver.destroy()
    assert solver._ksp is None
    assert solver._mat is None
    # Double destroy should be safe (no-op)
    solver.destroy()
```

- [ ] **Step 4: Run solver tests**

```bash
.venv/bin/python -m pytest tests/test_solvers.py -v -x --timeout=120
```

- [ ] **Step 5: Commit**

```bash
git add src/linstab2d/solvers/linsolve.py tests/test_solvers.py
git commit -m "feat: add destroy() to LinearSolver for PETSc resource cleanup"
```

---

## Final Verification

### Task 9: Run full test suite

- [ ] **Step 1: Run the core test files**

```bash
.venv/bin/python -m pytest tests/test_utils.py tests/test_operators.py tests/test_solvers.py tests/test_integration.py -v --timeout=300
```

All tests should pass with identical behavior to before these changes.

- [ ] **Step 2: Verify no regressions in other test files**

```bash
.venv/bin/python -m pytest tests/ -v --timeout=600 -x
```

---

## Definition of Done

- Duplicate MPI test helpers extracted into `tests/mpi_helpers.py` and all three test files import from it
- `PETSc.IntType` used for row-index arrays in `linsolve.py` and `assembly.py`
- Relaxed tolerances in `test_integration.py` have TODO comments with tightening conditions
- `LinearSolver` exposes `destroy()` for PETSc resource cleanup, with a test
- Full test suite passes

## Out of Scope

- Changing `dtype=np.int32` in files outside the current diff (`dirichlet.py`, `neumann.py`, `eigenvalue.py`, `spatial.py`)
- `petsc_owned_rows` return type refactor (low priority, dict is fine for now)
- Song2023 validation ladder
- Any behavioral or numerical changes
