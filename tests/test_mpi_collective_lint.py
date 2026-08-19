# SPDX-License-Identifier: GPL-3.0-only
"""Scan sources so no MPI collective sits inside a rank-0-only region.

Why this is a source scan and not a runtime check.
------------------------------------------------
An MPI collective (``allreduce``, ``bcast``, ``barrier``, ...) reached by only
one rank hangs every other rank forever.
The serial test suite runs at size 1, where rank 0 is the only rank.
The deadlock stays structurally invisible.
The suite goes green.
The hang appears only on a multi-rank cluster run, typically after a long
queue wait.
That is what happened here.
A reduction inside the rank-0-only NPZ writer deadlocked every multi-rank
run while the suite passed.

``test_refactorize_mpi_guard.py`` covers one function's collective branch
agreement.
This scan covers the whole package.
It is cheap: pure AST, no MPI, no PETSc.

The scan detects two shapes.
Both make a region rank-0-only.

1. ``if rank == 0:`` -- collectives lexically inside the body.
2. ``if rank != 0: return`` -- collectives anywhere in the rest of the function.

The repair is always the same.
Do the collective in the collective region.
Then pass the resulting scalar or array into the rank-0 block.

If a collective here is ever genuinely correct, name that communicator so it
does not match ``*rank*``, or restructure.
An example is a sub-communicator that really does contain only this rank.
Do not weaken the lint.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import linstabpy

#: mpi4py collective entry points. Deliberately does NOT include PETSc object
#: methods (``assemble``, ``setUp``, ...): those are collective too, but they appear
#: so widely that a lint over them would be noise rather than signal.
COLLECTIVES = frozenset(
    {
        "allreduce",
        "Allreduce",
        "bcast",
        "Bcast",
        "barrier",
        "Barrier",
        "gather",
        "Gather",
        "Gatherv",
        "allgather",
        "Allgather",
        "Allgatherv",
        "scatter",
        "Scatter",
        "Scatterv",
        "reduce",
        "Reduce",
        "Reduce_scatter",
        "alltoall",
        "Alltoall",
    }
)

PACKAGE_ROOT = pathlib.Path(linstabpy.__file__).parent


def _called_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", "")


def _is_rank_zero_test(test: ast.expr) -> bool:
    """Match ``rank == 0`` / ``comm.rank == 0`` / ``self.rank == 0``."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    right = test.comparators[0]
    if not (isinstance(right, ast.Constant) and right.value == 0):
        return False
    left = test.left
    name = left.attr if isinstance(left, ast.Attribute) else getattr(left, "id", "")
    return "rank" in name.lower()


def _is_rank_nonzero_early_exit(stmt: ast.stmt) -> bool:
    """Match ``if rank != 0: return`` (or ``raise``) -- rest of function is rank-0-only."""
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return False
    if not isinstance(test.ops[0], ast.NotEq):
        return False
    right = test.comparators[0]
    if not (isinstance(right, ast.Constant) and right.value == 0):
        return False
    left = test.left
    name = left.attr if isinstance(left, ast.Attribute) else getattr(left, "id", "")
    if "rank" not in name.lower():
        return False
    return any(isinstance(s, (ast.Return, ast.Raise)) for s in stmt.body)


def find_violations(source: str, filename: str = "<src>") -> list[str]:
    """Return ``file:line collective()`` for each collective in a rank-0-only region."""
    tree = ast.parse(source, filename=filename)
    found: list[str] = []

    def scan(body: list[ast.stmt], why: str) -> None:
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call) and _called_name(node) in COLLECTIVES:
                    found.append(f"{filename}:{node.lineno} {_called_name(node)}() [{why}]")

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_rank_zero_test(node.test):
            scan(node.body, "inside `if rank == 0:`")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for index, stmt in enumerate(node.body):
                if _is_rank_nonzero_early_exit(stmt):
                    scan(node.body[index + 1 :], f"after `if rank != 0: return` in {node.name}()")
                    break

    return found


def test_no_mpi_collective_in_rank_zero_region() -> None:
    """The real assertion: the shipped package contains no such deadlock."""
    violations: list[str] = []
    scanned = 0
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        scanned += 1
        rel = path.relative_to(PACKAGE_ROOT.parent)
        violations.extend(find_violations(path.read_text(encoding="utf-8"), str(rel)))

    assert scanned > 0, "lint scanned no files -- the package path is wrong"
    assert not violations, (
        "MPI collective(s) reached by rank 0 only -- this deadlocks every multi-rank "
        "run and the serial suite cannot see it:\n  "
        + "\n  ".join(violations)
        + "\nFix: perform the collective in the collective region and pass the result "
        "into the rank-0 block."
    )


# ----------------------------------------------------------------------------------
# Positive controls. A lint that has never been shown to fire is indistinguishable
# from one that cannot fire -- these pin its sensitivity to both deadlock shapes.
# ----------------------------------------------------------------------------------

VIOLATING_IF_BLOCK = """
def write_results(comm, data):
    if comm.rank == 0:
        total = comm.allreduce(len(data))
        save(total)
"""

VIOLATING_EARLY_RETURN = """
def write_results(comm, data):
    if comm.rank != 0:
        return
    total = comm.allreduce(len(data))
    save(total)
"""

CLEAN_EQUIVALENT = """
def write_results(comm, data):
    total = comm.allreduce(len(data))
    if comm.rank == 0:
        save(total)
"""


@pytest.mark.parametrize(
    "source,expect",
    [
        (VIOLATING_IF_BLOCK, True),
        (VIOLATING_EARLY_RETURN, True),
        (CLEAN_EQUIVALENT, False),
    ],
    ids=["if-rank-0-block", "early-return-guard", "clean-collective-first"],
)
def test_lint_sensitivity(source: str, expect: bool) -> None:
    assert bool(find_violations(source)) is expect
