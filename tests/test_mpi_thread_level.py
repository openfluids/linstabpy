# SPDX-License-Identifier: GPL-3.0-only
"""MPI must be initialized at ``MPI_THREAD_MULTIPLE``, and PETSc must not win the race.

Why this matters
----------------
Whoever calls ``MPI_Init`` first decides the thread level for the whole process,
and the level can only be *requested* at init -- it can never be raised afterwards.
``mpi4py`` requests ``MPI_THREAD_MULTIPLE`` when it is imported; PETSc requests
only ``MPI_THREAD_FUNNELED``. So the level is decided by whichever of the two is
imported first, anywhere in the process.

STRUMPACK's SLATE backend requires ``MPI_THREAD_MULTIPLE``. When PETSc wins the
race, ``MatGetFactor_aij_strumpack()`` fails with PETSc error 57 ("No support for
this operation on this system") and the message ``SLATE requires
MPI_THREAD_MULTIPLE``. Measured on the GPU venv: 8 of 14 suite failures had this
single cause, and the entire STRUMPACK/GPU factorization route was unusable.

``utils.petsc_init.init_petsc_from_argv`` already imports ``mpi4py.MPI`` first and
documents the rule, but eight modules import ``petsc4py``/``slepc4py`` at *module
scope*, bypassing that helper entirely. Importing any one of them first loses the
race. Hence the lint below: the invariant is a property of the import graph, not
of any single call site, so it is enforced by a source scan.

The runtime check is kept separate and skips without petsc4py, so a CPU-only or
PETSc-free environment reports "skipped", never a silent pass.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

import linstabpy

#: Import roots that initialize MPI at the *wrong* thread level if they win the race.
_PETSC_ROOTS = ("petsc4py", "slepc4py")

#: The import that requests MPI_THREAD_MULTIPLE. Any module-scope PETSc import
#: must be preceded by it, in either spelling (``import mpi4py.MPI`` or
#: ``from mpi4py import MPI`` -- ``driver/run.py`` uses the latter next to its
#: ``init_petsc_from_argv()`` call). What matters is only that mpi4py binds MPI
#: first, not how it is spelled.
#:
#: Deliberately a *third-party* module rather than a first-party helper: isort
#: orders third-party imports above first-party ones, so a ``linstabpy.*`` guard
#: could never stay above ``petsc4py`` and ``ruff --fix`` would silently reorder
#: the fix away. ``mpi4py`` < ``petsc4py`` alphabetically, so the required order
#: is what the formatter produces on its own.
_GUARDS = frozenset({"mpi4py"})


def _package_root() -> pathlib.Path:
    return pathlib.Path(linstabpy.__file__).parent


def _module_scope_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Return (lineno, root_package) for every module-scope import.

    Only ``tree.body`` is walked, so imports inside functions, ``try`` blocks and
    ``if TYPE_CHECKING:`` guards are deliberately excluded: those do not run at
    import time and therefore cannot win the MPI init race.
    """
    out: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name.split(".", 1)[0]))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.lineno, node.module.split(".", 1)[0]))
    return out


def test_module_scope_petsc_imports_are_preceded_by_an_mpi4py_import() -> None:
    """No module may import petsc4py/slepc4py at module scope before the guard.

    Failing this means some import path can initialize MPI at THREAD_FUNNELED,
    which silently disables STRUMPACK. The fix is to add the guard import
    immediately above the offending petsc4py/slepc4py import -- never to relax
    this lint.
    """
    offenders: list[str] = []
    for path in sorted(_package_root().rglob("*.py")):
        imports = _module_scope_imports(ast.parse(path.read_text()))
        guard_line = min(
            (ln for ln, mod in imports if mod in _GUARDS),
            default=None,
        )
        for lineno, root in imports:
            if root not in _PETSC_ROOTS:
                continue
            if guard_line is None or guard_line > lineno:
                rel = path.relative_to(_package_root().parent)
                offenders.append(f"{rel}:{lineno} imports {root} before mpi4py")

    assert not offenders, (
        "module-scope PETSc import not preceded by the MPI_THREAD_MULTIPLE guard:\n  "
        + "\n  ".join(offenders)
        + "\n\nAdd `import mpi4py.MPI  # noqa: F401` immediately above the"
        " petsc4py/slepc4py import."
    )


def test_conftest_binds_mpi_before_collection_imports_petsc() -> None:
    """The test harness must bind MPI first, because collection can lose the race.

    Guarding the package source is not enough. pytest imports every test module
    during collection, and several import ``petsc4py``/``slepc4py`` at module
    scope; whichever is imported first sets the process-wide thread level for the
    whole session. ``tests/conftest.py`` is imported before any of them, so one
    ``import mpi4py.MPI`` there covers the entire suite.

    This was measured, not assumed: with the package guarded but conftest not, the
    GPU venv still reported all 14 STRUMPACK failures, while the same test passed
    when run as the only module.
    """
    conftest = pathlib.Path(__file__).with_name("conftest.py")
    assert conftest.is_file(), f"missing {conftest}"
    # ``ast.walk``, not ``_module_scope_imports``: the import is wrapped in a
    # ``contextlib.suppress(ModuleNotFoundError)`` block so a PETSc-free
    # environment can still run the pure-Python suite. It executes at import
    # time all the same, which is what decides the race.
    tree = ast.parse(conftest.read_text())
    found = any(
        alias.name.split(".", 1)[0] == "mpi4py"
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert found, (
        "tests/conftest.py must import mpi4py at import time, before pytest "
        "collects test modules that import petsc4py."
    )


@pytest.mark.parametrize(
    "first_import",
    [
        "import linstabpy.solvers.eigenvalue",
        "import linstabpy.driver.run",
        "import linstabpy.utils.cost_metrics",
    ],
)
def test_mpi_thread_level_is_multiple_after_importing_a_petsc_module(first_import: str) -> None:
    """Importing any PETSc-backed module must leave MPI at THREAD_MULTIPLE.

    Runs in a subprocess because the thread level is process-global and set once:
    a previously-imported mpi4py in the pytest process would mask the very bug
    this test exists to catch.
    """
    pytest.importorskip("petsc4py", reason="SKIPPED: petsc4py not installed in this venv")
    pytest.importorskip("mpi4py", reason="SKIPPED: mpi4py not installed in this venv")

    # A sentinel-tagged line, located by prefix: importing the solver stack can emit
    # PETSc/MPI banner text on stdout, so positional parsing of the stream is unsafe.
    code = (
        f"{first_import}\n"
        "from mpi4py import MPI\n"
        'print(f"__THREADLEVEL__ {MPI.Query_thread()} {MPI.THREAD_MULTIPLE}")\n'
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    tagged = [ln for ln in proc.stdout.splitlines() if ln.startswith("__THREADLEVEL__")]
    assert tagged, f"sentinel line missing from subprocess stdout:\n{proc.stdout}\n{proc.stderr}"
    provided, multiple = (int(x) for x in tagged[-1].split()[1:3])
    assert provided == multiple, (
        f"after `{first_import}` MPI thread level is {provided}, expected "
        f"{multiple} (MPI_THREAD_MULTIPLE). PETSc won the MPI init race and "
        "STRUMPACK/SLATE will refuse to build a factor."
    )
