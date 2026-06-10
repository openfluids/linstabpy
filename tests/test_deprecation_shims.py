# SPDX-License-Identifier: GPL-3.0-only
"""Compatibility tests for deprecated solver import paths."""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np

from linstabpy.linalg import backends as linalg_backends
from linstabpy.linalg.factorize import Factorization, factorize


def _deprecated_import(import_statement: str, namespace: dict[str, object]) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        exec(import_statement, namespace)  # noqa: S102
    assert any(item.category is DeprecationWarning for item in caught)


def test_linsolve_shim_warns_and_resolves_to_linalg_objects() -> None:
    namespace: dict[str, object] = {}

    _deprecated_import(
        "from linstabpy.solvers.linsolve import make_linear_solver, LinearSolver",
        namespace,
    )

    assert namespace["make_linear_solver"] is factorize
    assert namespace["LinearSolver"] is Factorization


def test_backends_shim_warns_and_resolves_to_linalg_objects() -> None:
    namespace: dict[str, object] = {}

    _deprecated_import(
        "from linstabpy.solvers.backends import resolve_factor_backend, apply_factor_backend",
        namespace,
    )

    assert namespace["resolve_factor_backend"] is linalg_backends.resolve_factor_backend
    assert namespace["apply_factor_backend"] is linalg_backends.apply_factor_backend


def test_solve_transpose_alias_forwards_to_solve_adjoint() -> None:
    def solve_adjoint(rhs: np.ndarray) -> np.ndarray:
        return rhs + 1.0

    fac = Factorization(
        backend="test",
        solve=lambda rhs: rhs,
        solve_adjoint=solve_adjoint,
        solve_vec=lambda rhs: rhs,
        solve_adjoint_vec=lambda rhs: rhs,
        residual=lambda _rhs, _x: 0.0,
    )

    rhs = np.array([1.0, 2.0])
    assert fac.solve_transpose is fac.solve_adjoint
    np.testing.assert_allclose(fac.solve_transpose(rhs), [2.0, 3.0])


def test_plain_import_linstabpy_does_not_import_petsc() -> None:
    code = "import sys; import linstabpy; print('petsc4py' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "False"
