# SPDX-License-Identifier: GPL-3.0-only
"""Compatibility tests for deprecated solver import paths."""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np

from linstabpy.linalg import backends as linalg_backends
from linstabpy.linalg.factorize import Factorization, factorize
from linstabpy.resolvent.analysis import resolvent_analysis


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


def test_resolvent_shim_attrs_warn_and_work() -> None:
    import scipy.sparse as sp

    import linstabpy.solvers.resolvent as resolvent_shim

    L0 = sp.diags(-np.array([1.0, 2.0, 3.0], dtype=float), format="csc", dtype=complex)
    W_diag = np.ones(3)
    solver_cfg = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        old_analysis = resolvent_shim.resolvent_analysis
        old_postcheck = resolvent_shim.compute_resolvent_svd_residuals
        gains, responses, forcings = old_analysis(
            L0,
            omega=0.25,
            n_modes=1,
            W_diag=W_diag,
            solver_cfg=solver_cfg,
        )
        residual_payload = resolvent_shim.resolvent_analysis_with_residuals(
            L0,
            omega=0.25,
            n_modes=1,
            W_diag=W_diag,
            solver_cfg=solver_cfg,
        )
    assert old_postcheck is not None
    assert sum(item.category is DeprecationWarning for item in caught) >= 3

    np.testing.assert_array_equal(gains, residual_payload[0])
    np.testing.assert_array_equal(responses, residual_payload[1])
    np.testing.assert_array_equal(forcings, residual_payload[2])
    assert len(residual_payload) == 5


def test_plain_import_linstabpy_does_not_import_petsc() -> None:
    code = "import sys; import linstabpy; print('petsc4py' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "False"


def test_plain_import_linstabpy_resolvent_does_not_import_petsc() -> None:
    code = "import sys; import linstabpy.resolvent; print('petsc4py' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "False"
