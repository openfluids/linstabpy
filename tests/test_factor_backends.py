# SPDX-License-Identifier: GPL-3.0-only
"""Tests for PETSc sparse-direct factor backend selection."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp
from petsc4py import PETSc

from linstabpy.solvers import backends
from linstabpy.solvers.linsolve import make_linear_solver


def test_resolve_factor_backend_precedence(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_FACTOR_BACKEND", "superlu")
    monkeypatch.setenv("LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE", "mumps")

    assert backends.resolve_factor_backend(None) == "superlu"
    assert (
        backends.resolve_factor_backend(
            None,
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
        )
        == "mumps"
    )
    assert (
        backends.resolve_factor_backend(
            {"factor_backend": "petsc"},
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
            default="mumps",
        )
        is None
    )
    assert (
        backends.resolve_factor_backend(
            {"factor_backend": "MUMPS"},
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
        )
        == "mumps"
    )


def test_mumps_probe_and_apply_configure_lu_without_cuda_imports() -> None:
    before_cuda_modules = {name for name in sys.modules if "cuda" in name.lower()}

    assert backends.resolve_factor_backend(None) == "mumps"
    assert backends.factor_backend_available("mumps") is True

    A = PETSc.Mat().createAIJ(size=(1, 1), comm=PETSc.COMM_SELF)
    A.setValue(0, 0, 1.0 + 0.0j)
    A.assemble()
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    try:
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        selected = backends.apply_factor_backend(pc, "mumps")
        ksp.setUp()

        assert selected == "mumps"
        assert ksp.getType() == PETSc.KSP.Type.PREONLY
        assert pc.getType() == PETSc.PC.Type.LU
        assert pc.getFactorSolverType() == "mumps"
    finally:
        ksp.destroy()
        A.destroy()

    after_cuda_modules = {name for name in sys.modules if "cuda" in name.lower()}
    assert after_cuda_modules == before_cuda_modules


def test_absent_backend_probe_and_error_message() -> None:
    missing = "linstabpy_absent_factor_backend"

    assert backends.factor_backend_available(missing) is False
    with pytest.raises(RuntimeError) as excinfo:
        backends.ensure_factor_backend_available(missing)

    message = str(excinfo.value)
    assert missing in message
    assert "PETSc" in message
    assert "Rebuild PETSc/petsc4py" in message
    assert "docs/COMPUTE_BACKENDS.md" in message


def test_direct_linsolve_uses_mumps_backend(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_FACTOR_BACKEND", raising=False)
    operator = sp.eye(2, format="csr", dtype=complex)

    solver = make_linear_solver(operator, solver_cfg={"linsolve": {"type": "direct"}})
    try:
        assert solver.backend == "MUMPS"
        assert solver._ksp is not None
        assert solver._ksp.getType() == PETSc.KSP.Type.PREONLY
        assert solver._ksp.getPC().getType() == PETSc.PC.Type.LU
        assert solver._ksp.getPC().getFactorSolverType() == "mumps"
        np.testing.assert_allclose(solver.solve(np.array([1.0, 2.0])), [1.0, 2.0])
    finally:
        solver.destroy()


def test_iterative_linsolve_does_not_enter_factor_backend_seam(monkeypatch) -> None:
    def fail_apply(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("direct factor seam should not run for iterative linsolve")

    monkeypatch.setattr("linstabpy.solvers.linsolve.apply_factor_backend", fail_apply)
    operator = sp.eye(2, format="csr", dtype=complex)

    solver = make_linear_solver(
        operator,
        solver_cfg={
            "linsolve": {
                "type": "iterative",
                "ksp_type": "gmres",
                "pc_type": "none",
                "ksp_max_it": 10,
            }
        },
    )
    try:
        assert solver.backend == "PETSc-iterative"
        assert solver._ksp is not None
        assert solver._ksp.getType() == "gmres"
        assert solver._ksp.getPC().getType() == "none"
    finally:
        solver.destroy()
