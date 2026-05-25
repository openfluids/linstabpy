# SPDX-License-Identifier: GPL-3.0-only
"""Tests for MatrixOperator: apply, apply_adjoint, solve, solve_adjoint.

Validates all four actions against dense numpy references on small random
matrices.  Adapted from resolvent4py test patterns (Padovan et al.).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la

from linstabpy.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from linstabpy.linear_operators import MatrixOperator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_complex_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Invertible n×n complex matrix."""
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    # Ensure invertibility by adding diagonal dominance.
    A += n * np.eye(n)
    return A


def _numpy_to_petsc_mat(A: np.ndarray) -> PETSc.Mat:
    """Dense numpy → PETSc AIJ."""
    import scipy.sparse as sp

    csr = sp.csr_matrix(A)
    mat = PETSc.Mat().createAIJ(size=csr.shape, csr=(csr.indptr, csr.indices, csr.data))
    mat.assemble()
    return mat


def _numpy_to_petsc_vec(v: np.ndarray) -> PETSc.Vec:
    return PETSc.Vec().createWithArray(np.ascontiguousarray(v, dtype=complex))


def _petsc_vec_to_numpy(v: PETSc.Vec) -> np.ndarray:
    return np.array(v.array_r, copy=True)


def _create_ksp(A_petsc: PETSc.Mat) -> PETSc.KSP:
    """Direct MUMPS KSP for a PETSc Mat."""
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setUp()
    return ksp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_system():
    """Return (A_numpy, A_petsc, ksp, x_numpy, x_petsc) for a small problem."""
    rng = np.random.default_rng(42)
    n = 20
    A_np = _random_complex_matrix(n, rng)
    A_petsc = _numpy_to_petsc_mat(A_np)
    ksp = _create_ksp(A_petsc)
    x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x_petsc = _numpy_to_petsc_vec(x_np)
    return A_np, A_petsc, ksp, x_np, x_petsc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMatrixOperatorApply:
    """Test forward and adjoint application."""

    def test_apply(self, random_system):
        A_np, A_petsc, ksp, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp)

        y = op.apply(x_petsc)
        y_np = _petsc_vec_to_numpy(y)
        y_ref = A_np @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()

    def test_apply_adjoint(self, random_system):
        A_np, A_petsc, ksp, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp)

        y = op.apply_adjoint(x_petsc)
        y_np = _petsc_vec_to_numpy(y)
        y_ref = A_np.conj().T @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()

    def test_apply_with_preallocated_output(self, random_system):
        A_np, A_petsc, ksp, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp)

        y = op.create_left_vector()
        result = op.apply(x_petsc, y)
        assert result is y

        y_np = _petsc_vec_to_numpy(y)
        y_ref = A_np @ x_np
        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()


class TestMatrixOperatorSolve:
    """Test forward and adjoint solves."""

    def test_solve(self, random_system):
        A_np, A_petsc, ksp, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp)

        y = op.solve(x_petsc)
        y_np = _petsc_vec_to_numpy(y)
        y_ref = la.solve(A_np, x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()

    def test_solve_adjoint(self, random_system):
        A_np, A_petsc, ksp, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp)

        y = op.solve_adjoint(x_petsc)
        y_np = _petsc_vec_to_numpy(y)
        y_ref = la.solve(A_np.conj().T, x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()

    def test_solve_without_ksp_raises(self, random_system):
        A_np, A_petsc, _, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp=None)

        with pytest.raises(RuntimeError, match="requires a KSP"):
            op.solve(x_petsc)

    def test_solve_adjoint_without_ksp_raises(self, random_system):
        A_np, A_petsc, _, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp=None)

        with pytest.raises(RuntimeError, match="requires a KSP"):
            op.solve_adjoint(x_petsc)


class TestMatrixOperatorRoundtrip:
    """Test that solve(apply(x)) ≈ x and apply(solve(x)) ≈ x."""

    def test_solve_of_apply(self, random_system):
        """L^{-1} L x ≈ x."""
        A_np, A_petsc, ksp, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp)

        Lx = op.apply(x_petsc)
        recovered = op.solve(Lx)
        rec_np = _petsc_vec_to_numpy(recovered)

        assert np.linalg.norm(rec_np - x_np) / np.linalg.norm(x_np) < 1e-12
        Lx.destroy()
        recovered.destroy()

    def test_adjoint_solve_of_adjoint_apply(self, random_system):
        """L^{-H} L^H x ≈ x."""
        A_np, A_petsc, ksp, x_np, x_petsc = random_system
        op = MatrixOperator(A_petsc, ksp)

        LHx = op.apply_adjoint(x_petsc)
        recovered = op.solve_adjoint(LHx)
        rec_np = _petsc_vec_to_numpy(recovered)

        assert np.linalg.norm(rec_np - x_np) / np.linalg.norm(x_np) < 1e-12
        LHx.destroy()
        recovered.destroy()


class TestMatrixOperatorMetadata:
    """Test dimensions, name, vector constructors."""

    def test_dimensions(self, random_system):
        A_np, A_petsc, ksp, _, _ = random_system
        op = MatrixOperator(A_petsc, ksp)
        n = A_np.shape[0]
        # Single rank: local = global.
        assert op.dimensions == ((n, n), (n, n))

    def test_name(self, random_system):
        _, A_petsc, ksp, _, _ = random_system
        op = MatrixOperator(A_petsc, ksp)
        assert op.name == "MatrixOperator"

    def test_create_vectors(self, random_system):
        A_np, A_petsc, ksp, _, _ = random_system
        op = MatrixOperator(A_petsc, ksp)
        n = A_np.shape[0]

        vr = op.create_right_vector()
        vl = op.create_left_vector()
        assert vr.getSize() == n
        assert vl.getSize() == n
        vr.destroy()
        vl.destroy()
