"""Tests for ProductOperator: forward and adjoint chains.

Validates that composite operator actions match dense numpy references.
Adapted from resolvent4py test patterns (Padovan et al.).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la

from linstab2d.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from linstab2d.linear_operators import MatrixOperator, ProductOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_complex_matrix(
    rows: int, cols: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.standard_normal((rows, cols)) + 1j * rng.standard_normal(
        (rows, cols)
    )


def _invertible_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    A = _random_complex_matrix(n, n, rng)
    A += n * np.eye(n)
    return A


def _np_to_petsc_mat(A: np.ndarray) -> PETSc.Mat:
    import scipy.sparse as sp

    csr = sp.csr_matrix(A)
    mat = PETSc.Mat().createAIJ(
        size=csr.shape, csr=(csr.indptr, csr.indices, csr.data)
    )
    mat.assemble()
    return mat


def _np_to_vec(v: np.ndarray) -> PETSc.Vec:
    return PETSc.Vec().createWithArray(np.ascontiguousarray(v, dtype=complex))


def _vec_to_np(v: PETSc.Vec) -> np.ndarray:
    return np.array(v.array_r, copy=True)


def _create_ksp(A_petsc: PETSc.Mat) -> PETSc.KSP:
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setUp()
    return ksp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProductApply:
    """Test ProductOperator.apply against dense references."""

    def test_two_applies(self):
        """L = A2 @ A1 (both apply)."""
        rng = np.random.default_rng(100)
        n = 16
        A1_np = _random_complex_matrix(n, n, rng)
        A2_np = _random_complex_matrix(n, n, rng)

        A1 = MatrixOperator(_np_to_petsc_mat(A1_np))
        A2 = MatrixOperator(_np_to_petsc_mat(A2_np))

        product = ProductOperator([A1, A2], [A1.apply, A2.apply])

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = product.apply(x)
        y_np = _vec_to_np(y)
        y_ref = A2_np @ (A1_np @ x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()
        product.destroy()

    def test_solve_then_apply(self):
        """L = A2 @ A1^{-1} (solve, then apply)."""
        rng = np.random.default_rng(101)
        n = 16
        A1_np = _invertible_matrix(n, rng)
        A2_np = _random_complex_matrix(n, n, rng)

        A1_petsc = _np_to_petsc_mat(A1_np)
        ksp1 = _create_ksp(A1_petsc)
        op1 = MatrixOperator(A1_petsc, ksp1)
        op2 = MatrixOperator(_np_to_petsc_mat(A2_np))

        product = ProductOperator([op1, op2], [op1.solve, op2.apply])

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = product.apply(x)
        y_np = _vec_to_np(y)
        y_ref = A2_np @ la.solve(A1_np, x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()
        product.destroy()

    def test_three_operator_chain(self):
        """L = A3 @ A2^{-1} @ A1^H."""
        rng = np.random.default_rng(102)
        n = 12
        A1_np = _random_complex_matrix(n, n, rng)
        A2_np = _invertible_matrix(n, rng)
        A3_np = _random_complex_matrix(n, n, rng)

        A2_petsc = _np_to_petsc_mat(A2_np)
        ksp2 = _create_ksp(A2_petsc)

        op1 = MatrixOperator(_np_to_petsc_mat(A1_np))
        op2 = MatrixOperator(A2_petsc, ksp2)
        op3 = MatrixOperator(_np_to_petsc_mat(A3_np))

        product = ProductOperator(
            [op1, op2, op3],
            [op1.apply_adjoint, op2.solve, op3.apply],
        )

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = product.apply(x)
        y_np = _vec_to_np(y)
        y_ref = A3_np @ la.solve(A2_np, A1_np.conj().T @ x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()
        product.destroy()


class TestProductAdjoint:
    """Test ProductOperator.apply_adjoint against dense references."""

    def test_adjoint_of_two_applies(self):
        """(A2 A1)^H = A1^H A2^H."""
        rng = np.random.default_rng(200)
        n = 14
        A1_np = _random_complex_matrix(n, n, rng)
        A2_np = _random_complex_matrix(n, n, rng)

        op1 = MatrixOperator(_np_to_petsc_mat(A1_np))
        op2 = MatrixOperator(_np_to_petsc_mat(A2_np))

        product = ProductOperator([op1, op2], [op1.apply, op2.apply])

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = product.apply_adjoint(x)
        y_np = _vec_to_np(y)
        # (A2 A1)^H x = A1^H A2^H x
        y_ref = A1_np.conj().T @ (A2_np.conj().T @ x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()
        product.destroy()

    def test_adjoint_of_solve_then_apply(self):
        """(A2 A1^{-1})^H = A1^{-H} A2^H."""
        rng = np.random.default_rng(201)
        n = 14
        A1_np = _invertible_matrix(n, rng)
        A2_np = _random_complex_matrix(n, n, rng)

        A1_petsc = _np_to_petsc_mat(A1_np)
        ksp1 = _create_ksp(A1_petsc)
        op1 = MatrixOperator(A1_petsc, ksp1)
        op2 = MatrixOperator(_np_to_petsc_mat(A2_np))

        product = ProductOperator([op1, op2], [op1.solve, op2.apply])

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = product.apply_adjoint(x)
        y_np = _vec_to_np(y)
        # (A2 A1^{-1})^H x = A1^{-H} A2^H x
        y_ref = la.solve(A1_np.conj().T, A2_np.conj().T @ x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()
        product.destroy()

    def test_adjoint_consistency_inner_product(self):
        """<Lx, z> = <x, L^H z> for random x, z."""
        rng = np.random.default_rng(202)
        n = 14
        A1_np = _random_complex_matrix(n, n, rng)
        A2_np = _random_complex_matrix(n, n, rng)

        op1 = MatrixOperator(_np_to_petsc_mat(A1_np))
        op2 = MatrixOperator(_np_to_petsc_mat(A2_np))
        product = ProductOperator([op1, op2], [op1.apply, op2.apply])

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        z_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)
        z = _np_to_vec(z_np)

        Lx = product.apply(x)
        LHz = product.apply_adjoint(z)

        Lx_np = _vec_to_np(Lx)
        LHz_np = _vec_to_np(LHz)

        ip_left = np.vdot(Lx_np, z_np)    # <Lx, z>
        ip_right = np.vdot(x_np, LHz_np)  # <x, L^H z>

        assert abs(ip_left - ip_right) / abs(ip_left) < 1e-12
        Lx.destroy()
        LHz.destroy()
        x.destroy()
        z.destroy()
        product.destroy()


class TestProductValidation:
    """Edge cases and error handling."""

    def test_single_operator_raises(self):
        rng = np.random.default_rng(300)
        A = MatrixOperator(_np_to_petsc_mat(_random_complex_matrix(4, 4, rng)))
        with pytest.raises(ValueError, match="at least 2"):
            ProductOperator([A], [A.apply])

    def test_mismatched_lengths_raises(self):
        rng = np.random.default_rng(301)
        A = MatrixOperator(_np_to_petsc_mat(_random_complex_matrix(4, 4, rng)))
        B = MatrixOperator(_np_to_petsc_mat(_random_complex_matrix(4, 4, rng)))
        with pytest.raises(ValueError, match="len"):
            ProductOperator([A, B], [A.apply])

    def test_invalid_action_raises(self):
        rng = np.random.default_rng(302)
        A = MatrixOperator(_np_to_petsc_mat(_random_complex_matrix(4, 4, rng)))
        B = MatrixOperator(_np_to_petsc_mat(_random_complex_matrix(4, 4, rng)))
        with pytest.raises(ValueError, match="not one of"):
            ProductOperator([A, B], [A.apply, lambda x, y=None: y])
