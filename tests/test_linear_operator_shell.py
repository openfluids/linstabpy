# SPDX-License-Identifier: GPL-3.0-only
"""Tests for PetscShellOperator: wrapping LinearOperator as PETSc Mat.

Validates that the shell matrix produces identical results to direct
operator application.
"""

from __future__ import annotations

import numpy as np
import pytest

from linstab2d.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from linstab2d.linear_operators import (
    MatrixOperator,
    ProductOperator,
    PetscShellOperator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_complex_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPetscShellOperator:
    """Test PetscShellOperator.as_mat."""

    def test_shell_mult(self):
        """Shell mult matches direct MatrixOperator.apply."""
        rng = np.random.default_rng(400)
        n = 12
        A_np = _random_complex_matrix(n, rng)
        op = MatrixOperator(_np_to_petsc_mat(A_np))
        shell = PetscShellOperator.as_mat(op)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)
        y = shell.createVecLeft()

        shell.mult(x, y)
        y_np = _vec_to_np(y)
        y_ref = A_np @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        x.destroy()
        y.destroy()
        shell.destroy()

    def test_shell_mult_hermitian(self):
        """Shell multHermitian matches A^H x."""
        rng = np.random.default_rng(401)
        n = 12
        A_np = _random_complex_matrix(n, rng)
        op = MatrixOperator(_np_to_petsc_mat(A_np))
        shell = PetscShellOperator.as_mat(op)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)
        y = shell.createVecRight()

        shell.multHermitian(x, y)
        y_np = _vec_to_np(y)
        y_ref = A_np.conj().T @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        x.destroy()
        y.destroy()
        shell.destroy()

    def test_shell_mult_transpose(self):
        r"""Shell multTranspose matches A^T x (NOT A^H x).

        Uses the identity A^T x = conj(A^H conj(x)).
        """
        rng = np.random.default_rng(402)
        n = 12
        A_np = _random_complex_matrix(n, rng)
        op = MatrixOperator(_np_to_petsc_mat(A_np))
        shell = PetscShellOperator.as_mat(op)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)
        y = shell.createVecRight()

        shell.multTranspose(x, y)
        y_np = _vec_to_np(y)
        y_ref = A_np.T @ x_np  # Transpose, NOT Hermitian transpose

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        x.destroy()
        y.destroy()
        shell.destroy()

    def test_shell_wrapping_product(self):
        """Shell can wrap a ProductOperator."""
        rng = np.random.default_rng(403)
        n = 10
        A1_np = _random_complex_matrix(n, rng)
        A2_np = _random_complex_matrix(n, rng)

        op1 = MatrixOperator(_np_to_petsc_mat(A1_np))
        op2 = MatrixOperator(_np_to_petsc_mat(A2_np))
        product = ProductOperator([op1, op2], [op1.apply, op2.apply])

        shell = PetscShellOperator.as_mat(product)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)
        y = shell.createVecLeft()

        shell.mult(x, y)
        y_np = _vec_to_np(y)
        y_ref = A2_np @ (A1_np @ x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        x.destroy()
        y.destroy()
        shell.destroy()
        product.destroy()

    def test_shell_dimensions(self):
        """Shell matrix has correct global size."""
        rng = np.random.default_rng(404)
        n = 12
        A_np = _random_complex_matrix(n, rng)
        op = MatrixOperator(_np_to_petsc_mat(A_np))
        shell = PetscShellOperator.as_mat(op)

        assert shell.getSize() == (n, n)
        shell.destroy()
