# SPDX-License-Identifier: GPL-3.0-only
"""Tests for adapter functions and composed resolvent operator.

B1: scipy_to_operator, scipy_to_solve_operator, DiagonalOperator, NumpyCallableOperator
B2: build_shifted_operator
B3: Operator-composed resolvent vs current closure-based implementation
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from linstab2d.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc
from slepc4py import SLEPc

from linstab2d.linear_operators import (
    MatrixOperator,
    ProductOperator,
    PetscShellOperator,
    scipy_to_operator,
    scipy_to_solve_operator,
    build_shifted_operator,
    DiagonalOperator,
    NumpyCallableOperator,
)
from linstab2d.utils.petsc_core import petsc_vec, vec_to_numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_invertible(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A += n * np.eye(n)
    return A


def _np_to_vec(v: np.ndarray) -> PETSc.Vec:
    return PETSc.Vec().createWithArray(np.ascontiguousarray(v, dtype=complex))


def _vec_to_np(v: PETSc.Vec) -> np.ndarray:
    return np.array(v.array_r, copy=True)


# ---------------------------------------------------------------------------
# B1: Adapter tests
# ---------------------------------------------------------------------------

class TestScipyToOperator:
    """Test scipy_to_operator (apply-only adapter)."""

    def test_apply(self):
        rng = np.random.default_rng(500)
        n = 15
        A_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A_sp = sp.csc_matrix(A_np)

        op = scipy_to_operator(A_sp)
        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = op.apply(x)
        y_np = _vec_to_np(y)
        y_ref = A_np @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()

    def test_apply_adjoint(self):
        rng = np.random.default_rng(501)
        n = 15
        A_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A_sp = sp.csc_matrix(A_np)

        op = scipy_to_operator(A_sp)
        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = op.apply_adjoint(x)
        y_np = _vec_to_np(y)
        y_ref = A_np.conj().T @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()


class TestScipyToSolveOperator:
    """Test scipy_to_solve_operator (with MUMPS KSP)."""

    def test_solve(self):
        rng = np.random.default_rng(510)
        n = 15
        A_np = _random_invertible(n, rng)
        A_sp = sp.csc_matrix(A_np)

        op = scipy_to_solve_operator(A_sp)
        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = op.solve(x)
        y_np = _vec_to_np(y)
        y_ref = la.solve(A_np, x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()

    def test_roundtrip(self):
        """apply(solve(x)) ≈ x."""
        rng = np.random.default_rng(511)
        n = 15
        A_np = _random_invertible(n, rng)
        A_sp = sp.csc_matrix(A_np)

        op = scipy_to_solve_operator(A_sp)
        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        s = op.solve(x)
        rec = op.apply(s)
        rec_np = _vec_to_np(rec)

        assert np.linalg.norm(rec_np - x_np) / np.linalg.norm(x_np) < 1e-12
        s.destroy()
        rec.destroy()
        x.destroy()


class TestDiagonalOperator:
    """Test DiagonalOperator."""

    def test_apply(self):
        rng = np.random.default_rng(520)
        n = 20
        d = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        op = DiagonalOperator(d)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = op.apply(x)
        y_np = _vec_to_np(y)
        assert np.linalg.norm(y_np - d * x_np) < 1e-14
        y.destroy()
        x.destroy()

    def test_apply_adjoint(self):
        rng = np.random.default_rng(521)
        n = 20
        d = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        op = DiagonalOperator(d)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = op.apply_adjoint(x)
        y_np = _vec_to_np(y)
        assert np.linalg.norm(y_np - d.conj() * x_np) < 1e-14
        y.destroy()
        x.destroy()

    def test_solve(self):
        rng = np.random.default_rng(522)
        n = 20
        d = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        d[np.abs(d) < 0.1] = 1.0  # avoid near-zero
        op = DiagonalOperator(d)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = op.solve(x)
        y_np = _vec_to_np(y)
        assert np.linalg.norm(y_np - x_np / d) < 1e-13
        y.destroy()
        x.destroy()

    def test_real_diagonal_is_self_adjoint(self):
        """For real d, apply == apply_adjoint."""
        rng = np.random.default_rng(523)
        n = 15
        d = rng.standard_normal(n).astype(float)
        op = DiagonalOperator(d)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y1 = op.apply(x)
        y2 = op.apply_adjoint(x)
        assert np.linalg.norm(_vec_to_np(y1) - _vec_to_np(y2)) < 1e-14
        y1.destroy()
        y2.destroy()
        x.destroy()


class TestNumpyCallableOperator:
    """Test NumpyCallableOperator."""

    def test_apply(self):
        n = 10
        fn = lambda v: 2.0 * v + 1j * v
        op = NumpyCallableOperator(fn, n)

        x_np = np.ones(n, dtype=complex)
        x = _np_to_vec(x_np)

        y = op.apply(x)
        y_np = _vec_to_np(y)
        assert np.allclose(y_np, fn(x_np))
        y.destroy()
        x.destroy()

    def test_adjoint_raises_when_none(self):
        n = 10
        op = NumpyCallableOperator(lambda v: v, n)
        x = _np_to_vec(np.zeros(n, dtype=complex))

        with pytest.raises(NotImplementedError, match="no adjoint"):
            op.apply_adjoint(x)
        x.destroy()

    def test_adjoint_when_provided(self):
        n = 10
        fn = lambda v: 3.0 * v
        fn_adj = lambda v: 3.0 * v  # self-adjoint for real scalar
        op = NumpyCallableOperator(fn, n, fn_adjoint=fn_adj)

        x_np = np.arange(n, dtype=complex)
        x = _np_to_vec(x_np)

        y = op.apply_adjoint(x)
        y_np = _vec_to_np(y)
        assert np.allclose(y_np, fn_adj(x_np))
        y.destroy()
        x.destroy()


# ---------------------------------------------------------------------------
# B2: Shifted operator
# ---------------------------------------------------------------------------

class TestBuildShiftedOperator:
    """Test build_shifted_operator."""

    def test_shifted_solve(self):
        """inv(L0 - i*omega*I) x matches dense reference."""
        rng = np.random.default_rng(530)
        n = 20
        L0_np = _random_invertible(n, rng)
        omega = 0.5
        L_sigma_np = L0_np - 1j * omega * np.eye(n)

        L0_sp = sp.csc_matrix(L0_np)
        op = build_shifted_operator(L0_sp, omega)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = op.solve(x)
        y_np = _vec_to_np(y)
        y_ref = la.solve(L_sigma_np, x_np)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()

    def test_shifted_apply(self):
        """(L0 - i*omega*I) x matches dense reference."""
        rng = np.random.default_rng(531)
        n = 20
        L0_np = _random_invertible(n, rng)
        omega = 0.5
        L_sigma_np = L0_np - 1j * omega * np.eye(n)

        L0_sp = sp.csc_matrix(L0_np)
        op = build_shifted_operator(L0_sp, omega)

        x_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = _np_to_vec(x_np)

        y = op.apply(x)
        y_np = _vec_to_np(y)
        y_ref = L_sigma_np @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()


# ---------------------------------------------------------------------------
# B3: Operator-composed resolvent vs closure-based implementation
# ---------------------------------------------------------------------------

class TestComposedResolvent:
    """Compare operator-composed resolvent gains against the current
    closure-based resolvent_analysis() on a small random problem."""

    @staticmethod
    def _resolvent_gains_closure(L0_np, omega, W_diag, n_modes):
        """Current-style closure resolvent (numpy level)."""
        n = L0_np.shape[0]
        L_sigma = L0_np - 1j * omega * np.eye(n)
        L_sigma_inv = la.inv(L_sigma)
        invW = 1.0 / W_diag

        def _apply_H(v):
            return L_sigma_inv @ v

        def _apply_H_adj(v):
            return la.solve(L_sigma.conj().T, v)

        # Build dense HtrH and compute eigenvalues directly.
        HtrH = np.zeros((n, n), dtype=complex)
        for j in range(n):
            ej = np.zeros(n, dtype=complex)
            ej[j] = 1.0
            Hej = _apply_H(ej)
            WHej = W_diag * Hej
            HH_WHej = _apply_H_adj(WHej)
            HtrH[:, j] = invW * HH_WHej

        eigs = la.eigvals(HtrH)
        eigs_real = np.real(eigs)
        eigs_real = np.sort(eigs_real)[::-1]
        gains = np.sqrt(np.maximum(eigs_real[:n_modes], 0.0))
        return gains

    @staticmethod
    def _resolvent_gains_composed(L0_sp, omega, W_diag, n_modes):
        """Operator-composed resolvent using new abstraction layer."""
        n = L0_sp.shape[0]
        invW_diag = 1.0 / W_diag

        L_sigma_op = build_shifted_operator(L0_sp, omega)
        W_op = DiagonalOperator(W_diag)
        invW_op = DiagonalOperator(invW_diag)

        # H = inv(L_sigma)  (identity B and C)
        # H^H = inv(L_sigma^H)
        # HtrH(v) = invW * H^H(W * H(v))
        #
        # As a ProductOperator chain:
        #   v → solve(L_sigma, v) → W*(...) → solve_adjoint(L_sigma, ...) → invW*(...)
        product = ProductOperator(
            [L_sigma_op, W_op, L_sigma_op, invW_op],
            [L_sigma_op.solve, W_op.apply, L_sigma_op.solve_adjoint, invW_op.apply],
        )

        # Wrap as PETSc MatShell for SLEPc EPS.
        htrh_shell = PetscShellOperator.as_mat(product)

        ncv = min(max(2 * n_modes + 3, 20), n - 1)
        eps = SLEPc.EPS().create()
        eps.setOperators(htrh_shell)
        eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setDimensions(nev=n_modes, ncv=ncv)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
        eps.setTolerances(tol=1e-12, max_it=1000)
        eps.solve()

        nconv = eps.getConverged()
        gains = np.empty(min(nconv, n_modes), dtype=float)
        vr = htrh_shell.createVecRight()
        vi = htrh_shell.createVecRight()
        for i in range(gains.size):
            eig = eps.getEigenpair(i, vr, vi)
            gains[i] = float(np.sqrt(max(float(np.real(eig)), 0.0)))

        gains = np.sort(gains)[::-1]

        eps.destroy()
        vr.destroy()
        vi.destroy()
        htrh_shell.destroy()
        product.destroy()
        return gains

    def test_gains_match(self):
        """Operator-composed gains match dense reference on a small problem."""
        rng = np.random.default_rng(600)
        n = 20
        n_modes = 5
        omega = 0.3

        L0_np = _random_invertible(n, rng)
        L0_sp = sp.csc_matrix(L0_np)
        W_diag = np.abs(rng.standard_normal(n)) + 0.1  # positive weights

        gains_ref = self._resolvent_gains_closure(L0_np, omega, W_diag, n_modes)
        gains_ops = self._resolvent_gains_composed(L0_sp, omega, W_diag, n_modes)

        # Both should agree to high precision.
        assert gains_ref.size == gains_ops.size
        rel_err = np.linalg.norm(gains_ref - gains_ops) / np.linalg.norm(gains_ref)
        assert rel_err < 1e-10, f"Gains mismatch: rel_err={rel_err:.2e}"
