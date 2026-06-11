# SPDX-License-Identifier: GPL-3.0-only

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from linstabpy.resolvent import ResolventOperator, WeightedNormalOperator


class _DummyFactorization:
    def __init__(self, solve_mat: np.ndarray) -> None:
        self.solve_mat = solve_mat

    def solve(self, vector: np.ndarray) -> np.ndarray:
        return self.solve_mat @ vector

    def solve_transpose(self, vector: np.ndarray) -> np.ndarray:
        return self.solve_mat.conj().T @ vector


def test_resolvent_operator_import() -> None:
    from linstabpy.resolvent import ResolventOperator as imported

    assert imported is ResolventOperator


def test_weighted_normal_operator_import() -> None:
    from linstabpy.resolvent import WeightedNormalOperator as imported

    assert imported is WeightedNormalOperator


def test_resolvent_operator_matches_inline_chain_with_b_c() -> None:
    rng = np.random.default_rng(12345)
    n = 5
    solve_mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b_dense = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    c_dense = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    B_mat = sp.csr_matrix(b_dense)
    C_mat = sp.csr_matrix(c_dense)
    filt_mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    filt_ct_mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    lu = _DummyFactorization(solve_mat)

    def filt(vector: np.ndarray) -> np.ndarray:
        return filt_mat @ vector

    def filt_ct(vector: np.ndarray) -> np.ndarray:
        return filt_ct_mat @ vector

    op = ResolventOperator(lu, B_mat, C_mat, filt, filt_ct)

    w = filt(v)
    w = B_mat @ w
    w = lu.solve(w)
    w = C_mat @ w
    w = filt(w)
    np.testing.assert_array_equal(op.apply(v), w)

    w_adj = filt_ct(v)
    w_adj = C_mat.conj().T @ w_adj
    w_adj = lu.solve_transpose(w_adj)
    w_adj = B_mat.conj().T @ w_adj
    w_adj = filt_ct(w_adj)
    np.testing.assert_array_equal(op.apply_adjoint(v), w_adj)


def test_resolvent_operator_matches_inline_chain_without_b_c_or_filters() -> None:
    rng = np.random.default_rng(54321)
    n = 5
    solve_mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    lu = _DummyFactorization(solve_mat)
    op = ResolventOperator(lu, None, None, None, None)

    w = v
    w = lu.solve(w)
    np.testing.assert_array_equal(op.apply(v), w)

    w_adj = v
    w_adj = lu.solve_transpose(w_adj)
    np.testing.assert_array_equal(op.apply_adjoint(v), w_adj)


def test_weighted_normal_operator_matches_inline_weighted_chain() -> None:
    from petsc4py import PETSc

    rng = np.random.default_rng(24680)
    n = 5
    solve_mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    W_diag = 0.25 + rng.random(n)
    lu = _DummyFactorization(solve_mat)
    op = ResolventOperator(lu, None, None, None, None)
    weighted_op = WeightedNormalOperator(op, W_diag, n)

    x = PETSc.Vec().createSeq(n)
    y = PETSc.Vec().createSeq(n)
    try:
        x.setArray(v)
        weighted_op.mult(None, x, y)
        expected = (1.0 / W_diag) * op.apply_adjoint(W_diag * op.apply(v))
        np.testing.assert_array_equal(y.getArray(readonly=True), expected)
    finally:
        x.destroy()
        y.destroy()
