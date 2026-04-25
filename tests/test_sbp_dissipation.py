"""Tests for the SBP artificial dissipation operator (MSN 2004).

Gates T2 (SPSD), T3 (sawtooth damping), and T4 (smooth-mode bound) from
phase1_execution.md, verified for fd4 and fd6 interior orders.
"""
from __future__ import annotations

import numpy as np
import pytest

from linstab2d.mesh.sbp import build_sbp_dissipation_matrices

_N = 51
_H = 1.0 / (_N - 1)


class TestSBPDissipationNaN:
    """T2-tight: no NaN/inf, H diagonal strictly positive."""

    @pytest.mark.parametrize("order_interior", [4, 6], ids=["fd4", "fd6"])
    def test_no_nan_inf(self, order_interior: int) -> None:
        Q_diss, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )

        assert np.all(np.isfinite(Q_diss.data)), "Q_diss contains NaN/Inf"
        assert np.all(np.isfinite(H_norm.data)), "H_norm contains NaN/Inf"

    @pytest.mark.parametrize("order_interior", [4, 6], ids=["fd4", "fd6"])
    def test_h_positive(self, order_interior: int) -> None:
        _, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )

        assert np.all(H_norm.diagonal() > 0), "H_norm diagonal has non-positive entries"


class TestSBPDissipationSPSD:
    """T2: H*Q_diss is symmetric positive semidefinite."""

    @pytest.mark.parametrize("order_interior", [4, 6], ids=["fd4", "fd6"])
    def test_symmetry(self, order_interior: int) -> None:
        Q_diss, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )
        M = (H_norm @ Q_diss).toarray()
        sym_err = np.max(np.abs(M - M.T))

        assert sym_err < 1e-10, f"H*Q_diss not symmetric: max|M - M.T| = {sym_err:.2e}"

    @pytest.mark.parametrize("order_interior", [4, 6], ids=["fd4", "fd6"])
    def test_positive_semidefinite(self, order_interior: int) -> None:
        Q_diss, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )
        M = (H_norm @ Q_diss).toarray()
        eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
        min_ev = eigvals.min()

        assert min_ev > -1e-10, f"H*Q_diss not PSD: min eigenvalue = {min_ev:.2e}"


class TestSBPDissipationSawtoothDamping:
    """T3: sawtooth probe is damped at O(1) rate, independent of h."""

    @pytest.mark.parametrize("order_interior,c_T3", [(4, 1e-2), (6, 1e-3)])
    def test_sawtooth_damping(self, order_interior: int, c_T3: float) -> None:
        """c_T3 chosen conservatively; analytical rate is O(1) from MSN 2004 §3 sawtooth Fourier limit."""
        Q_diss, _ = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )
        j = np.arange(_N)
        v = (-1.0) ** j
        r = float(v @ (Q_diss @ v)) / float(v @ v)

        assert r >= c_T3, (
            f"Sawtooth damping rate {r:.4e} below threshold {c_T3} "
            f"for order {order_interior}"
        )


class TestSBPDissipationSmoothBound:
    """T4: smooth probe damped at O(h^{2p-1}); MSN 2004 §3 k->0 Fourier limit."""

    @pytest.mark.parametrize(
        "order_interior,c_T4",
        [(4, 1e6), (6, 1e10)],
        ids=["fd4", "fd6"],
    )
    def test_smooth_mode_bound(self, order_interior: int, c_T4: float) -> None:
        p = order_interior // 2
        Q_diss, _ = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )
        x = np.arange(_N) * _H
        v = np.sin(2.0 * np.pi * x)
        damp = np.linalg.norm(Q_diss @ v) / np.linalg.norm(v)
        bound = c_T4 * _H ** (2 * p - 1)

        assert damp <= bound, (
            f"Smooth-mode damping {damp:.4e} exceeds bound {bound:.4e} "
            f"(c_T4={c_T4}, h^{2*p-1}={_H ** (2 * p - 1):.4e}) "
            f"for order {order_interior}"
        )
