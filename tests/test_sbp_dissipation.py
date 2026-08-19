# SPDX-License-Identifier: GPL-3.0-only
"""Check SBP artificial dissipation (MSN 2004) T2, T3, and T4.

Gates T2 (SPSD), T3 (sawtooth damping), and T4 (smooth-mode bound) from
phase1_execution.md, for fd4, fd6, and fd8 interior orders.

T3 and T4 are dimensionless predicates so a future scaling regression
(e.g. an h^{2p+1} prefactor that is not h^{2p+1} in
build_sbp_dissipation_matrices) is caught at unit-test time.
The original Phase 1 step 7 integration test caught a factor-of-h^{-2}
error that the previous loose T3/T4 gates did not.
These tighter forms exist so that error does not return.
"""

from __future__ import annotations

import numpy as np
import pytest

from linstabpy.mesh.sbp import (
    _build_dp_matrix,
    _build_sbp_with_pole,
    _mohseni_pole_stencil,
    build_sbp_dissipation_matrices,
    build_sbp_pole_matrices,
    sbp_stencils,
)

_N = 51
_H = 1.0 / (_N - 1)


class TestSBPDissipationNaN:
    """T2-tight: no NaN/inf, H diagonal strictly positive."""

    @pytest.mark.parametrize("order_interior", [4, 6, 8], ids=["fd4", "fd6", "fd8"])
    def test_no_nan_inf(self, order_interior: int) -> None:
        Q_diss, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )

        assert np.all(np.isfinite(Q_diss.data)), "Q_diss contains NaN/Inf"
        assert np.all(np.isfinite(H_norm.data)), "H_norm contains NaN/Inf"

    @pytest.mark.parametrize("order_interior", [4, 6, 8], ids=["fd4", "fd6", "fd8"])
    def test_h_positive(self, order_interior: int) -> None:
        _, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )

        assert np.all(H_norm.diagonal() > 0), "H_norm diagonal has non-positive entries"


class TestSBPDissipationSPSD:
    """T2: H*Q_diss is symmetric positive semidefinite."""

    @pytest.mark.parametrize("order_interior", [4, 6, 8], ids=["fd4", "fd6", "fd8"])
    def test_symmetry(self, order_interior: int) -> None:
        Q_diss, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )
        M = (H_norm @ Q_diss).toarray()
        sym_err = np.max(np.abs(M - M.T))

        assert sym_err < 1e-10, f"H*Q_diss not symmetric: max|M - M.T| = {sym_err:.2e}"

    @pytest.mark.parametrize("order_interior", [4, 6, 8], ids=["fd4", "fd6", "fd8"])
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
    """T3: dimensionless sawtooth eigenvalue is finite and O(1) — not exploding.

    For even p (fd4/fd8, p=2/4), the centered p-th derivative stencil applied
    to v[j] = (-1)^j gives a finite per-entry response.
    The H-norm Rayleigh quotient v^T H Q_diss v / (v^T H v) remains O(1).
    Higher-order centered stencils can have larger Nyquist symbols than the
    compact repeated second-difference estimate.

    For odd p (fd6, p=3), the centered (2p+1)-th order stencil is
    anti-symmetric.
    It annihilates the sawtooth at every interior row by parity.
    The Rayleigh quotient then comes from the boundary closure only.
    That is the design intent: the saw-tooth artefact in the curved-wall
    LNSE problem is also boundary-localised.
    A boundary-only damping mechanism then acts at that same place.

    The bounds below cover both regimes.
    They still catch the historical h^{-2} scaling error.
    That error would push the quotient by ~h^-2 = 2500x at _N=51,
    far above the upper bound.
    """

    @pytest.mark.parametrize(
        "order_interior, lo, hi",
        [
            (4, 5.0, 100.0),  # ~27 vs ~67500
            (6, 1.0, 50.0),  # ~6.4 vs ~16000
            (8, 500.0, 1600.0),  # ~1028 vs ~2.6e6
        ],
        ids=["fd4", "fd6", "fd8"],
    )
    def test_sawtooth_dimensionless(self, order_interior: int, lo: float, hi: float) -> None:
        Q_diss, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )
        j = np.arange(_N)
        v = ((-1.0) ** j).astype(float)

        rayleigh_H = float(v @ (H_norm @ (Q_diss @ v))) / float(v @ (H_norm @ v))

        assert lo <= rayleigh_H <= hi, (
            f"Sawtooth H-Rayleigh {rayleigh_H:.3f} outside band [{lo}, {hi}] "
            f"for fd{order_interior}; possible scaling regression."
        )


class TestSBPDissipationSmoothBound:
    """T4: smooth-mode damping is small and not exploding.

    A smooth probe v(x) = sin(2 pi x) gives `||Q_diss v|| / ||v||`.
    That scales as O(h^{2p}) for resolved Fourier content.
    The constant absorbs the (2 pi)^{2p} amplitude of the analytical
    p-th derivative plus a boundary-closure factor.
    A loose c covers the aliased boundary stencil.
    It still catches an h^{-2} scaling error (damp up 2500x at _N = 51).
    """

    @pytest.mark.parametrize(
        "order_interior, c",
        [
            (4, 1.0e5),  # ~1.3e-3 vs ~3.2
            (6, 1.0e9),  # ~9.4e-3 vs ~24
            (8, 1.0e13),
        ],
        ids=["fd4", "fd6", "fd8"],
    )
    def test_smooth_mode_bound(self, order_interior: int, c: float) -> None:
        p = order_interior // 2
        Q_diss, _ = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )
        x = np.arange(_N) * _H
        v = np.sin(2.0 * np.pi * x)
        damp = float(np.linalg.norm(Q_diss @ v)) / float(np.linalg.norm(v))
        bound = c * _H ** (2 * p)

        assert damp <= bound, (
            f"Smooth-mode damping {damp:.4e} exceeds bound {bound:.4e} "
            f"(c={c:.1e}, h^{{2p}}={_H ** (2 * p):.3e}) for fd{order_interior}; "
            f"possible scaling regression."
        )


class TestSBPDissipationScalingRegression:
    """Detect a future regression in the h^{2p+1} divided-form prefactor.

    If someone re-introduces an h^{2p-1} or h^{2p+3} prefactor in
    build_sbp_dissipation_matrices, the diagonal of H*Q_diss should drift
    by orders of magnitude.
    Pin a representative diagonal entry against the analytical scale.
    A precise number would be fragile to legitimate stencil tweaks.

    H*Q_diss = h^{2p+1} * D_p^T B_p D_p, with D_p having O(1/h^p) entries
    and B_p O(1) interior.
    A diagonal entry of H*Q_diss in the interior then has scale
    O(h * bw) from h^{2p+1} * (1/h^p)^2.
    That scale is independent of p.
    Check the order of magnitude.
    """

    @pytest.mark.parametrize(
        "order_interior, hi_factor",
        [
            (4, 100.0),
            (6, 100.0),
            (8, 500.0),  # 9-point D4 has sum(c_i^2) ~= 278.
        ],
        ids=["fd4", "fd6", "fd8"],
    )
    def test_diagonal_scaling(self, order_interior: int, hi_factor: float) -> None:
        Q_diss, H_norm = build_sbp_dissipation_matrices(
            _N,
            _H,
            order_interior=order_interior,
        )
        HQ = (H_norm @ Q_diss).toarray()
        # Pick an interior row well clear of boundary closure.
        center = _N // 2
        diag_center = HQ[center, center]
        # Expected magnitude: h * O(1) * (interior B_p ~ 1) = O(h).
        # Assert it lies in an O(h) band — the upper factor accommodates the
        # larger fd8 D4 stencil norm but still catches the historical h^{-2}
        # prefactor (diagonal ~1/h instead of ~h, four orders of
        # magnitude off at _H = 1/50).
        lo, hi = 0.01 * _H, hi_factor * _H
        assert lo <= diag_center <= hi, (
            f"H*Q_diss[{center},{center}] = {diag_center:.4e} outside expected "
            f"O(h) band [{lo:.2e}, {hi:.2e}] for fd{order_interior}; "
            f"likely scaling regression in build_sbp_dissipation_matrices."
        )


def test_dissipation_rejects_unknown_and_decoupled_orders() -> None:
    with pytest.raises(ValueError, match="order_interior"):
        build_sbp_dissipation_matrices(_N, _H, order_interior=2)
    with pytest.raises(NotImplementedError, match="order_diss"):
        build_sbp_dissipation_matrices(_N, _H, order_interior=4, order_diss=6)


def test_dp_matrix_rejects_unsupported_derivative_order() -> None:
    with pytest.raises(ValueError, match="Unsupported p"):
        _build_dp_matrix(20, 0.1, 5)


def test_pole_stencil_rejects_a_bad_parity() -> None:
    a = sbp_stencils(2)["a_i1"]
    with pytest.raises(ValueError, match="pole parity"):
        _mohseni_pole_stencil(a, 0)


def test_pole_builder_rejects_a_malformed_interior_stencil(monkeypatch) -> None:
    real_stencils = sbp_stencils

    def _bad(_order: int):
        tables = dict(real_stencils(2))
        tables["a_i1"] = np.ones((2, 3))
        return tables

    monkeypatch.setattr("linstabpy.mesh.sbp.sbp_stencils", _bad)
    with pytest.raises(ValueError, match="interior stencil"):
        _build_sbp_with_pole(12, 0.1, 2)


def test_pole_matrices_have_matching_shapes() -> None:
    D1s, D1a, D2s, D2a = build_sbp_pole_matrices(16, 0.1, 2)
    assert D1s.shape == (16, 16)
    assert D1a.shape == D2s.shape == D2a.shape == (16, 16)
