"""Tests for the automatic SBP stencil generator.

Validates that generated stencils match hardcoded reference tables and
satisfy the mathematical properties required for SBP operators.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstab2d.mesh.sbp import sbp_stencils, build_sbp_matrix
from linstab2d.mesh.sbp_generator import (
    fornberg_interior,
    generate_sbp_stencils,
    _sbp_norm_weights,
)


# ---------------------------------------------------------------------------
# Phase 2a: Fornberg interior stencils match hardcoded tables
# ---------------------------------------------------------------------------

class TestFornbergInterior:
    """Interior stencils generated via Fornberg must match hardcoded values."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d1_interior_matches_reference(self, order: int) -> None:
        ref = sbp_stencils(order)["a_i1"]
        gen = fornberg_interior(order, 1)
        np.testing.assert_allclose(gen, ref, atol=1e-14)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d2_interior_matches_reference(self, order: int) -> None:
        ref = sbp_stencils(order)["a_i2"]
        gen = fornberg_interior(order, 2)
        np.testing.assert_allclose(gen, ref, atol=1e-14)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d1_antisymmetric(self, order: int) -> None:
        """D1 central stencil is antisymmetric: a_i1[w-k] = -a_i1[w+k]."""
        a = fornberg_interior(order, 1)
        w = order // 2
        for k in range(1, w + 1):
            assert abs(a[w - k] + a[w + k]) < 1e-14

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d2_symmetric(self, order: int) -> None:
        """D2 central stencil is symmetric: a_i2[w-k] = a_i2[w+k]."""
        a = fornberg_interior(order, 2)
        w = order // 2
        for k in range(1, w + 1):
            assert abs(a[w - k] - a[w + k]) < 1e-14

    def test_d1_weights_sum_to_zero(self) -> None:
        for order in [2, 4, 6, 8, 10, 12]:
            a = fornberg_interior(order, 1)
            assert abs(np.sum(a)) < 1e-14, f"D1 weights for order {order} don't sum to 0"

    def test_d2_weights_sum_to_zero(self) -> None:
        for order in [2, 4, 6, 8, 10, 12]:
            a = fornberg_interior(order, 2)
            assert abs(np.sum(a)) < 1e-14, f"D2 weights for order {order} don't sum to 0"

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError):
            fornberg_interior(3, 1)
        with pytest.raises(ValueError):
            fornberg_interior(0, 1)

    def test_invalid_deriv_raises(self) -> None:
        with pytest.raises(ValueError):
            fornberg_interior(4, 3)


# ---------------------------------------------------------------------------
# Phase 2b: Full SBP stencils (interior + boundary) match hardcoded tables
# ---------------------------------------------------------------------------

class TestSBPGeneratorMatchesReference:
    """Generated boundary closures must match hardcoded reference to ~machine eps."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_a_b1_matches_reference(self, order: int) -> None:
        ref = sbp_stencils(order)["a_b1"]
        gen = generate_sbp_stencils(order)["a_b1"]
        assert gen.shape == ref.shape, f"Shape mismatch: {gen.shape} vs {ref.shape}"
        np.testing.assert_allclose(gen, ref, atol=1e-10, err_msg=f"a_b1 order {order}")

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_a_b2_matches_reference(self, order: int) -> None:
        ref = sbp_stencils(order)["a_b2"]
        gen = generate_sbp_stencils(order)["a_b2"]
        assert gen.shape == ref.shape, f"Shape mismatch: {gen.shape} vs {ref.shape}"
        np.testing.assert_allclose(gen, ref, atol=1e-10, err_msg=f"a_b2 order {order}")


# ---------------------------------------------------------------------------
# SBP property tests
# ---------------------------------------------------------------------------

class TestSBPProperty:
    """Verify the SBP property Q + Q^T = B for generated operators."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_sbp_property_d1(self, order: int) -> None:
        """Q + Q^T = B where Q = H D1 and B = diag(-1, 0, ..., 0, 1)."""
        n = max(50, 4 * order)
        dz = 1.0
        D1, _ = build_sbp_matrix(n, dz, order)
        h = _sbp_norm_weights(order)

        # Build H (diagonal norm)
        H_diag = np.ones(n)
        r = len(h)
        H_diag[:r] = h
        H_diag[n - r:] = h[::-1]
        H = sp.diags(H_diag) * dz

        Q = H @ D1 * dz  # undo the 1/dz in D1
        Q_dense = Q.toarray()

        B = np.zeros((n, n))
        B[0, 0] = -1.0
        B[n - 1, n - 1] = 1.0

        residual = Q_dense + Q_dense.T - B
        assert np.max(np.abs(residual)) < 1e-12, (
            f"SBP property violated for order {order}: max residual = {np.max(np.abs(residual)):.2e}"
        )


# ---------------------------------------------------------------------------
# Derivative accuracy tests
# ---------------------------------------------------------------------------

class TestDerivativeAccuracy:
    """D1 and D2 applied to polynomials of appropriate degree must be exact."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d1_exact_on_polynomial(self, order: int) -> None:
        """D1 applied to x^p (p = order) should be exact at interior points."""
        n = max(50, 4 * order)
        dz = 0.1
        D1, _ = build_sbp_matrix(n, dz, order)

        x = np.arange(n) * dz
        # Test with polynomial of degree order (should be exact for interior)
        p = order
        f = x ** p
        df_exact = p * x ** (p - 1)
        df_num = D1 @ f

        # Check interior points only (boundary accuracy is p/2)
        w = order // 2
        r = 1 if order == 2 else order
        interior = slice(r, n - r)
        np.testing.assert_allclose(
            df_num[interior], df_exact[interior], atol=1e-10,
            err_msg=f"D1 interior accuracy failed for order {order}"
        )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d2_exact_on_polynomial(self, order: int) -> None:
        """D2 applied to x^p should be exact at interior points."""
        n = max(50, 4 * order)
        dz = 0.1
        _, D2 = build_sbp_matrix(n, dz, order)

        x = np.arange(n) * dz
        p = order
        f = x ** p
        d2f_exact = p * (p - 1) * x ** (p - 2)
        d2f_num = D2 @ f

        r = 1 if order == 2 else order
        interior = slice(r, n - r)
        np.testing.assert_allclose(
            d2f_num[interior], d2f_exact[interior], atol=1e-8,
            err_msg=f"D2 interior accuracy failed for order {order}"
        )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d1_boundary_accuracy(self, order: int) -> None:
        """D1 boundary closure has accuracy p = order/2."""
        n = max(50, 4 * order)
        dz = 0.1
        D1, _ = build_sbp_matrix(n, dz, order)

        x = np.arange(n) * dz
        p = order // 2  # boundary accuracy
        f = x ** p
        df_exact = p * x ** (p - 1) if p > 0 else np.zeros(n)
        df_num = D1 @ f

        r = 1 if order == 2 else order
        np.testing.assert_allclose(
            df_num[:r], df_exact[:r], atol=1e-10,
            err_msg=f"D1 boundary accuracy failed for order {order}"
        )


# ---------------------------------------------------------------------------
# Higher-order generation (orders > 8)
# ---------------------------------------------------------------------------

class TestHigherOrders:
    """Verify that orders 10+ can be generated and produce valid operators."""

    @pytest.mark.parametrize("order", [10, 12])
    def test_generates_without_error(self, order: int) -> None:
        s = generate_sbp_stencils(order)
        assert s["a_i1"].shape == (order + 1,)
        assert s["a_i2"].shape == (order + 1,)
        assert s["a_b1"].shape[0] == order  # r = order
        assert s["a_b1"].shape[1] == 3 * order // 2  # m = 3*order/2

    @pytest.mark.parametrize("order", [10, 12])
    def test_d1_interior_exact_high_order(self, order: int) -> None:
        """Higher-order interior stencil differentiates x^order exactly."""
        n = 4 * order
        dz = 0.1
        a_i1 = generate_sbp_stencils(order)["a_i1"]
        w = order // 2
        x = np.arange(n) * dz
        f = x ** order
        df_exact = order * x ** (order - 1)

        # Apply interior stencil at midpoint
        mid = n // 2
        df_num = np.dot(a_i1, f[mid - w:mid + w + 1]) / dz
        np.testing.assert_allclose(df_num, df_exact[mid], rtol=1e-8)
