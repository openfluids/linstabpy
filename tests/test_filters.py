# SPDX-License-Identifier: GPL-3.0-only
"""Tests for the filter module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from linstabpy.mesh import Mesh
from linstabpy.mesh.filters import filter_O10_diffMat, build_mesh_filters


# ---------------------------------------------------------------------------
# TestFilterConstruction
# ---------------------------------------------------------------------------

class TestFilterConstruction:
    """Tests for filter matrix construction and basic behaviour."""

    def test_filter_O10_diffMat_shapes(self):
        """M and F from filter_O10_diffMat have shape (nx, nx)."""
        nx = 40
        M, F = filter_O10_diffMat(nx, 0.4)
        assert M.shape == (nx, nx)
        assert F.shape == (nx, nx)

    def test_filter_preserves_constants(self):
        """Filtering a constant field returns the same constant field."""
        mesh = Mesh.create(
            xrange=(0, 2), yrange=(0, 1),
            nx=20, ny=30, fd_order=4, alpha_filter=0.4,
        )
        ngp = mesh.X.size
        ones = np.ones(ngp)
        result = mesh.filters.filter(ones)
        assert_allclose(result, ones, atol=1e-12)

    def test_alpha_zero_is_real_filter(self):
        """alpha_filter=0 produces a real (non-identity) explicit filter.

        MATLAB convention: alpha_filter=0 is the maximum-strength explicit
        10th-order filter (M=I). Only 'none' gives identity.
        """
        mesh = Mesh.create(
            xrange=(0, 2), yrange=(0, 1),
            nx=20, ny=30, fd_order=4, alpha_filter=0,
        )
        rng = np.random.default_rng(42)
        ngp = mesh.X.size
        v = rng.standard_normal(ngp)
        result = mesh.filters.filter(v)
        # NOT identity — the filter modifies the signal
        assert not np.allclose(result, v, atol=1e-10)
        # But preserves constants
        ones = np.ones(ngp)
        assert_allclose(mesh.filters.filter(ones), ones, atol=1e-12)

    def test_alpha_none_gives_identity(self):
        """alpha_filter='none' produces an identity filter."""
        mesh = Mesh.create(
            xrange=(0, 2), yrange=(0, 1),
            nx=20, ny=30, fd_order=4, alpha_filter="none",
        )
        rng = np.random.default_rng(99)
        ngp = mesh.X.size
        v = rng.standard_normal(ngp)
        result = mesh.filters.filter(v)
        assert_allclose(result, v, atol=1e-14)


# ---------------------------------------------------------------------------
# TestFilterAdjointProperty
# ---------------------------------------------------------------------------

class TestFilterAdjointProperty:
    """Tests for the adjoint-compliant filter."""

    @pytest.fixture()
    def mesh_with_filter(self):
        return Mesh.create(
            xrange=(0, 2), yrange=(0, 1),
            nx=20, ny=30, fd_order=4, alpha_filter=0.4,
        )

    def test_adjoint_property_single_column(self, mesh_with_filter):
        """filter_ct(u)^T @ v == u^T @ filter(v) for random real vectors."""
        mesh = mesh_with_filter
        ngp = mesh.X.size
        rng = np.random.default_rng(0)
        u = rng.standard_normal(ngp)
        v = rng.standard_normal(ngp)

        lhs = mesh.filters.filter_ct(u) @ v
        rhs = u @ mesh.filters.filter(v)
        assert_allclose(lhs, rhs, atol=1e-10)

    def test_adjoint_property_multi_column(self, mesh_with_filter):
        """Adjoint property holds for real vectors of length 5*ngp."""
        mesh = mesh_with_filter
        n = 5 * mesh.X.size
        rng = np.random.default_rng(1)
        u = rng.standard_normal(n)
        v = rng.standard_normal(n)

        lhs = mesh.filters.filter_ct(u) @ v
        rhs = u @ mesh.filters.filter(v)
        assert_allclose(lhs, rhs, atol=1e-10)

    def test_high_frequency_damped(self, mesh_with_filter):
        """A high-frequency signal is damped by the filter."""
        mesh = mesh_with_filter
        nx = mesh.X.shape[1]
        x = mesh.X.ravel(order="F")
        high_freq = np.pi * (nx - 1)  # near-Nyquist frequency
        v_high = np.sin(high_freq * x)
        v_filtered = mesh.filters.filter(v_high)
        assert np.linalg.norm(v_filtered) < np.linalg.norm(v_high)
