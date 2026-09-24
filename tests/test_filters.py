# SPDX-License-Identifier: GPL-3.0-only
"""Tests for the filter module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from linstabpy.mesh import Mesh
from linstabpy.mesh.filters import _build_2d_filter_pair, build_mesh_filters, filter_O10_diffMat


def test_mesh_package_lazy_exports_filters() -> None:
    import linstabpy.mesh as mesh_pkg

    assert mesh_pkg.build_mesh_filters is build_mesh_filters
    assert mesh_pkg.MeshFilters is not None
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(mesh_pkg, "not_a_mesh_export")


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
            xrange=(0, 2),
            yrange=(0, 1),
            nx=20,
            ny=30,
            fd_order=4,
            alpha_filter=0.4,
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
            xrange=(0, 2),
            yrange=(0, 1),
            nx=20,
            ny=30,
            fd_order=4,
            alpha_filter=0,
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
            xrange=(0, 2),
            yrange=(0, 1),
            nx=20,
            ny=30,
            fd_order=4,
            alpha_filter="none",
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
            xrange=(0, 2),
            yrange=(0, 1),
            nx=20,
            ny=30,
            fd_order=4,
            alpha_filter=0.4,
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
        # Exact discrete-transpose identity: measured residual ~1e-14, so the
        # check asserts ~1e-12 (frozen-numerics guard 1).
        assert_allclose(lhs, rhs, atol=1e-12)

    def test_adjoint_property_multi_column(self, mesh_with_filter):
        """Adjoint property holds for real vectors of length 5*ngp."""
        mesh = mesh_with_filter
        n = 5 * mesh.X.size
        rng = np.random.default_rng(1)
        u = rng.standard_normal(n)
        v = rng.standard_normal(n)

        lhs = mesh.filters.filter_ct(u) @ v
        rhs = u @ mesh.filters.filter(v)
        # Exact discrete-transpose identity; ~1e-12 closed-form tolerance.
        assert_allclose(lhs, rhs, atol=1e-12)

    def test_high_frequency_damped(self, mesh_with_filter):
        """The filter damps a high-frequency signal."""
        mesh = mesh_with_filter
        nx = mesh.X.shape[1]
        x = mesh.X.ravel(order="F")
        high_freq = np.pi * (nx - 1)  # near-Nyquist frequency
        v_high = np.sin(high_freq * x)
        v_filtered = mesh.filters.filter(v_high)
        assert np.linalg.norm(v_filtered) < np.linalg.norm(v_high)


# ---------------------------------------------------------------------------
# alpha_filter = 0 is what every shipped case runs: M = I, no implicit solve
# ---------------------------------------------------------------------------


def test_adjoint_property_alpha_zero():
    """filter_ct is the exact discrete transpose of filter at alpha_filter=0 (M = I)."""
    mesh = Mesh.create(xrange=(0, 2), yrange=(0, 1), nx=20, ny=30, fd_order=4, alpha_filter=0)
    n = 5 * mesh.X.size
    rng = np.random.default_rng(3)
    u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    lhs = np.vdot(mesh.filters.filter_ct(u), v)
    rhs = np.vdot(u, mesh.filters.filter(v))
    assert_allclose(lhs, rhs, atol=1e-12)


def test_building_and_applying_filters_needs_no_petsc():
    """The implicit filter solve is scipy (SuperLU, or nothing at alpha=0), never PETSc/MUMPS.

    2026-08-18.
    One PETSc/MUMPS solve per column per filter call was 62 % of a
    resolvent H^H H apply.
    That was on the Savarino case.
    Run in a fresh interpreter so petsc4py imported by other tests cannot mask
    a regression.
    """
    import subprocess
    import sys

    code = (
        "import sys, numpy as np\n"
        "from linstabpy.mesh import Mesh\n"
        "for alpha in (0, 0.4):\n"
        "    m = Mesh.create(xrange=(0, 2), yrange=(0, 1), nx=20, ny=30, fd_order=4, alpha_filter=alpha)\n"
        "    m.filters.filter_ct(m.filters.filter(np.ones(5 * m.X.size)))\n"
        "assert 'petsc4py' not in sys.modules, 'filters pulled in petsc4py'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")


def test_filter_o10_rejects_a_single_point_line() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        filter_O10_diffMat(1, 0.4)


def test_mesh_filters_reject_negative_alpha() -> None:
    mesh = Mesh.create(xrange=(0, 1), yrange=(0, 1), nx=12, ny=10, fd_order=4)
    with pytest.raises(ValueError, match="non-negative"):
        build_mesh_filters(mesh.X, mesh.Y, mesh.used_ind, alpha_filter=-0.1)


def test_mesh_filter_accepts_row_or_column_layout() -> None:
    mesh = Mesh.create(xrange=(0, 2), yrange=(0, 1), nx=16, ny=12, fd_order=4, alpha_filter=0.4)
    n = mesh.ngp
    rng = np.random.default_rng(4)
    cols = rng.standard_normal((n, 3))
    out_cols = mesh.filters.filter(cols)
    assert out_cols.shape == (n, 3)
    rows = cols.T
    out_rows = mesh.filters.filter(rows)
    np.testing.assert_allclose(out_rows, out_cols.ravel(order="F"), atol=1e-12)
    with pytest.raises(ValueError, match="n or n\\*k"):
        mesh.filters.filter(np.ones(n + 1))
    with pytest.raises(ValueError, match="n rows"):
        mesh.filters.filter(np.ones((3, 3)))


def test_build_2d_filter_pair_handles_empty_and_short_lines() -> None:
    with pytest.raises(ValueError, match="same length"):
        _build_2d_filter_pair(np.ones(3), np.ones(2), 0.4)
    m0, f0 = _build_2d_filter_pair(np.zeros(0), np.zeros(0), 0.4)
    assert m0.shape == (0, 0)
    assert f0.shape == (0, 0)
    primary = np.array([0.0, 1.0, 2.0])
    secondary = np.array([0.0, 0.0, 1.0])
    m, f = _build_2d_filter_pair(primary, secondary, 0.4)
    assert m.shape == (3, 3)
    assert f.shape == (3, 3)
    ident_m, ident_f = _build_2d_filter_pair(np.array([0.0]), np.array([0.0]), 0.4)
    np.testing.assert_array_equal(ident_m.toarray(), np.eye(1))
    np.testing.assert_array_equal(ident_f.toarray(), np.eye(1))
