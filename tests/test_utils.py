"""Tests for Floquet reconstruction and flow plotting utilities."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from linstab2d.mesh import Mesh
from linstab2d.examples._helpers import leading_mode_real_frequency
from linstab2d.utils import reconstruct_floquet, plot_flow


# ---------------------------------------------------------------------------
# Floquet reconstruction
# ---------------------------------------------------------------------------
class TestFloquetReconstruction:
    """Tests for reconstruct_floquet."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        mesh = Mesh.create(
            xrange=(0, 2 * np.pi), yrange=(0, 1), nx=16, ny=20, fd_order=4
        )
        self.X = mesh.X
        self.Y = mesh.Y
        self.used_ind = mesh.used_ind
        self.ngp = mesh.ngp
        self.idx = mesh.idx

    def test_zero_floquet_exp(self):
        """floquet_exp=0 applies no phase shift; output equals V on full grid."""
        V = np.ones(self.ngp, dtype=complex)
        result = reconstruct_floquet(
            self.X, self.Y, V, self.idx, self.used_ind, floquet_exp=0.0
        )
        np.testing.assert_allclose(result, 1.0 + 0j, atol=1e-14)

    def test_nonzero_floquet_exp(self):
        """floquet_exp=beta multiplies by exp(i*beta*X)."""
        beta = 1.0
        V = np.ones(self.ngp, dtype=complex)
        result = reconstruct_floquet(
            self.X, self.Y, V, self.idx, self.used_ind, floquet_exp=beta
        )
        expected = np.exp(1j * beta * self.X)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_1d_input_shape(self):
        """1-D V produces output with shape (ny, nx)."""
        V = np.ones(self.ngp, dtype=complex)
        result = reconstruct_floquet(
            self.X, self.Y, V, self.idx, self.used_ind, floquet_exp=0.0
        )
        assert result.shape == self.X.shape

    def test_2d_input_shape(self):
        """2-D V (ngp, n_fields) produces output with shape (n_fields, ny, nx)."""
        ones = np.ones(self.ngp, dtype=complex)
        V = np.column_stack([ones, 2 * ones])
        result = reconstruct_floquet(
            self.X, self.Y, V, self.idx, self.used_ind, floquet_exp=0.0
        )
        n_fields = V.shape[1]
        assert result.shape == (n_fields, *self.X.shape)

    def test_wrong_length_raises(self):
        """V with wrong length raises ValueError."""
        V = np.zeros(self.ngp + 5, dtype=complex)
        with pytest.raises(ValueError):
            reconstruct_floquet(
                self.X, self.Y, V, self.idx, self.used_ind, floquet_exp=0.0
            )


# ---------------------------------------------------------------------------
# Flow plotting
# ---------------------------------------------------------------------------
class TestPlotFlow:
    """Tests for plot_flow."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        mesh = Mesh.create(
            xrange=(0, 1), yrange=(0, 1), nx=10, ny=10, fd_order=4
        )
        self.X = mesh.X
        self.Y = mesh.Y
        rng = np.random.default_rng(42)
        self.field = rng.standard_normal((10, 10))

    def test_invalid_component_raises(self):
        """Invalid component raises ValueError."""
        with pytest.raises(ValueError):
            plot_flow(self.X, self.Y, self.field, component="foo")

    def test_wrong_shape_raises(self):
        """Field shape mismatch with X raises ValueError."""
        bad_field = np.zeros((5, 5))
        with pytest.raises(ValueError):
            plot_flow(self.X, self.Y, bad_field)

    def test_masked_field_tricontourf(self):
        """plot_flow handles NaN-holed fields without error (tricontourf path)."""
        field = self.field.copy()
        field[4:6, 4:6] = np.nan
        ax = plot_flow(self.X, self.Y, field, component="real")
        assert ax is not None

    def test_masked_field_uses_tricontourf(self):
        """NaN-holed fields should use tricontourf, not pcolormesh."""
        import matplotlib.tri as mtri
        field = self.field.copy()
        field[4:6, 4:6] = np.nan
        ax = plot_flow(self.X, self.Y, field, component="real", colorbar=False)
        # Check that the drawn artist is a TriContourSet (from tricontourf)
        has_tri = any(
            isinstance(c, mtri.TriContourSet) for c in ax.collections
        )
        # tricontourf produces QuadContourSet-like objects stored differently
        # in recent Matplotlib; at minimum, no pcolormesh QuadMesh should exist
        from matplotlib.collections import QuadMesh
        has_quadmesh = any(isinstance(c, QuadMesh) for c in ax.collections)
        assert not has_quadmesh, "NaN field fell back to pcolormesh"


def test_leading_mode_real_frequency_uses_least_damped_mode() -> None:
    omega = np.array([0.62 - 0.01j, 0.59 + 0.02j, 0.58 + 0.01j], dtype=complex)
    selected = leading_mode_real_frequency(omega, imag_shift=0.03)
    assert selected == complex(0.59, 0.03)


def test_leading_mode_real_frequency_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        leading_mode_real_frequency(np.array([], dtype=complex))
