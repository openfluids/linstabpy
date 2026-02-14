"""Tests for Floquet reconstruction and flow plotting utilities."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from linstab2d.mesh import Mesh
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

    def test_real_component(self):
        """component='real' completes without error."""
        ax = plot_flow(self.X, self.Y, self.field, component="real")
        assert ax is not None

    def test_abs_component(self):
        """component='abs' completes without error."""
        ax = plot_flow(self.X, self.Y, self.field, component="abs")
        assert ax is not None

    def test_invalid_component_raises(self):
        """Invalid component raises ValueError."""
        with pytest.raises(ValueError):
            plot_flow(self.X, self.Y, self.field, component="foo")

    def test_wrong_shape_raises(self):
        """Field shape mismatch with X raises ValueError."""
        bad_field = np.zeros((5, 5))
        with pytest.raises(ValueError):
            plot_flow(self.X, self.Y, bad_field)
