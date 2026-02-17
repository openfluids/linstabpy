"""Tests for base flow loading and viscosity (Sutherland's law)."""

import numpy as np
import pytest

from linstab2d.baseflow import BaseFlow, load_blasius, sutherland_air
from linstab2d.mesh import Mesh


# ---------------------------------------------------------------------------
# TestSutherlandAir
# ---------------------------------------------------------------------------
class TestSutherlandAir:
    """Tests for Sutherland's law viscosity model."""

    def test_mu_positive(self, uniform_baseflow):
        bf = uniform_baseflow
        assert np.all(bf.MU > 0)

    def test_mu_monotone_increasing_with_temperature(self, small_mesh_2d):
        ny, nx = small_mesh_2d.Y.shape
        T_field = np.linspace(0.5, 2.0, ny * nx).reshape(ny, nx)
        bf = BaseFlow(
            Re=1e4,
            Ma=0.3,
            Pr=0.72,
            kappa=1.4,
            T_0=288.15,
            RHO=np.ones((ny, nx)),
            U=np.ones((ny, nx)),
            V=np.zeros((ny, nx)),
            W=np.zeros((ny, nx)),
            T=T_field,
        )
        sutherland_air(bf)
        assert np.all(bf.dmudT > 0), "dmudT should be positive (MU increases with T)"

    def test_mu_reference_at_T1(self, uniform_baseflow):
        bf = uniform_baseflow
        np.testing.assert_allclose(bf.MU, 1.0, atol=1e-12)

    def test_cv_value(self, uniform_baseflow):
        bf = uniform_baseflow
        expected = 1.0 / (bf.kappa * (bf.kappa - 1.0) * bf.Ma**2)
        np.testing.assert_allclose(bf.cv, expected, rtol=1e-12)

    def test_c1_value(self, uniform_baseflow):
        bf = uniform_baseflow
        expected = (bf.kappa - 1.0) * bf.Re * bf.Pr * bf.Ma**2
        np.testing.assert_allclose(bf.c1, expected, rtol=1e-12)

    def test_c2_value(self, uniform_baseflow):
        bf = uniform_baseflow
        expected = bf.kappa * bf.Ma**2
        np.testing.assert_allclose(bf.c2, expected, rtol=1e-12)



# ---------------------------------------------------------------------------
# TestBlasiusLoader
# ---------------------------------------------------------------------------
class TestBlasiusLoader:
    """Tests for compressible Blasius boundary-layer base flow."""

    def test_no_slip_wall_U(self, blasius_baseflow_small):
        _, bf = blasius_baseflow_small
        np.testing.assert_allclose(bf.U[0, :], 0.0, atol=1e-10)

    def test_no_slip_wall_V(self, blasius_baseflow_small):
        _, bf = blasius_baseflow_small
        np.testing.assert_allclose(bf.V[0, :], 0.0, atol=1e-10)

    def test_freestream_U(self, blasius_baseflow_small):
        _, bf = blasius_baseflow_small
        np.testing.assert_allclose(bf.U[-1, :], 1.0, atol=1e-3)

    def test_freestream_T(self, blasius_baseflow_small):
        _, bf = blasius_baseflow_small
        np.testing.assert_allclose(bf.T[-1, :], 1.0, atol=1e-3)

    def test_freestream_RHO(self, blasius_baseflow_small):
        _, bf = blasius_baseflow_small
        np.testing.assert_allclose(bf.RHO[-1, :], 1.0, atol=1e-3)

    def test_ideal_gas_law(self, blasius_baseflow_small):
        _, bf = blasius_baseflow_small
        np.testing.assert_allclose(bf.RHO * bf.T, 1.0, atol=1e-6)

    def test_W_zero(self, blasius_baseflow_small):
        _, bf = blasius_baseflow_small
        np.testing.assert_allclose(bf.W, 0.0, atol=1e-15)

    @pytest.mark.slow
    def test_continuity_residual(self, blasius_baseflow_small):
        mesh, bf = blasius_baseflow_small
        rhoU = bf.RHO * bf.U
        rhoV = bf.RHO * bf.V

        # Finite-difference approximation of partial derivatives on the
        # interior grid points (second-order central differences).
        dx = np.diff(mesh.X[0, :])
        dy = np.diff(mesh.Y[:, 0])

        drhoU_dx = np.diff(rhoU, axis=1) / dx[np.newaxis, :]
        drhoV_dy = np.diff(rhoV, axis=0) / dy[:, np.newaxis]

        # Trim to common interior grid
        ny_min = min(drhoU_dx.shape[0], drhoV_dy.shape[0])
        nx_min = min(drhoU_dx.shape[1], drhoV_dy.shape[1])
        residual = drhoU_dx[:ny_min, :nx_min] + drhoV_dy[:ny_min, :nx_min]

        assert np.max(np.abs(residual)) < 0.1, (
            f"Continuity residual too large: {np.max(np.abs(residual)):.4e}"
        )
