# SPDX-License-Identifier: GPL-3.0-only
"""Tests for base flow loading and viscosity (Sutherland's law)."""

import numpy as np
import pytest

from examples.linstab2d.matlab_baseflow import _BASEFLOW_DATA_DIR, load_blasius, load_streaky_bl
from linstabpy.baseflow import BaseFlow, sutherland_air
from linstabpy.mesh import Mesh
from tests._matlab_parity import matlab_parity, require_matlab_reference
from tests._paths import martini_fixture


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


@pytest.fixture
def blasius_baseflow_small():
    """Blasius on a 30x20 mesh, Re=1e4, Ma=0.3."""
    mesh = Mesh.create(
        xrange=(200.0, 300.0),
        yrange=(0.0, 30.0),
        nx=30,
        ny=20,
        fd_order=4,
    )
    bf = load_blasius(
        mesh.X,
        mesh.Y,
        Re=1e4,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
    )
    sutherland_air(bf)
    return mesh, bf


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


# MATLAB-export parity check. The reference data is deliberately not committed, so
# this is opt-in: deselected by default, and a FAILURE rather than a skip once
# LINSTABPY_MATLAB_PARITY=1 asks for it.
@matlab_parity
def test_streaky_bl_loader_matches_matlab_exported_base_bundle() -> None:
    # The loader needs TWO references, and the guard must cover BOTH. Checking only the
    # npz bundle meant that restoring it -- while the .mat base-flow directory stayed
    # absent -- turned a clean skip into a hard FileNotFoundError that reads like a
    # numerical parity failure. A partially satisfied guard is worse than an unsatisfied
    # one, because it hides which reference is actually missing.
    bundle_path = martini_fixture("example_3_Streaky_BL_results_base.npz")
    require_matlab_reference(
        bundle_path,
        _BASEFLOW_DATA_DIR,
        description="the Streaky-BL MATLAB parity bundle and base-flow data directory",
    )
    bundle = np.load(bundle_path, allow_pickle=False)

    class _MeshView:
        pass

    mesh = _MeshView()
    mesh.X = np.asarray(bundle["results_StreakyBL_base__mesh__X"], dtype=float)
    mesh.Y = np.asarray(bundle["results_StreakyBL_base__mesh__Y"], dtype=float)

    baseflow = load_streaky_bl(mesh)

    np.testing.assert_allclose(
        baseflow.U,
        np.asarray(bundle["results_StreakyBL_base__baseFlow__U"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        baseflow.V,
        np.asarray(bundle["results_StreakyBL_base__baseFlow__V"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        baseflow.W,
        np.asarray(bundle["results_StreakyBL_base__baseFlow__W"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
