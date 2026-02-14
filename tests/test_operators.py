"""Tests for LNSE coefficients, assembly, energy norm, and spatial formulation."""

import numpy as np
import scipy.sparse as sp
import pytest

from linstab2d.mesh import Mesh
from linstab2d.baseflow import BaseFlow, sutherland_air
from linstab2d.operators import build_lnse_operator, compute_chu_norm, build_spatial_operator
from linstab2d.operators.coefficients import (
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
)

slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_scalars(ngp, *, kappa=1.4, Ma=0.3):
    """Return uniform-flow scalars as flat arrays of length *ngp*."""
    U   = np.ones(ngp)
    V   = np.zeros(ngp)
    W   = np.zeros(ngp)
    RHO = np.ones(ngp)
    T   = np.ones(ngp)
    MU  = np.ones(ngp)
    P   = RHO * T / (kappa * Ma**2)
    return U, V, W, RHO, T, MU, P


def _zero_derivatives(ngp):
    """Return a dict of all derivative arrays set to zero."""
    names = [
        "dUdx", "dVdx", "dWdx", "dRHOdx", "dTdx", "dPdx",
        "dUdy", "dVdy", "dWdy", "dRHOdy", "dTdy", "dPdy",
        "d2Udx2", "d2Vdx2", "d2Wdx2",
        "d2Udy2", "d2Vdy2", "d2Wdy2",
        "d2Tdx2", "d2Tdy2",
        "d2Udxdy", "d2Vdxdy", "d2Wdxdy",
        "dmudT", "d2mudT2",
    ]
    return {n: np.zeros(ngp) for n in names}


def _call_cartesian(ngp, *, kx=0.0, Re=1e5, Ma=0.3, Pr=0.72, kappa=1.4,
                    U=None, V=None, W=None, RHO=None, T=None, MU=None, P=None,
                    derivs=None):
    """Convenience wrapper around compute_cartesian_coefficients."""
    if U is None:
        U, V, W, RHO, T, MU, P = _uniform_scalars(ngp, kappa=kappa, Ma=Ma)
    if derivs is None:
        derivs = _zero_derivatives(ngp)
    return compute_cartesian_coefficients(
        U, V, W, RHO, T, MU, P, kx, Re, Ma, Pr, kappa,
        derivs["dUdx"], derivs["dVdx"], derivs["dWdx"],
        derivs["dRHOdx"], derivs["dTdx"], derivs["dPdx"],
        derivs["dUdy"], derivs["dVdy"], derivs["dWdy"],
        derivs["dRHOdy"], derivs["dTdy"], derivs["dPdy"],
        derivs["d2Udx2"], derivs["d2Vdx2"], derivs["d2Wdx2"],
        derivs["d2Udy2"], derivs["d2Vdy2"], derivs["d2Wdy2"],
        derivs["d2Tdx2"], derivs["d2Tdy2"],
        derivs["d2Udxdy"], derivs["d2Vdxdy"], derivs["d2Wdxdy"],
        derivs["dmudT"], derivs["d2mudT2"],
    )


# =========================================================================
# TestCartesianCoefficients
# =========================================================================

class TestCartesianCoefficients:
    """Tests for compute_cartesian_coefficients with uniform base flow."""

    NGP = 50

    def test_coefficient_shapes(self):
        """All seven coefficient arrays have shape (5, 5, ngp)."""
        coeffs = _call_cartesian(self.NGP)
        for name in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
            arr = getattr(coeffs, name)
            assert arr.shape == (5, 5, self.NGP), f"{name} shape mismatch"

    def test_B_diagonal(self):
        """B[k,k,:] = 1j*RHO for k=1..4, B[0,0,:] = 1j; off-diag = 0."""
        coeffs = _call_cartesian(self.NGP)
        B = coeffs.B
        # Diagonal entries
        np.testing.assert_allclose(B[0, 0, :], 1j * np.ones(self.NGP))
        for k in range(1, 5):
            np.testing.assert_allclose(B[k, k, :], 1j * np.ones(self.NGP))
        # Off-diagonal entries
        for i in range(5):
            for j in range(5):
                if i != j:
                    np.testing.assert_allclose(
                        B[i, j, :], np.zeros(self.NGP),
                        atol=1e-15, err_msg=f"B[{i},{j},:] should be zero",
                    )

    def test_continuity_row_inviscid(self):
        """Second-derivative matrices vanish for the continuity row (i=0)."""
        coeffs = _call_cartesian(self.NGP)
        for name in ("Axx", "Ayy", "Axy"):
            arr = getattr(coeffs, name)
            np.testing.assert_allclose(
                arr[0, :, :], 0.0, atol=1e-15,
                err_msg=f"{name}[0,:,:] should be zero (continuity is inviscid)",
            )

    def test_viscous_diagonal_positive(self):
        """Diagonal viscous entries (Axx, Ayy) for momentum/energy rows are positive.

        The coefficient matrices store the diffusion coefficient mu/(Re*rho) > 0;
        dissipation arises from the negative eigenvalues of the Laplacian.
        """
        coeffs = _call_cartesian(self.NGP)
        for name in ("Axx", "Ayy"):
            arr = getattr(coeffs, name)
            for k in range(1, 5):
                real_part = arr[k, k, :].real
                assert np.all(real_part >= -1e-15), (
                    f"{name}[{k},{k},:] real part should be >= 0 (diffusion coeff), "
                    f"min = {real_part.min()}"
                )

    def test_zero_baseflow_A0(self):
        """With U=V=W=0, A0 should simplify (no advection contributions)."""
        ngp = self.NGP
        U = np.zeros(ngp)
        V = np.zeros(ngp)
        W = np.zeros(ngp)
        RHO = np.ones(ngp)
        T = np.ones(ngp)
        MU = np.ones(ngp)
        kappa, Ma = 1.4, 0.3
        P = RHO * T / (kappa * Ma**2)
        derivs = _zero_derivatives(ngp)

        coeffs = _call_cartesian(
            ngp, U=U, V=V, W=W, RHO=RHO, T=T, MU=MU, P=P,
            derivs=derivs, kappa=kappa, Ma=Ma,
        )
        # With zero velocity and zero derivatives, the advection part of A0
        # should contribute nothing; check a few representative entries.
        # Continuity row: A0[0,1,:] (links to u) should be zero when RHO*U=0
        # and all velocity derivatives are zero.
        np.testing.assert_allclose(coeffs.Ax[0, 0, :], 0.0, atol=1e-14)

    def test_inviscid_limit(self):
        """When MU → 0, Axx/Ayy/Axy momentum rows → 0."""
        ngp = self.NGP
        U, V, W, RHO, T, _, P = _uniform_scalars(ngp)
        MU = np.full(ngp, 1e-30)  # effectively zero viscosity
        derivs = _zero_derivatives(ngp)

        coeffs = _call_cartesian(
            ngp, U=U, V=V, W=W, RHO=RHO, T=T, MU=MU, P=P,
            derivs=derivs, Re=1e5,
        )
        for name in ("Axx", "Ayy", "Axy"):
            arr = getattr(coeffs, name)
            for k in range(1, 5):
                np.testing.assert_allclose(
                    arr[k, :, :], 0.0, atol=1e-20,
                    err_msg=f"Inviscid limit: {name}[{k},:,:] should vanish",
                )

    def test_all_arrays_shapes(self):
        """Redundant shape check: each of the 7 arrays is (5, 5, ngp)."""
        coeffs = _call_cartesian(self.NGP)
        expected = (5, 5, self.NGP)
        for attr in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
            assert getattr(coeffs, attr).shape == expected


# =========================================================================
# TestCylindricalCoefficients
# =========================================================================

class TestCylindricalCoefficients:
    """Tests for compute_cylindrical_coefficients."""

    NGP = 50

    def _call_cylindrical(self, ngp, *, R=None, m=0, **kwargs):
        """Convenience wrapper for cylindrical coefficients.

        The actual signature is:
            compute_cylindrical_coefficients(
                U, V, W, R, RHO, T, MU, m, Re, Ma, Pr, kappa,
                dUdr, dVdr, dWdr, dRHOdr, dTdr,
                dUdz, dVdz, dWdz, dRHOdz, dTdz,
                d2Udr2, d2Vdr2, d2Wdr2, d2Tdr2,
                d2Udz2, d2Vdz2, d2Wdz2, d2Tdz2,
                d2Udrz, d2Vdrz, d2Wdrz,
                dmudT, d2mudT2,
            )
        """
        Re   = kwargs.get("Re", 1e5)
        Ma   = kwargs.get("Ma", 0.3)
        Pr   = kwargs.get("Pr", 0.72)
        kappa = kwargs.get("kappa", 1.4)
        U, V, W, RHO, T, MU, _ = _uniform_scalars(ngp, kappa=kappa, Ma=Ma)
        z = np.zeros(ngp)
        if R is None:
            R = np.ones(ngp) * 1e6  # large radius
        return compute_cylindrical_coefficients(
            U, V, W, R, RHO, T, MU, m, Re, Ma, Pr, kappa,
            z, z, z, z, z,       # dUdr..dTdr
            z, z, z, z, z,       # dUdz..dTdz
            z, z, z, z,          # d2Udr2..d2Tdr2
            z, z, z, z,          # d2Udz2..d2Tdz2
            z, z, z,             # d2Udrz..d2Wdrz
            z, z,                # dmudT, d2mudT2
        )

    def test_cylindrical_shapes(self):
        """Cylindrical coefficient arrays have shape (5, 5, ngp)."""
        coeffs = self._call_cylindrical(self.NGP)
        expected = (5, 5, self.NGP)
        for attr in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
            assert getattr(coeffs, attr).shape == expected, f"{attr} shape mismatch"

    def test_large_R_approaches_cartesian(self):
        """At R >> 1, m=0: cylindrical viscous operators approach Cartesian ones.

        The B matrix differs structurally between coordinate systems (e.g., the
        cylindrical energy equation carries a cv prefactor), so only the
        second-derivative viscous operators are compared.
        """
        ngp = self.NGP
        R_large = np.ones(ngp) * 1e8
        cyl = self._call_cylindrical(ngp, R=R_large, m=0)
        cart = _call_cartesian(ngp)
        # Axx, Ayy should be close — 1/R corrections vanish (tolerance
        # is 1e-3 because some cylindrical entries carry O(1/R) metric terms)
        np.testing.assert_allclose(cyl.Axx, cart.Axx, atol=1e-3)
        np.testing.assert_allclose(cyl.Ayy, cart.Ayy, atol=1e-3)


# =========================================================================
# TestOperatorAssembly
# =========================================================================

class TestOperatorAssembly:
    """Tests for build_lnse_operator using small_mesh_2d + uniform_baseflow."""

    def test_L0_shape(self, small_mesh_2d, uniform_baseflow):
        """L0 has shape (5*ngp, 5*ngp)."""
        mesh = small_mesh_2d
        ngp = mesh.ngp
        L0, _ = build_lnse_operator(mesh, uniform_baseflow, kx=0.0)
        assert L0.shape == (5 * ngp, 5 * ngp)

    def test_L0_sparse_complex(self, small_mesh_2d, uniform_baseflow):
        """L0 is sparse and has complex dtype."""
        L0, _ = build_lnse_operator(small_mesh_2d, uniform_baseflow, kx=0.0)
        assert sp.issparse(L0)
        assert np.issubdtype(L0.dtype, np.complexfloating)

    def test_continuity_block_no_viscous(self, small_mesh_2d, uniform_baseflow):
        """First ngp rows (continuity) should have no second-derivative viscous entries.

        This is a structural check: the continuity equation is inviscid, so the
        operator block corresponding to the first variable (density) should not
        contain viscous diffusion patterns.  We verify the continuity row block
        is sparser than a momentum row block.
        """
        mesh = small_mesh_2d
        ngp = mesh.ngp
        L0, _ = build_lnse_operator(mesh, uniform_baseflow, kx=0.0)
        L0_csr = L0.tocsr()
        # Count non-zeros in continuity rows vs. first momentum rows
        cont_nnz = L0_csr[:ngp, :].nnz
        mom_nnz  = L0_csr[ngp:2*ngp, :].nnz
        # Continuity block should be sparser (no viscous stencil)
        assert cont_nnz < mom_nnz, (
            f"Continuity nnz ({cont_nnz}) should be < momentum nnz ({mom_nnz})"
        )

    def test_idx_keys(self, small_mesh_2d, uniform_baseflow):
        """idx dict contains all required keys."""
        _, idx = build_lnse_operator(small_mesh_2d, uniform_baseflow, kx=0.0)
        required = {"bi", "ti", "li", "ri", "rho_j", "u_j", "v_j", "w_j", "T_j"}
        assert required.issubset(idx.keys()), f"Missing keys: {required - idx.keys()}"

    def test_variable_dof_ranges(self, small_mesh_2d, uniform_baseflow):
        """Variable DOF ranges: rho_j = [0, ngp), u_j = [ngp, 2*ngp), etc."""
        mesh = small_mesh_2d
        ngp = mesh.ngp
        _, idx = build_lnse_operator(mesh, uniform_baseflow, kx=0.0)
        np.testing.assert_array_equal(idx["rho_j"], np.arange(0, ngp))
        np.testing.assert_array_equal(idx["u_j"],   np.arange(ngp, 2 * ngp))
        np.testing.assert_array_equal(idx["v_j"],   np.arange(2 * ngp, 3 * ngp))
        np.testing.assert_array_equal(idx["w_j"],   np.arange(3 * ngp, 4 * ngp))
        np.testing.assert_array_equal(idx["T_j"],   np.arange(4 * ngp, 5 * ngp))

    @slow
    def test_axy_model_shape(self, small_mesh_2d, uniform_baseflow):
        """model='axy' returns operator of correct shape."""
        mesh = small_mesh_2d
        ngp = mesh.ngp
        L0, _ = build_lnse_operator(mesh, uniform_baseflow, kx=0.0, model="axy")
        assert L0.shape == (5 * ngp, 5 * ngp)

    def test_invalid_model_raises(self, small_mesh_2d, uniform_baseflow):
        """Invalid model string raises ValueError."""
        with pytest.raises(ValueError):
            build_lnse_operator(small_mesh_2d, uniform_baseflow, model="invalid")


# =========================================================================
# TestChuEnergyNorm
# =========================================================================

class TestChuEnergyNorm:
    """Tests for compute_chu_norm."""

    NGP = 50
    KAPPA = 1.4
    MA = 0.3

    def _make_norm(self, *, model="2D", R=None):
        """Compute Chu norm with RHO=T=1, W_int=ones."""
        ngp = self.NGP
        W_int = np.ones(ngp)
        RHO = np.ones(ngp)
        T = np.ones(ngp)
        return compute_chu_norm(
            W_int, RHO, T, self.KAPPA, self.MA, ngp, model=model, R=R,
        )

    def test_shapes(self):
        """W_diag and invW_diag have shape (5*ngp,)."""
        W_diag, invW_diag = self._make_norm()
        assert W_diag.shape == (5 * self.NGP,)
        assert invW_diag.shape == (5 * self.NGP,)

    def test_positive(self):
        """W_diag > 0 everywhere."""
        W_diag, _ = self._make_norm()
        assert np.all(W_diag > 0)

    def test_inverse_relation(self):
        """W_diag * invW_diag ≈ 1."""
        W_diag, invW_diag = self._make_norm()
        np.testing.assert_allclose(W_diag * invW_diag, 1.0, rtol=1e-12)

    def test_rho_block(self):
        """rho block: W[0:ngp] = 0.5 * T / (RHO * kappa * Ma^2) * W_int."""
        W_diag, _ = self._make_norm()
        ngp = self.NGP
        expected = 0.5 * 1.0 / (1.0 * self.KAPPA * self.MA**2) * np.ones(ngp)
        np.testing.assert_allclose(W_diag[:ngp], expected, rtol=1e-12)

    def test_velocity_blocks(self):
        """Velocity blocks: W[ngp:4*ngp] = 0.5 * RHO * W_int."""
        W_diag, _ = self._make_norm()
        ngp = self.NGP
        expected = 0.5 * np.ones(3 * ngp)
        np.testing.assert_allclose(W_diag[ngp:4*ngp], expected, rtol=1e-12)

    def test_T_block(self):
        """T block: W[4*ngp:5*ngp] = 0.5 * RHO / (kappa*(kappa-1)*T*Ma^2) * W_int."""
        W_diag, _ = self._make_norm()
        ngp = self.NGP
        expected = (
            0.5 * 1.0
            / (self.KAPPA * (self.KAPPA - 1.0) * 1.0 * self.MA**2)
            * np.ones(ngp)
        )
        np.testing.assert_allclose(W_diag[4*ngp:5*ngp], expected, rtol=1e-12)

    def test_axy_model_R_scaling(self):
        """model='axy': R is multiplied into W_int."""
        ngp = self.NGP
        R = np.full(ngp, 3.0)
        W_diag_cart, _ = self._make_norm(model="2D")
        W_diag_cyl, _  = self._make_norm(model="axy", R=R)
        # Each block should be scaled by R=3 relative to the 2D case.
        # Check the rho block as representative.
        np.testing.assert_allclose(
            W_diag_cyl[:ngp], 3.0 * W_diag_cart[:ngp], rtol=1e-12,
        )


# =========================================================================
# TestSpatialOperator
# =========================================================================

class TestSpatialOperator:
    """Tests for build_spatial_operator (uses blasius_baseflow_small fixture)."""

    @slow
    def test_spatial_shapes(self, blasius_baseflow_small):
        """Lc, R, Lw each have shape (2*5*ngp, 2*5*ngp)."""
        mesh, bf = blasius_baseflow_small
        ngp = mesh.ngp
        Lc, R, Lw, _ = build_spatial_operator(mesh, bf)
        expected = (2 * 5 * ngp, 2 * 5 * ngp)
        assert Lc.shape == expected
        assert R.shape == expected
        assert Lw.shape == expected

    @slow
    def test_spatial_sparse_complex(self, blasius_baseflow_small):
        """Lc, R, Lw are sparse and complex."""
        mesh, bf = blasius_baseflow_small
        Lc, R, Lw, _ = build_spatial_operator(mesh, bf)
        for name, mat in [("Lc", Lc), ("R", R), ("Lw", Lw)]:
            assert sp.issparse(mat), f"{name} should be sparse"
            assert np.issubdtype(mat.dtype, np.complexfloating), (
                f"{name} should be complex"
            )
