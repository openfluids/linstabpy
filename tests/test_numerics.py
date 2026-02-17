"""Numerical soundness tests for the cylindrical (axy) LNSE operators.

These tests verify mathematical properties that must hold regardless of the
specific eigenvalue spectrum, ensuring the assembled operators are self-consistent
and the discretization converges.

All tests are marked @pytest.mark.slow.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from linstab2d.mesh import Mesh
from linstab2d.baseflow import BaseFlow, sutherland_air
from linstab2d.operators import build_lnse_operator, compute_chu_norm, build_spatial_operator
from linstab2d.operators.assembly import _apply_inv_B
from linstab2d.boundary_conditions import apply_dirichlet, apply_dirichlet_spatial
from linstab2d.solvers import temporal_eigenvalues, resolvent_analysis

from tests.conftest import build_synthetic_jet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eigenvalues_at_resolution(nx, ny, m=0, target=-0.1j, n_modes=5):
    """Compute temporal eigenvalues on a synthetic jet at given resolution."""
    d = build_synthetic_jet(nx=nx, ny=ny, m=m)
    omega, modes = temporal_eigenvalues(d["L0"], n_modes=n_modes, target=target)
    return omega, modes, d


# =========================================================================
# TestGridConvergence
# =========================================================================

@pytest.mark.slow
class TestGridConvergence:
    """Verify eigenvalue residuals improve under grid refinement.

    Direct eigenvalue convergence is hard to test on very coarse grids because
    different modes dominate at each resolution.  Instead we check that the
    eigenpair residual ||iL0@v - ω*v||/||v|| decreases as the grid refines,
    which confirms the discretization is self-consistent.
    """

    def test_residual_improves_with_refinement(self):
        """Eigenpair residual decreases on finer grids (axy, m=0).

        Grid sequence: 10×8 → 15×10 → 20×14.
        """
        resolutions = [(10, 8), (15, 10), (20, 14)]
        max_residuals = []

        for nx, ny in resolutions:
            omega, modes, d = _eigenvalues_at_resolution(
                nx, ny, m=0, target=-0.1j, n_modes=3,
            )
            iL0 = 1j * d["L0"]
            worst = 0.0
            for i in range(len(omega)):
                v = modes[:, i]
                res = np.linalg.norm(iL0 @ v - omega[i] * v) / np.linalg.norm(v)
                worst = max(worst, res)
            max_residuals.append(worst)

        # Finer grids should have smaller (or equal) residuals
        assert max_residuals[2] <= max_residuals[0], (
            f"Residual not improving: {max_residuals}"
        )


# =========================================================================
# TestResolventNumerics
# =========================================================================

@pytest.mark.slow
class TestResolventNumerics:
    """Verify resolvent operator properties for the axy model."""

    @pytest.fixture(scope="class")
    def jet_resolvent(self):
        """Build jet + compute resolvent."""
        d = build_synthetic_jet(nx=15, ny=10, m=0)
        omega = 2.0
        gains, responses, forcings = resolvent_analysis(
            d["L0"], omega=omega, n_modes=3, W_diag=d["W_diag"],
        )
        return d, omega, gains, responses, forcings

    def test_resolvent_gain_upper_bound(self, jet_resolvent):
        """σ₁ ≥ ||H(ω)f||_W / ||f||_W for random forcing vectors.

        The leading singular value of the resolvent is its operator norm.
        Any random forcing should produce a response with gain ≤ σ₁.
        """
        d, omega, gains, _, _ = jet_resolvent
        L0 = d["L0"]
        W_diag = d["W_diag"]
        sigma_1 = gains[0]

        # Build resolvent H(ω) = (iωI - L0)^{-1}
        n = L0.shape[0]
        L_sigma = 1j * omega * sp.eye(n) - L0
        from linstab2d.solvers.linsolve import make_linear_solver
        solver = make_linear_solver(L_sigma.tocsc())

        rng = np.random.default_rng(42)
        for trial in range(5):
            f = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            response = solver.solve(f)
            # W-weighted norms
            f_norm = np.sqrt(np.real(f.conj() @ (W_diag * f)))
            r_norm = np.sqrt(np.real(response.conj() @ (W_diag * response)))
            achieved_gain = r_norm / f_norm
            assert achieved_gain <= sigma_1 * 1.01, (
                f"Trial {trial}: achieved gain {achieved_gain:.4f} > sigma_1 {sigma_1:.4f}"
            )

    def test_optimal_forcing_achieves_sigma_1(self, jet_resolvent):
        """The leading forcing mode achieves exactly σ₁ (within tolerance)."""
        d, omega, gains, responses, forcings = jet_resolvent
        L0 = d["L0"]
        W_diag = d["W_diag"]

        n = L0.shape[0]
        L_sigma = 1j * omega * sp.eye(n) - L0
        from linstab2d.solvers.linsolve import make_linear_solver
        solver = make_linear_solver(L_sigma.tocsc())

        f1 = forcings[:, 0]
        response = solver.solve(f1)
        f_norm = np.sqrt(np.real(f1.conj() @ (W_diag * f1)))
        r_norm = np.sqrt(np.real(response.conj() @ (W_diag * response)))
        achieved = r_norm / f_norm

        np.testing.assert_allclose(
            achieved, gains[0], rtol=1e-4,
            err_msg="Optimal forcing should achieve σ₁",
        )



# =========================================================================
# TestSpectrumProperties
# =========================================================================

@pytest.mark.slow
class TestSpectrumProperties:
    """Verify mathematical properties of the LNSE eigenvalue spectrum."""

    def test_eigenvalues_near_real_axis(self):
        """Temporal eigenvalues have small real parts for this axisymmetric flow.

        Convention: ω = eig(iL0), so Re(ω) is the frequency and Im(ω) is
        related to the growth rate.  For the synthetic sech² jet, the physical
        eigenvalues (large |ω|) should have near-zero real parts because the
        base flow has no azimuthal velocity and the problem is axisymmetric.
        """
        d = build_synthetic_jet(nx=15, ny=10, m=0)
        omega, _ = temporal_eigenvalues(d["L0"], n_modes=10, target=-0.01j)

        # Physical modes (filter numerical near-zero artifacts)
        physical = omega[np.abs(omega) > 1e-4]
        for i, w in enumerate(physical):
            assert abs(w.real) < 1e-6, (
                f"Eigenvalue {i} has unexpectedly large Re(ω): ω = {w} "
                f"(expected near-zero frequency for axisymmetric m=0 flow)"
            )

    def test_sponge_increases_damping(self):
        """Adding a sponge shifts the dominant eigenvalue toward more negative Im(ω).

        The sponge subtracts a positive diagonal from L0.  Since ω = eig(iL0),
        the sponge contribution to ω is i*(-sponge) = -i*sponge, which shifts
        Im(ω) more negative. We verify this on the dominant mode.
        """
        # Without sponge
        d = build_synthetic_jet(nx=15, ny=10, m=0)
        omega_no_sponge, _ = temporal_eigenvalues(
            d["L0"], n_modes=5, target=-0.1j,
        )

        # With sponge: add damping in the freestream region
        mesh = d["mesh"]
        p = mesh.used_ind
        yf = mesh.Y.ravel(order="F")[p]
        sponge_amp = 2.0
        sponge_vals = sponge_amp * 0.5 * (np.tanh((yf - 2.5) / 0.2) + 1)
        sponge_diag = np.tile(sponge_vals, 5)
        L0_sponge = d["L0"] - sp.diags(sponge_diag)

        omega_sponge, _ = temporal_eigenvalues(
            L0_sponge, n_modes=5, target=-0.1j,
        )

        # The dominant mode (most negative Im) should be even more negative
        dom_no_sponge = np.min(omega_no_sponge.imag)
        dom_sponge = np.min(omega_sponge.imag)
        assert dom_sponge < dom_no_sponge, (
            f"Sponge did not increase damping of dominant mode: "
            f"without={dom_no_sponge:.4e}, with={dom_sponge:.4e}"
        )

    def test_m0_m1_spectra_differ(self):
        """m=0 and m=1 spectra should be different (different parity).

        Different azimuthal modes use different radial derivative operators
        at the pole, so they must produce different eigenvalue spectra.
        """
        omega_m0, _, _ = _eigenvalues_at_resolution(15, 10, m=0)
        omega_m1, _, _ = _eigenvalues_at_resolution(15, 10, m=1)

        # Sort by imaginary part for comparison
        o0 = np.sort(omega_m0.imag)
        o1 = np.sort(omega_m1.imag)

        # They should NOT be identical
        assert not np.allclose(o0, o1, atol=1e-6), (
            "m=0 and m=1 spectra are identical — parity selection has no effect"
        )


# =========================================================================
# TestAdjointConsistency
# =========================================================================

@pytest.mark.slow
class TestAdjointConsistency:
    """Verify adjoint operator consistency for cylindrical model."""

    def test_bilinear_form_multiple_vectors(self):
        """<Lu, v>_W = <u, L†v>_W for 10 random vector pairs.

        Tests the fundamental adjoint identity that must hold for the
        Chu energy-norm weighted inner product.
        """
        d = build_synthetic_jet(nx=15, ny=10, m=0)
        L0 = d["L0"]
        W = d["W_diag"]
        invW = d["invW_diag"]
        n = L0.shape[0]

        rng = np.random.default_rng(123)
        for trial in range(10):
            u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            v = rng.standard_normal(n) + 1j * rng.standard_normal(n)

            lhs = np.vdot(v, W * (L0 @ u))
            L_adj_v = invW * (L0.conj().T @ (W * v))
            rhs = np.vdot(L_adj_v, W * u)

            np.testing.assert_allclose(
                lhs, rhs, rtol=1e-8,
                err_msg=f"Adjoint identity failed on trial {trial}",
            )

    def test_bilinear_form_m1(self):
        """Adjoint identity holds for m=1 (different parity)."""
        d = build_synthetic_jet(nx=15, ny=10, m=1)
        L0 = d["L0"]
        W = d["W_diag"]
        invW = d["invW_diag"]
        n = L0.shape[0]

        rng = np.random.default_rng(456)
        u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        lhs = np.vdot(v, W * (L0 @ u))
        L_adj_v = invW * (L0.conj().T @ (W * v))
        rhs = np.vdot(L_adj_v, W * u)

        np.testing.assert_allclose(lhs, rhs, rtol=1e-8)


# =========================================================================
# TestInvBConsistency
# =========================================================================

@pytest.mark.slow
class TestInvBConsistency:
    """Verify inv(B) consistency between Cartesian and cylindrical models."""

    def test_cartesian_axy_consistent_for_planar_flow(self):
        """For a uniform flow with large R, Cartesian and axy eigenvalues agree.

        When the radius R→∞ and m=0, cylindrical effects vanish and the
        operator should approach the Cartesian one.  The eigenvalues won't
        match exactly (metric terms scale as 1/R), but should be close.
        """
        # Cartesian problem
        mesh_cart = Mesh.create(
            xrange=(0.0, 4.0), yrange=(0.0, 2.0),
            nx=15, ny=10, fd_order=4,
        )
        bf_cart = BaseFlow(
            Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15,
            RHO=np.ones_like(mesh_cart.X), U=np.ones_like(mesh_cart.X),
            V=np.zeros_like(mesh_cart.X), W=np.zeros_like(mesh_cart.X),
            T=np.ones_like(mesh_cart.X),
        )
        sutherland_air(bf_cart)
        L0_cart, idx_cart = build_lnse_operator(mesh_cart, bf_cart, kx=0.0, model="2D")
        L0_cart, _ = apply_dirichlet(L0_cart, idx_cart, "bt", "uvwT", mesh_cart.ngp)
        omega_cart, _ = temporal_eigenvalues(L0_cart, n_modes=5, target=-0.1j)

        # Cylindrical problem with large R offset (simulates R→∞)
        mesh_cyl = Mesh.create(
            xrange=(0.0, 4.0), yrange=(0.0, 2.0),
            nx=15, ny=10, fd_order=4, y_symmetry=False,
        )
        # Shift Y to large radius so 1/R terms vanish
        Y_shifted = mesh_cyl.Y + 1000.0
        mesh_cyl = mesh_cyl.deform(Y_new=Y_shifted)

        bf_cyl = BaseFlow(
            Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15,
            RHO=np.ones_like(mesh_cyl.X), U=np.ones_like(mesh_cyl.X),
            V=np.zeros_like(mesh_cyl.X), W=np.zeros_like(mesh_cyl.X),
            T=np.ones_like(mesh_cyl.X),
        )
        sutherland_air(bf_cyl)
        L0_cyl, idx_cyl = build_lnse_operator(
            mesh_cyl, bf_cyl, kx=0.0, model="axy", m=0,
        )
        L0_cyl, _ = apply_dirichlet(L0_cyl, idx_cyl, "bt", "uvwT", mesh_cyl.ngp)
        omega_cyl, _ = temporal_eigenvalues(L0_cyl, n_modes=5, target=-0.1j)

        # Match eigenvalues: find closest pairs
        matched = 0
        for oc in omega_cart:
            dists = np.abs(omega_cyl - oc)
            if np.min(dists) < 0.5 * np.abs(oc) + 0.1:  # generous tolerance
                matched += 1

        assert matched >= 2, (
            f"Only {matched}/5 Cartesian eigenvalues matched cylindrical (R=1000).\n"
            f"Cartesian: {omega_cart}\n"
            f"Cylindrical: {omega_cyl}"
        )


# =========================================================================
# TestSpatialEVPNumerics
# =========================================================================

@pytest.mark.slow
class TestSpatialEVPNumerics:
    """Numerical tests for the spatial eigenvalue problem with axy model."""

    def test_spatial_operator_structure(self):
        """Spatial operator matrices satisfy structural identities.

        The spatial EVP matrices are constructed from three LNSE operators
        at different Floquet shifts:
            R0 = L(alpha=0), Lp = L(+delta), Lm = L(-delta)
        with R1 = (Lp - Lm)/(2*delta), R2 = (Lp - R0 - R1*delta)/delta^2.

        We verify Lc, R, Lw have the expected block structure.
        """
        mesh = Mesh.create(
            xrange=(0.0, 10.0), yrange=(0.0, 3.0),
            nx=15, ny=10, fd_order=4, y_symmetry=True,
        )
        R_field = mesh.Y
        U_axial = 0.9 / np.cosh(R_field)**2
        bf = BaseFlow(
            Re=1e4, Ma=0.9, Pr=0.7, kappa=1.4, T_0=288.15,
            RHO=np.ones_like(R_field), U=U_axial,
            V=np.zeros_like(R_field), W=np.zeros_like(R_field),
            T=np.ones_like(R_field),
        )
        sutherland_air(bf)

        Lc, R_mat, Lw, idx = build_spatial_operator(mesh, bf, model="axy", m=0)
        n = mesh.ngp * 5

        # Lw should have -iI in its bottom-left block
        Lw_dense = Lw.toarray()
        np.testing.assert_allclose(
            Lw_dense[:n, :n], 0.0, atol=1e-14,
            err_msg="Lw top-left should be zero",
        )
        np.testing.assert_allclose(
            Lw_dense[:n, n:], 0.0, atol=1e-14,
            err_msg="Lw top-right should be zero",
        )
        np.testing.assert_allclose(
            Lw_dense[n:, n:], 0.0, atol=1e-14,
            err_msg="Lw bottom-right should be zero",
        )
        # Bottom-left block should be -iI
        expected_bl = -1j * np.eye(n)
        np.testing.assert_allclose(
            Lw_dense[n:, :n], expected_bl, atol=1e-14,
            err_msg="Lw bottom-left should be -iI",
        )

        # Lc top-left block should be zero, top-right should be I
        Lc_dense = Lc.toarray()
        np.testing.assert_allclose(
            Lc_dense[:n, :n], 0.0, atol=1e-14,
            err_msg="Lc top-left should be zero",
        )
        np.testing.assert_allclose(
            Lc_dense[:n, n:], np.eye(n), atol=1e-14,
            err_msg="Lc top-right should be I",
        )

        # R top-left block should be I
        R_dense = R_mat.toarray()
        np.testing.assert_allclose(
            R_dense[:n, :n], np.eye(n), atol=1e-14,
            err_msg="R top-left should be I",
        )
        np.testing.assert_allclose(
            R_dense[:n, n:], 0.0, atol=1e-14,
            err_msg="R top-right should be zero",
        )
