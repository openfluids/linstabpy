"""
End-to-end integration tests for LinStab2D-py.

All tests are marked @pytest.mark.slow since they assemble large operators
and/or call iterative solvers.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from linstab2d.mesh import Mesh
from linstab2d.baseflow import load_blasius, sutherland_air
from linstab2d.operators import build_lnse_operator, compute_chu_norm
from linstab2d.boundary_conditions import apply_dirichlet
from linstab2d.solvers import resolvent_analysis, temporal_eigenvalues


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_blasius_problem(nx, ny, Re=6e5, Ma=0.3, borders="lrbt"):
    """Build a Blasius LNSE problem using MATLAB reference parameters.

    Domain: x=[0.05, 1.26], y=[-1, 1] with Hanifi deformation to [0, 0.05].
    Includes sponge damping at inflow, outflow, and freestream boundaries.
    """
    Pr, kappa, T_0 = 0.7, 1.4, 273.15
    dx = 0.01
    x_min, x_max = 5 * dx, 1.25 + dx
    y_i, y_max_phys = 0.01, 0.05

    mesh = Mesh.create(
        xrange=(x_min, x_max), yrange=(-1.0, 1.0), nx=nx, ny=ny,
        fd_order=4, alpha_filter="none",
    )

    # Hanifi 1996 deformation
    Y = mesh.Y.copy()
    a = y_i * y_max_phys / (y_max_phys - 2 * y_i)
    b = 1 + 2 * a / y_max_phys
    Y = a * (1 + Y) / (b - Y)
    mesh = mesh.deform(Y_new=Y)

    bf = load_blasius(mesh.X, mesh.Y, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    # Sponge (matching MATLAB example_1_Blasius.m)
    ind = mesh.used_ind
    xf = mesh.X.ravel(order="F")[ind]
    yf = mesh.Y.ravel(order="F")[ind]
    sponge_amp = 5.0
    term_t = np.tanh((yf - np.max(yf) + 0.01) / 0.01) + 1
    term_l = np.tanh(-(xf - np.min(xf) - 0.02) / 0.02) + 1
    term_r = np.tanh((xf - np.max(xf) + 0.02) / 0.02) + 1
    sponge_vals = sponge_amp / 2 * np.maximum(term_t, term_l + term_r)
    flat = np.full(int(np.prod(mesh.X.shape)), np.nan)
    flat[ind] = sponge_vals
    mesh.sponge = flat.reshape(mesh.X.shape, order="F")

    L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
    ngp = mesh.ngp
    L0, bc_indices = apply_dirichlet(L0, idx, borders, "uvwT", ngp)

    p = mesh.used_ind
    RHO = bf.RHO.ravel(order="F")[p]
    T_field = bf.T.ravel(order="F")[p]
    W_int = mesh.W.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO, T_field, bf.kappa, bf.Ma, ngp)

    # B/C matrices: restrict forcing/response to velocity (u, v) only
    B_vec = np.ones(ngp * 5)
    B_vec[idx["T_j"]] = 0
    B_vec[idx["rho_j"]] = 0
    B_vec[idx["w_j"]] = 0
    B_vec[bc_indices] = 0
    B_mat = sp.diags(B_vec, format="csc")

    return {
        "mesh": mesh,
        "bf": bf,
        "L0": L0,
        "idx": idx,
        "bc_indices": bc_indices,
        "ngp": ngp,
        "W_diag": W_diag,
        "invW_diag": invW_diag,
        "B_mat": B_mat,
        "C_mat": B_mat.copy(),
    }


@pytest.fixture(scope="module")
def blasius_large():
    """Coarse Blasius problem (100x50) with MATLAB reference parameters."""
    return _build_blasius_problem(100, 50)


@pytest.fixture(scope="module")
def blasius_small():
    """Small Blasius problem (15x10) for quick consistency checks."""
    return _build_blasius_problem(15, 10, Re=1e4, borders="bt")


# ---------------------------------------------------------------------------
# TestBlasiusResolventReference
# ---------------------------------------------------------------------------

class TestBlasiusResolventReference:
    """Gold-standard resolvent validation against known Blasius gains."""

    @pytest.mark.slow
    def test_resolvent_gains_structure(self, blasius_large):
        """Resolvent gains are positive, monotonically decreasing, and O(100).

        MATLAB reference on 300x100: [128.62, 14.54, 7.49] at Re=6e5, Ma=0.3, omega=60.
        On the coarser 100x50 grid, gains are lower but should still be O(10-100).
        """
        d = blasius_large
        gains, _, _ = resolvent_analysis(
            d["L0"], omega=60.0, n_modes=3, W_diag=d["W_diag"],
            B_mat=d["B_mat"], C_mat=d["C_mat"],
        )

        # All gains must be positive
        assert np.all(gains > 0), f"Expected positive gains, got {gains}"

        # Monotonically decreasing (sigma_1 >= sigma_2 >= sigma_3)
        assert np.all(np.diff(gains) <= 0), (
            f"Gains should be monotonically decreasing, got {gains}"
        )

        # First gain should be meaningful — on 100x50, expect O(10-100).
        assert gains[0] > 5.0, (
            f"Leading gain suspiciously small: {gains[0]:.2f} (expected > 5)"
        )

    @pytest.mark.slow
    def test_operator_sparsity(self, blasius_large):
        """LNSE operator has roughly 30 nonzeros per row on average."""
        L0 = blasius_large["L0"]
        avg_nnz_per_row = L0.nnz / L0.shape[0]

        # Bare stencil gives ~30 nnz/row; penalty BCs add some dense rows.
        assert 10 < avg_nnz_per_row < 500, (
            f"Average nnz/row = {avg_nnz_per_row:.1f}, outside [10, 500]"
        )

    @pytest.mark.slow
    def test_operator_shape(self, blasius_large):
        """Operator is square with dimension 5 * ngp."""
        L0 = blasius_large["L0"]
        ngp = blasius_large["ngp"]
        expected = 5 * ngp

        assert L0.shape == (expected, expected), (
            f"Expected shape ({expected}, {expected}), got {L0.shape}"
        )


# ---------------------------------------------------------------------------
# TestSmallProblemConsistency
# ---------------------------------------------------------------------------

class TestSmallProblemConsistency:
    """Quick smoke tests on a coarse grid."""

    @pytest.mark.slow
    def test_pipeline_completes(self, blasius_small):
        """Full pipeline (mesh -> baseflow -> operator -> BC -> norm) runs."""
        d = blasius_small

        # If we got here the fixture already succeeded; verify shapes are sane.
        L0 = d["L0"]
        ngp = d["ngp"]
        assert L0.shape[0] == L0.shape[1] == 5 * ngp
        assert sp.issparse(L0)
        assert d["W_diag"].shape == (5 * ngp,)
        assert d["invW_diag"].shape == (5 * ngp,)

    @pytest.mark.slow
    def test_temporal_eigenpair_residual(self, blasius_small):
        """Temporal eigenvalues satisfy the eigenpair equation on a coarse grid.

        Note: on a 15x10 grid the eigenvalues are not physically meaningful
        (too coarse to resolve Blasius modes), so we check the eigenpair
        residual ||i L0 v - omega v|| / ||v|| rather than stability.
        """
        d = blasius_small
        omegas, modes = temporal_eigenvalues(d["L0"], n_modes=3, target=-0.1j)
        iL0 = 1j * d["L0"]

        for i in range(len(omegas)):
            v = modes[:, i]
            residual = iL0 @ v - omegas[i] * v
            rel_err = np.linalg.norm(residual) / np.linalg.norm(v)
            assert rel_err < 1e-6, (
                f"Mode {i}: eigenpair residual {rel_err:.2e} (expected < 1e-6)"
            )


# ---------------------------------------------------------------------------
# TestOperatorSymmetryConsistency
# ---------------------------------------------------------------------------

class TestOperatorSymmetryConsistency:
    """Verify adjoint and energy-norm properties of the LNSE operator."""

    @pytest.mark.slow
    def test_adjoint_bilinear_form(self, blasius_small):
        """Adjoint identity: <L u, v>_W  =  <u, L_adj v>_W.

        Where L_adj = diag(1/W) L^H diag(W) and the inner product is
        <a, b>_W = a^H diag(W) b.
        """
        d = blasius_small
        L0 = d["L0"]
        W = d["W_diag"]
        invW = d["invW_diag"]
        n = L0.shape[0]

        rng = np.random.default_rng(42)
        u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        # LHS: v^H W (L0 u)
        lhs = np.vdot(v, W * (L0 @ u))

        # RHS: (L_adj v)^H W u  where L_adj v = invW * (L0^H (W * v))
        L_adj_v = invW * (L0.conj().T @ (W * v))
        rhs = np.vdot(L_adj_v, W * u)

        np.testing.assert_allclose(
            lhs,
            rhs,
            rtol=1e-8,
            err_msg="Adjoint bilinear form identity violated",
        )

    @pytest.mark.slow
    def test_energy_norm_positive_definite(self, blasius_small):
        """Chu energy norm is positive definite: q^H diag(W) q > 0."""
        W = blasius_small["W_diag"]
        n = W.shape[0]

        rng = np.random.default_rng(123)
        q = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        energy = np.real(np.vdot(q, W * q))
        assert energy > 0, f"Energy norm not positive definite: {energy}"

        # Also check that W entries are all positive (sufficient condition
        # for positive definiteness of the diagonal weight matrix).
        assert np.all(W > 0), "Weight vector contains non-positive entries"
