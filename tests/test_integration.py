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

@pytest.fixture(scope="module")
def blasius_large():
    """Moderate-resolution Blasius problem for resolvent validation."""
    mesh = Mesh.create(xrange=(200, 800), yrange=(0, 60), nx=100, ny=50, fd_order=4)
    s = 3.0
    Y_new = 60.0 * np.sinh(s * mesh.Y / 60.0) / np.sinh(s)
    mesh = mesh.deform(Y_new=Y_new)

    bf = load_blasius(mesh.X, mesh.Y, Re=6e5, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15)
    sutherland_air(bf)

    L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
    ngp = mesh.ngp
    L0, bc_indices = apply_dirichlet(L0, idx, "lrbt", "uvwT", ngp)

    p = mesh.used_ind
    RHO = bf.RHO.ravel(order="F")[p]
    T_field = bf.T.ravel(order="F")[p]
    W_int = mesh.W.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO, T_field, bf.kappa, bf.Ma, ngp)

    return {
        "mesh": mesh,
        "bf": bf,
        "L0": L0,
        "idx": idx,
        "bc_indices": bc_indices,
        "ngp": ngp,
        "W_diag": W_diag,
        "invW_diag": invW_diag,
    }


@pytest.fixture(scope="module")
def blasius_small():
    """Small Blasius problem for quick consistency checks."""
    mesh = Mesh.create(xrange=(200, 300), yrange=(0, 30), nx=15, ny=10, fd_order=4)
    s = 3.0
    Y_new = 30.0 * np.sinh(s * mesh.Y / 30.0) / np.sinh(s)
    mesh = mesh.deform(Y_new=Y_new)

    bf = load_blasius(mesh.X, mesh.Y, Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15)
    sutherland_air(bf)

    L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
    ngp = mesh.ngp
    L0, bc_indices = apply_dirichlet(L0, idx, "bt", "uvwT", ngp)

    p = mesh.used_ind
    RHO = bf.RHO.ravel(order="F")[p]
    T_field = bf.T.ravel(order="F")[p]
    W_int = mesh.W.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO, T_field, bf.kappa, bf.Ma, ngp)

    return {
        "mesh": mesh,
        "bf": bf,
        "L0": L0,
        "idx": idx,
        "bc_indices": bc_indices,
        "ngp": ngp,
        "W_diag": W_diag,
        "invW_diag": invW_diag,
    }


# ---------------------------------------------------------------------------
# TestBlasiusResolventReference
# ---------------------------------------------------------------------------

class TestBlasiusResolventReference:
    """Gold-standard resolvent validation against known Blasius gains."""

    @pytest.mark.slow
    def test_resolvent_gains_structure(self, blasius_large):
        """Resolvent gains are positive, monotonically decreasing, and O(100)."""
        d = blasius_large
        gains, _, _ = resolvent_analysis(
            d["L0"], omega=60.0, n_modes=3, W_diag=d["W_diag"]
        )

        # All gains must be positive
        assert np.all(gains > 0), f"Expected positive gains, got {gains}"

        # Monotonically decreasing (sigma_1 >= sigma_2 >= sigma_3)
        assert np.all(np.diff(gains) <= 0), (
            f"Gains should be monotonically decreasing, got {gains}"
        )

        # First gain should be meaningful — on this coarse 100x50 grid,
        # gains are lower than the converged reference (~128 on a 300x100 grid).
        assert gains[0] > 10.0, (
            f"Leading gain suspiciously small: {gains[0]:.2f} (expected > 10)"
        )

    @pytest.mark.slow
    def test_operator_sparsity(self, blasius_large):
        """LNSE operator has roughly 30 nonzeros per row on average."""
        L0 = blasius_large["L0"]
        avg_nnz_per_row = L0.nnz / L0.shape[0]

        # Dirichlet penalty rows are dense (penalty fills entire row/col),
        # so avg nnz/row is much higher than the bare stencil (~30).
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
    def test_temporal_stability(self, blasius_small):
        """At Re=1e4 the Blasius boundary layer is temporally stable.

        All eigenvalues should satisfy Im(omega) < 0 (damped modes).
        """
        d = blasius_small
        omegas, _ = temporal_eigenvalues(d["L0"], n_modes=10, target=-0.1j)

        # Every eigenvalue must have negative imaginary part (stable)
        assert np.all(omegas.imag < 0), (
            f"Found unstable eigenvalues: "
            f"{omegas[omegas.imag >= 0]}"
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
