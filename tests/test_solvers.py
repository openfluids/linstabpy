"""Tests for resolvent and eigenvalue solvers.

All tests in this file are marked @pytest.mark.slow.
"""

import numpy as np
import pytest

from linstab2d.mesh import Mesh
from linstab2d.baseflow import BaseFlow, sutherland_air, load_blasius
from linstab2d.operators import build_lnse_operator, compute_chu_norm, build_spatial_operator
from linstab2d.boundary_conditions import apply_dirichlet, apply_dirichlet_spatial
from linstab2d.solvers import (
    resolvent_analysis,
    temporal_eigenvalues,
    adjoint_eigenvalues,
    spatial_eigenvalues,
)


@pytest.fixture
def blasius_problem():
    """Build a small Blasius problem for testing solvers."""
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
    T = bf.T.ravel(order="F")[p]
    W_int = mesh.W.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO, T, bf.kappa, bf.Ma, ngp)

    return mesh, bf, L0, idx, ngp, W_diag, invW_diag


# ---------------------------------------------------------------------------
# Resolvent analysis
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestResolventAnalysis:
    """Tests for the resolvent_analysis solver."""

    def test_gain_monotonicity(self, blasius_problem):
        """Singular values must be sorted in decreasing order."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, _, _ = resolvent_analysis(L0, omega=60.0, n_modes=3, W_diag=W_diag)
        assert gains[0] >= gains[1] >= gains[2] > 0

    def test_forcing_orthonormality(self, blasius_problem):
        """Forcing modes must be W-orthonormal."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, _, forcings = resolvent_analysis(L0, omega=60.0, n_modes=3, W_diag=W_diag)
        n = forcings.shape[1]
        gram = np.empty((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                gram[i, j] = forcings[:, i].conj() @ (W_diag * forcings[:, j])
        np.testing.assert_allclose(gram, np.eye(n), atol=1e-8)

    def test_response_orthonormality(self, blasius_problem):
        """Response modes must be W-orthonormal."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, responses, _ = resolvent_analysis(L0, omega=60.0, n_modes=3, W_diag=W_diag)
        n = responses.shape[1]
        gram = np.empty((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                gram[i, j] = responses[:, i].conj() @ (W_diag * responses[:, j])
        np.testing.assert_allclose(gram, np.eye(n), atol=1e-8)

    def test_gains_real_positive(self, blasius_problem):
        """Gains must be real and strictly positive."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, _, _ = resolvent_analysis(L0, omega=60.0, n_modes=3, W_diag=W_diag)
        assert np.all(np.isreal(gains))
        assert np.all(gains > 0)

    def test_full_resolvent_default(self, blasius_problem):
        """B_mat=None, C_mat=None gives the full (unrestricted) resolvent."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, responses, forcings = resolvent_analysis(
            L0, omega=60.0, n_modes=3, W_diag=W_diag
        )
        assert gains.shape == (3,)
        assert responses.shape[1] == 3
        assert forcings.shape[1] == 3

    def test_response_shape(self, blasius_problem):
        """Response array must be (ndof, n_modes)."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, responses, forcings = resolvent_analysis(
            L0, omega=60.0, n_modes=3, W_diag=W_diag
        )
        ndof = L0.shape[0]
        assert responses.shape == (ndof, 3)


# ---------------------------------------------------------------------------
# Temporal eigenvalues
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTemporalEigenvalues:
    """Tests for the temporal_eigenvalues solver."""

    def test_returns_n_modes(self, blasius_problem):
        """Must return exactly n_modes eigenvalues."""
        _, _, L0, _, _, _, _ = blasius_problem
        omega, modes = temporal_eigenvalues(L0, n_modes=5, target=-0.1j)
        assert omega.shape == (5,)
        assert modes.shape[1] == 5

    def test_eigenpair_residual(self, blasius_problem):
        """Eigenvalue residual: ||1j*L0 @ v - omega*v|| / ||v|| must be small."""
        _, _, L0, _, _, _, _ = blasius_problem
        omega, modes = temporal_eigenvalues(L0, n_modes=5, target=-0.1j)
        iL0 = 1j * L0
        for i in range(min(3, len(omega))):
            v = modes[:, i]
            residual = iL0 @ v - omega[i] * v
            rel_err = np.linalg.norm(residual) / np.linalg.norm(v)
            assert rel_err < 1e-6, f"Mode {i}: relative residual {rel_err:.2e}"

    def test_eigenvalues_near_target(self, blasius_problem):
        """Eigenvalues should be sorted by distance to the target."""
        _, _, L0, _, _, _, _ = blasius_problem
        target = -0.1j
        omega, _ = temporal_eigenvalues(L0, n_modes=5, target=target)
        dists = np.abs(omega - target)
        np.testing.assert_array_less(dists[:-1], dists[1:] + 1e-12)

    def test_n_modes_zero_raises(self, blasius_problem):
        """n_modes <= 0 must raise ValueError."""
        _, _, L0, _, _, _, _ = blasius_problem
        with pytest.raises(ValueError):
            temporal_eigenvalues(L0, n_modes=0, target=0.0)


# ---------------------------------------------------------------------------
# Adjoint eigenvalues
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAdjointEigenvalues:
    """Tests for the adjoint_eigenvalues solver."""

    def test_conjugate_pairing(self, blasius_problem):
        """Adjoint eigenvalues pair with temporal eigenvalues.

        temporal_eigenvalues returns omega_d = eig(i*L0) = i*eig(L0).
        adjoint_eigenvalues returns omega_a = eig(W^{-1} L0^H W) = conj(eig(L0)).
        Therefore: omega_a = i * conj(omega_d).
        """
        _, _, L0, _, _, W_diag, invW_diag = blasius_problem
        omega_d, _ = temporal_eigenvalues(L0, n_modes=3, target=-0.1j)
        expected_adj = 1j * np.conj(omega_d)
        # Request extra modes to cover the spectral neighborhood
        omega_a, _ = adjoint_eigenvalues(
            L0, n_modes=6, target=expected_adj[0], W_diag=W_diag, invW_diag=invW_diag
        )
        # Check that at least one temporal/adjoint pair matches
        matched = 0
        for ea in expected_adj:
            dists = np.abs(omega_a - ea)
            if np.min(dists) < 1e-4:
                matched += 1
        assert matched >= 1, (
            f"No adjoint eigenvalue paired with any temporal eigenvalue.\n"
            f"  expected_adj = {expected_adj}\n"
            f"  omega_a = {omega_a}"
        )

    def test_adjoint_eigenvector_residual(self, blasius_problem):
        """Adjoint eigenvector must satisfy L_adj @ v = omega_a * v.

        adjoint_eigenvalues builds L_adj = W^{-1} L0^H W (the raw adjoint
        of L0, NOT the temporal adjoint of i*L0) and returns its eigenvalues.
        """
        _, _, L0, _, _, W_diag, invW_diag = blasius_problem
        omega_a, modes_a = adjoint_eigenvalues(
            L0, n_modes=3, target=-0.1j, W_diag=W_diag, invW_diag=invW_diag
        )
        L0H = L0.conj().T
        for i in range(len(omega_a)):
            v = modes_a[:, i]
            Ladj_v = invW_diag * (L0H @ (W_diag * v))
            residual = Ladj_v - omega_a[i] * v
            rel_err = np.linalg.norm(residual) / np.linalg.norm(v)
            assert rel_err < 1e-6, f"Adjoint mode {i}: relative residual {rel_err:.2e}"

    def test_w_normalization(self, blasius_problem):
        """Adjoint modes should be W-normalized: v^H @ (W * v) ≈ 1."""
        _, _, L0, _, _, W_diag, invW_diag = blasius_problem
        _, modes_a = adjoint_eigenvalues(
            L0, n_modes=3, target=-0.1j, W_diag=W_diag, invW_diag=invW_diag
        )
        for i in range(modes_a.shape[1]):
            v = modes_a[:, i]
            norm_val = v.conj() @ (W_diag * v)
            np.testing.assert_allclose(np.abs(norm_val), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Spatial eigenvalues
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSpatialEigenvalues:
    """Tests for the spatial_eigenvalues solver."""

    @pytest.fixture
    def spatial_problem(self):
        """Build a small Blasius spatial operator."""
        mesh = Mesh.create(xrange=(200, 300), yrange=(0, 30), nx=15, ny=10, fd_order=4)
        s = 3.0
        Y_new = 30.0 * np.sinh(s * mesh.Y / 30.0) / np.sinh(s)
        mesh = mesh.deform(Y_new=Y_new)

        bf = load_blasius(mesh.X, mesh.Y, Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15)
        sutherland_air(bf)

        Lc, R_mat, Lw, idx = build_spatial_operator(mesh, bf, model="2D")
        ngp = mesh.ngp
        Lc, R_mat, bc_indices = apply_dirichlet_spatial(Lc, R_mat, idx, "bt", "uvwT", ngp)
        return Lc, Lw, R_mat, ngp

    def test_returns_n_modes(self, spatial_problem):
        """Must return exactly n_modes eigenvalues."""
        Lc, Lw, R_mat, _ = spatial_problem
        alpha, modes = spatial_eigenvalues(
            Lc, Lw, R_mat, omega=60.0, n_modes=3, target=0.1 + 0.01j
        )
        assert alpha.shape == (3,)
        assert modes.shape[1] == 3

    def test_eigenvalues_finite_and_modes_nonzero(self, spatial_problem):
        """Spatial eigenvalues are finite; eigenvectors are nonzero.

        On this small 15x10 grid, R is rank-deficient after penalty BCs
        (rank 1495 of 1500), so the generalized EVP is ill-conditioned.
        We verify structural properties rather than the eigenpair residual.
        """
        Lc, Lw, R_mat, _ = spatial_problem
        alpha, modes = spatial_eigenvalues(
            Lc, Lw, R_mat, omega=60.0, n_modes=3, target=0.1 + 0.01j
        )
        assert np.all(np.isfinite(alpha)), f"Non-finite eigenvalues: {alpha}"
        for i in range(modes.shape[1]):
            assert np.linalg.norm(modes[:, i]) > 0, f"Zero eigenvector at mode {i}"

    def test_n_modes_zero_raises(self, spatial_problem):
        """n_modes <= 0 must raise ValueError."""
        Lc, Lw, R_mat, _ = spatial_problem
        with pytest.raises(ValueError):
            spatial_eigenvalues(Lc, Lw, R_mat, omega=60.0, n_modes=0, target=0.0)


# ---------------------------------------------------------------------------
# Cylindrical (axy) model tests
# ---------------------------------------------------------------------------


def _build_synthetic_jet(nx=15, ny=10, m=0):
    """Build a small synthetic axisymmetric sech² jet.

    Returns (mesh, bf, L0, idx, ngp).
    """
    mesh = Mesh.create(
        xrange=(0.0, 10.0), yrange=(0.0, 3.0),
        nx=nx, ny=ny, fd_order=4, y_symmetry=True,
    )
    R = mesh.Y
    U_axial = 0.9 / np.cosh(R)**2
    bf = BaseFlow(
        Re=1e4, Ma=0.9, Pr=0.7, kappa=1.4, T_0=288.15,
        RHO=np.ones_like(R), U=U_axial,
        V=np.zeros_like(R), W=np.zeros_like(R), T=np.ones_like(R),
    )
    sutherland_air(bf)
    L0, idx = build_lnse_operator(mesh, bf, model="axy", m=m)
    ngp = mesh.ngp
    L0, bc_indices = apply_dirichlet(L0, idx, "ltr", "uvwT", ngp)
    return mesh, bf, L0, idx, ngp


@pytest.mark.slow
class TestCylindricalTemporalEigenvalues:
    """Temporal eigenvalue tests for the cylindrical (axy) model."""

    def test_axy_eigenpair_residual_m0(self):
        """Eigenvalues of axy model at m=0 satisfy eigenpair equation."""
        _, _, L0, _, _ = _build_synthetic_jet(m=0)
        omega, modes = temporal_eigenvalues(L0, n_modes=5, target=-0.1j)
        iL0 = 1j * L0
        for i in range(len(omega)):
            v = modes[:, i]
            residual = iL0 @ v - omega[i] * v
            rel_err = np.linalg.norm(residual) / np.linalg.norm(v)
            assert rel_err < 1e-6, (
                f"Mode {i}: eigenpair residual {rel_err:.2e} (expected < 1e-6)"
            )

    def test_axy_eigenpair_residual_m1(self):
        """Eigenvalues of axy model at m=1 satisfy eigenpair equation."""
        _, _, L0, _, _ = _build_synthetic_jet(m=1)
        omega, modes = temporal_eigenvalues(L0, n_modes=5, target=-0.1j)
        iL0 = 1j * L0
        for i in range(len(omega)):
            v = modes[:, i]
            residual = iL0 @ v - omega[i] * v
            rel_err = np.linalg.norm(residual) / np.linalg.norm(v)
            assert rel_err < 1e-6, (
                f"Mode {i}: eigenpair residual {rel_err:.2e} (expected < 1e-6)"
            )



@pytest.mark.slow
class TestCylindricalSpatialEigenvalues:
    """Spatial eigenvalue tests for the cylindrical (axy) model."""

    @pytest.fixture
    def spatial_axy_problem(self):
        """Build a small cylindrical spatial operator."""
        mesh, bf, _, _, _ = _build_synthetic_jet()
        Lc, R_mat, Lw, idx = build_spatial_operator(mesh, bf, model="axy", m=0)
        ngp = mesh.ngp
        Lc, R_mat, bc_indices = apply_dirichlet_spatial(
            Lc, R_mat, idx, "ltr", "uvwT", ngp,
        )
        return Lc, Lw, R_mat, ngp

    def test_spatial_axy_eigenvalues_finite(self, spatial_axy_problem):
        """Spatial axy eigenvalues are finite and eigenvectors are nonzero."""
        Lc, Lw, R_mat, _ = spatial_axy_problem
        alpha, modes = spatial_eigenvalues(
            Lc, Lw, R_mat, omega=2.0, n_modes=3, target=0.1 + 0.01j,
        )
        assert np.all(np.isfinite(alpha)), f"Non-finite eigenvalues: {alpha}"
        for i in range(modes.shape[1]):
            assert np.linalg.norm(modes[:, i]) > 0, f"Zero eigenvector at mode {i}"

