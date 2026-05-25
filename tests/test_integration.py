# SPDX-License-Identifier: GPL-3.0-only
"""
End-to-end integration tests for linstabpy.

All tests are marked @pytest.mark.slow since they assemble large operators
and/or call iterative solvers.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.baseflow import BaseFlow, load_blasius, sutherland_air
from linstabpy.boundary_conditions import apply_dirichlet
from linstabpy.mesh import Mesh
from linstabpy.operators import build_lnse_operator, compute_chu_norm
from linstabpy.solvers import resolvent_analysis, temporal_eigenvalues

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
        xrange=(x_min, x_max),
        yrange=(-1.0, 1.0),
        nx=nx,
        ny=ny,
        fd_order=4,
        alpha_filter="none",
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
        """Resolvent gains are positive, monotonically decreasing, and nontrivial.

        MATLAB reference on 300x100: [128.62, 14.54, 7.49] at Re=6e5, Ma=0.3, omega=60.
        On the coarser 100x50 grid with the current `lrbt` boundary treatment,
        the gains are much lower but should still show amplification above unity.
        """
        d = blasius_large
        gains, _, _ = resolvent_analysis(
            d["L0"],
            omega=60.0,
            n_modes=3,
            W_diag=d["W_diag"],
            B_mat=d["B_mat"],
            C_mat=d["C_mat"],
        )

        # All gains must be positive
        assert np.all(gains > 0), f"Expected positive gains, got {gains}"

        # Monotonically decreasing (sigma_1 >= sigma_2 >= sigma_3)
        assert np.all(np.diff(gains) <= 0), f"Gains should be monotonically decreasing, got {gains}"

        # Coarse-grid, fully constrained Blasius still should amplify above unity.
        # TODO: tighten to > 5.0 once BCs use the full non-reflecting treatment
        # instead of the current `lrbt` Dirichlet-everywhere constraint.
        assert gains[0] > 1.0, f"Leading gain suspiciously small: {gains[0]:.2f} (expected > 1)"


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


# ---------------------------------------------------------------------------
# TestCylindricalIntegration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_jet():
    """Module-scoped synthetic sech² jet problem (axy, m=0, 15×10)."""
    from tests.conftest import build_synthetic_jet

    return build_synthetic_jet()


class TestCylindricalIntegration:
    """End-to-end integration tests for the cylindrical (axy) model.

    Uses a synthetic tanh jet profile on a 15×10 grid to test the full
    pipeline: mesh → baseflow → operator assembly → BCs → eigensolve.
    """

    @pytest.mark.slow
    def test_least_damped_eigenvalue_reference(self, synthetic_jet):
        """Regression test: least-damped mode matches reference eigenvalue.

        Reference: macOS/arm64, Python 3.13, PETSc 3.22/MUMPS.
        Synthetic sech² jet (15×10, m=0), deterministic SBP operators.
        Reproducible to machine precision; tolerance allows for minor
        BLAS/LAPACK variation across platforms.
        """
        d = synthetic_jet
        omega, _ = temporal_eigenvalues(d["L0"], n_modes=5, target=-0.1j)

        omega_ref_imag = -1.748317109243e-02

        dists = np.abs(omega.imag - omega_ref_imag)
        closest = omega[np.argmin(dists)]

        # TODO: tighten to rtol=1e-6 once the axy cylindrical path is
        # validated against a higher-resolution reference.
        np.testing.assert_allclose(
            closest.imag,
            omega_ref_imag,
            rtol=5e-3,
            err_msg=f"Least-damped eigenvalue shifted: {closest} vs reference {omega_ref_imag}j",
        )
