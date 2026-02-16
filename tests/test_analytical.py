"""Analytical benchmark tests against classical hydrodynamic stability results.

These tests validate the entire linstab2d pipeline — coefficient computation,
operator assembly, inv(B) application, boundary conditions, and eigenvalue
solver — against known analytical/reference solutions from the literature.

For parallel flows (U = U(y) only), the 2D global solver reduces to the
equivalent 1D Orr-Sommerfeld problem via:

  - Periodic x-mesh (x_periodicity=True, small nx) — x-direction transparent
  - floquet_exp=α — streamwise wavenumber α imposed via Floquet shift on Dx
  - kx=0 — 2D perturbation (no spanwise variation)

Eigenvalue convention (verified empirically):

  ω_code = eig(i·L0) = -ω_OS

  where ω_OS = α·c is the Orr-Sommerfeld eigenvalue.  Growth rate is
  -Im(ω_code); unstable when Im(ω_code) < 0.

References
----------
Orszag, S. A. (1971). JFM 50:689–703.
Michalke, A. (1964). JFM 19:543–556.
Romanov, V. A. (1973). Functional Analysis & Applications 7:137–146.
Schmid, P. J. & Henningson, D. S. (2001). Springer.
"""

from __future__ import annotations

import numpy as np
import pytest

from linstab2d.mesh import Mesh
from linstab2d.baseflow import BaseFlow, sutherland_air
from linstab2d.operators import build_lnse_operator
from linstab2d.boundary_conditions import apply_dirichlet
from linstab2d.solvers import temporal_eigenvalues


# ---------------------------------------------------------------------------
# Helper: build parallel-flow LNSE problem
# ---------------------------------------------------------------------------

def _build_parallel_flow(
    U_func,
    yrange,
    ny,
    Re,
    Ma,
    alpha=1.0,
    beta=0.0,
    borders="bt",
    variables="uvwT",
    nx=8,
    fd_order=4,
    Pr=0.7,
    kappa=1.4,
    T_0=288.15,
):
    """Build a parallel-flow LNSE eigenvalue problem.

    Uses a periodic x-mesh with Floquet shift to reduce the 2D global
    problem to an equivalent 1D Orr-Sommerfeld problem.

    Parameters
    ----------
    U_func : callable
        y_array → U_array, streamwise velocity profile.
    yrange : tuple
        (y_min, y_max) domain bounds.
    ny : int
        Wall-normal grid points.
    Re, Ma : float
        Reynolds and Mach numbers.
    alpha : float
        Streamwise wavenumber (Floquet exponent).
    beta : float
        Spanwise wavenumber (kx parameter).
    borders : str
        BC border specification ('b'=bottom, 't'=top).
    variables : str
        BC variable specification.
    nx : int
        Periodic x-points (minimal, must be > fd_order).
    fd_order : int
        SBP-FD order.
    """
    mesh = Mesh.create(
        xrange=(0.0, 1.0),
        yrange=yrange,
        nx=nx,
        ny=ny,
        fd_order=fd_order,
        x_periodicity=True,
    )

    bf = BaseFlow(
        Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0,
        RHO=np.ones_like(mesh.Y),
        U=U_func(mesh.Y),
        V=np.zeros_like(mesh.Y),
        W=np.zeros_like(mesh.Y),
        T=np.ones_like(mesh.Y),
    )
    sutherland_air(bf)

    L0, idx = build_lnse_operator(mesh, bf, kx=beta, floquet_exp=alpha)
    ngp = mesh.ngp
    L0, bc_indices = apply_dirichlet(L0, idx, borders, variables, ngp)

    return {"mesh": mesh, "bf": bf, "L0": L0, "idx": idx, "ngp": ngp}


# ===========================================================================
# Test 1: Poiseuille Flow — Orszag (1971) benchmark
# ===========================================================================

class TestPoiseuilleFlow:
    """Plane Poiseuille flow: U(y) = 1 - y², y ∈ [-1, 1].

    Reference: Orszag (1971), JFM 50:689–703.
    At Re=10000, α=1: c = 0.23752649 + 0.00373967i (unstable TS wave).

    Convention: ω_code = -ω_OS, so the reference eigenvalue in code
    convention is -ORSZAG_OMEGA.
    """

    ORSZAG_OMEGA = 0.23752649 + 0.00373967j  # O-S eigenvalue
    ORSZAG_CODE = -ORSZAG_OMEGA               # code convention

    @staticmethod
    def _poiseuille(y):
        return 1.0 - y**2

    @pytest.mark.slow
    def test_orszag_eigenvalue(self):
        """Least-stable eigenvalue matches Orszag (1971) to ~1%."""
        d = _build_parallel_flow(
            self._poiseuille, yrange=(-1, 1), ny=128,
            Re=10000, Ma=0.01, alpha=1.0,
        )

        omegas, _ = temporal_eigenvalues(
            d["L0"], n_modes=10, target=self.ORSZAG_CODE,
        )

        # Find closest eigenvalue to Orszag reference (in code convention)
        dists = np.abs(omegas - self.ORSZAG_CODE)
        best = omegas[np.argmin(dists)]

        # Phase speed: c = -Re(ω_code)/α ≈ 0.2375  (within 1%)
        alpha = 1.0
        c_phase = -best.real / alpha
        np.testing.assert_allclose(
            c_phase, self.ORSZAG_OMEGA.real, rtol=0.01,
            err_msg=f"Phase speed mismatch: got {c_phase:.6f}, "
                    f"expected {self.ORSZAG_OMEGA.real:.6f}",
        )

        # Growth rate: Im(ω_code) < 0 → unstable TS wave
        assert best.imag < 0, (
            f"TS wave should be unstable (Im(ω_code) < 0), "
            f"got Im(ω_code) = {best.imag:.6e}"
        )

    @pytest.mark.slow
    def test_poiseuille_convergence(self):
        """Eigenvalue converges toward Orszag with grid refinement."""
        grids = [64, 128, 256]
        errors = []

        for ny in grids:
            d = _build_parallel_flow(
                self._poiseuille, yrange=(-1, 1), ny=ny,
                Re=10000, Ma=0.01, alpha=1.0,
            )
            omegas, _ = temporal_eigenvalues(
                d["L0"], n_modes=5, target=self.ORSZAG_CODE,
            )
            dists = np.abs(omegas - self.ORSZAG_CODE)
            errors.append(dists.min())

        # Monotonic convergence
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"Error not decreasing: ny={grids[i]} err={errors[i]:.4e}, "
                f"ny={grids[i+1]} err={errors[i+1]:.4e}"
            )

        # Convergence ratio ~4× between doubling grids (O(h²) for SBP)
        # Allow generous range [1.5, 100] to account for SBP boundary closures
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            assert ratio > 1.5, (
                f"Convergence ratio (ny={grids[i]}→{grids[i+1]}) = "
                f"{ratio:.2f} < 1.5 (expected > 1.5 for O(h²))"
            )

    @pytest.mark.slow
    def test_poiseuille_stable_below_critical(self):
        """Below critical Re ≈ 5772, the TS wave is stable."""
        d = _build_parallel_flow(
            self._poiseuille, yrange=(-1, 1), ny=128,
            Re=5000, Ma=0.01, alpha=1.0,
        )

        # Search near the Orszag phase speed (same Re(ω_code) region)
        target = self.ORSZAG_CODE.real + 0.01j  # slightly positive Im = stable
        omegas, _ = temporal_eigenvalues(
            d["L0"], n_modes=5, target=target,
        )

        # Mode closest to Orszag phase speed should be stable:
        # Im(ω_code) > 0 in code convention
        dists = np.abs(omegas.real - self.ORSZAG_CODE.real)
        closest = omegas[np.argmin(dists)]

        assert closest.imag > 0, (
            f"At Re=5000 < Re_c≈5772, TS wave should be stable "
            f"(Im(ω_code) > 0), got Im(ω_code) = {closest.imag:.6e}"
        )


# ===========================================================================
# Test 2: Couette Flow — all modes stable
# ===========================================================================

class TestCouetteFlow:
    """Plane Couette flow: U(y) = y, y ∈ [-1, 1].

    Linearly stable for all Re (Romanov 1973, Davey 1973).
    In code convention: all modes have Im(ω_code) > 0.
    """

    @staticmethod
    def _couette(y):
        return y

    @pytest.mark.slow
    def test_couette_all_stable(self):
        """All eigenvalues of Couette flow are damped."""
        d = _build_parallel_flow(
            self._couette, yrange=(-1, 1), ny=80,
            Re=2000, Ma=0.01, alpha=1.0,
        )

        # Search near the center of the spectrum with small positive Im
        omegas, _ = temporal_eigenvalues(
            d["L0"], n_modes=10, target=0.0 + 0.1j,
        )

        # ALL modes must be stable: Im(ω_code) > 0 in code convention
        # Allow small numerical noise (solver tolerance is 1e-10)
        assert np.all(omegas.imag > -1e-8), (
            f"Couette flow should be linearly stable for all Re. "
            f"Unstable eigenvalue(s) found with Im(ω_code) < -1e-8: "
            f"{omegas[omegas.imag < -1e-8]}"
        )


# ===========================================================================
# Test 3: Kelvin-Helmholtz Mixing Layer
# ===========================================================================

class TestMixingLayerKH:
    """Hyperbolic-tangent mixing layer: U(y) = 0.5·(1 + tanh(2y)).

    Reference: Michalke (1964), JFM 19:543–556.
    Inviscid tanh profile is unstable for 0 < α < 1, stable for α > 1.
    Maximum growth rate near α ≈ 0.4446.

    Code convention: unstable modes have Im(ω_code) < 0.
    Phase speed of KH mode ≈ 0.5 (mean velocity at inflection point).
    """

    @staticmethod
    def _tanh_profile(y):
        return 0.5 * (1.0 + np.tanh(2.0 * y))

    @pytest.mark.slow
    def test_kh_instability_exists(self):
        """KH instability exists at α=0.4 (below cutoff α=1)."""
        d = _build_parallel_flow(
            self._tanh_profile, yrange=(-10, 10), ny=100,
            Re=1000, Ma=0.01, alpha=0.4,
        )

        # Phase speed ≈ 0.5, so ω_OS_r ≈ 0.4·0.5 = 0.2
        # Code convention: ω_code_r ≈ -0.2, Im < 0 (unstable)
        omegas, _ = temporal_eigenvalues(
            d["L0"], n_modes=5, target=-0.2 - 0.05j,
        )

        # At least one unstable mode: Im(ω_code) < 0
        growth_rates = -omegas.imag  # O-S growth rate = -Im(ω_code)
        assert np.any(growth_rates > 0), (
            f"KH instability expected at α=0.4 but no unstable mode found. "
            f"Growth rates (O-S): {growth_rates}"
        )

        # Growth rate should be O(0.1), not spuriously small or large
        max_growth = growth_rates.max()
        assert max_growth > 0.01, (
            f"KH growth rate suspiciously small: {max_growth:.4e}"
        )
        assert max_growth < 1.0, (
            f"KH growth rate suspiciously large: {max_growth:.4e}"
        )

    @pytest.mark.slow
    def test_kh_phase_speed(self):
        """KH mode phase speed ≈ 0.5 (mean velocity at inflection)."""
        d = _build_parallel_flow(
            self._tanh_profile, yrange=(-10, 10), ny=100,
            Re=1000, Ma=0.01, alpha=0.4,
        )

        omegas, _ = temporal_eigenvalues(
            d["L0"], n_modes=5, target=-0.2 - 0.05j,
        )

        # Find the most unstable mode (most negative Im)
        idx_unstable = np.argmin(omegas.imag)
        omega_kh = omegas[idx_unstable]

        # Phase speed c = -Re(ω_code)/α (code convention)
        alpha = 0.4
        c_phase = -omega_kh.real / alpha

        np.testing.assert_allclose(
            c_phase, 0.5, atol=0.1,
            err_msg=f"KH phase speed should be ≈ 0.5, got {c_phase:.4f}",
        )

    @pytest.mark.slow
    def test_kh_stable_above_cutoff(self):
        """At α=1.5 > 1, the tanh mixing layer is stable."""
        d = _build_parallel_flow(
            self._tanh_profile, yrange=(-10, 10), ny=100,
            Re=1000, Ma=0.01, alpha=1.5,
        )

        # Phase speed ≈ 0.5, ω_OS_r ≈ 1.5·0.5 = 0.75
        # Code: ω_code_r ≈ -0.75, Im > 0 (stable)
        omegas, _ = temporal_eigenvalues(
            d["L0"], n_modes=5, target=-0.75 + 0.1j,
        )

        # All modes should be stable: Im(ω_code) > 0
        assert np.all(omegas.imag > -1e-6), (
            f"Mixing layer should be stable at α=1.5 > 1. "
            f"Unstable eigenvalue(s): {omegas[omegas.imag < -1e-6]}"
        )


# ===========================================================================
# Test 4: Uniform Flow — analytical dispersion
# ===========================================================================

class TestUniformFlowEigenvalues:
    """Uniform base flow: U=1, V=W=0, T=RHO=1.

    No instability mechanism; eigenvalues follow the compressible
    dispersion relation with discrete y-mode wavenumbers.

    In code convention, convective modes cluster near ω_code ≈ -α·U
    (since ω_OS ≈ α·U and ω_code = -ω_OS).
    """

    @staticmethod
    def _uniform(y):
        return np.ones_like(y)

    @pytest.mark.slow
    def test_convective_cluster(self):
        """Vorticity/entropy modes cluster near ω_code ≈ -α·U = -2."""
        d = _build_parallel_flow(
            self._uniform, yrange=(0, 1), ny=40,
            Re=1e6, Ma=0.5, alpha=2.0,
        )

        alpha_U = 2.0  # α * U
        # Code convention: modes near ω_code ≈ -α·U = -2
        omegas, _ = temporal_eigenvalues(
            d["L0"], n_modes=10, target=-alpha_U,
        )

        # Count modes with Re(ω_code) within 20% of -α·U
        near_convective = np.abs(omegas.real + alpha_U) < 0.2 * alpha_U
        n_near = np.sum(near_convective)

        assert n_near >= 3, (
            f"Expected ≥ 3 convective modes near Re(ω_code) ≈ {-alpha_U}, "
            f"found {n_near}. Eigenvalue real parts: {omegas.real}"
        )

    @pytest.mark.slow
    def test_all_damped(self):
        """Uniform flow has no instability mechanism."""
        d = _build_parallel_flow(
            self._uniform, yrange=(0, 1), ny=40,
            Re=1e6, Ma=0.5, alpha=2.0,
        )

        omegas, _ = temporal_eigenvalues(
            d["L0"], n_modes=10, target=-2.0,
        )

        # All modes must be stable: Im(ω_code) > 0 (or neutral within tolerance)
        assert np.all(omegas.imag > -1e-6), (
            f"Uniform flow should have no unstable modes. "
            f"Most negative Im(ω_code) = {omegas.imag.min():.6e}"
        )
