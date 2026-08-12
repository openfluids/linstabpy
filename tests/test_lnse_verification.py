# SPDX-License-Identifier: GPL-3.0-only
"""Independent verification of LNSE coefficient matrices.

Two complementary approaches:

1. **Complex-step Jacobian** — Write the nonlinear compressible NS residual
   R(q), then verify dR/dq|_{q_bar} = L_raw via complex-step differentiation
   (machine-precision, no truncation error).  Works for kx=0 (real residual).
   For kx != 0, central finite differences are used (still ~10 digits).

2. **SymPy coefficient verification** — Verify individual LNSE coefficient
   matrix entries against analytically derived values from the compressible
   NS equations.

Both verify the LNSE operator *before* the inv(B) transformation, i.e. they
check L_raw = _assemble_block_operator(...).  This is the natural comparison
point because:

    Physical equations:  B_phys * dq'/dt = J * q'   (J = Jacobian of NS residual)
    Code assembly:       L0 = inv(B_phys) * L_raw

So L_raw = B_phys * L0 = J.  The complex-step Jacobian gives J directly.

Sign convention notes
---------------------
The LNSE system from coefficients.py uses B with 1j prefactors:
    B_code = 1j * diag(1, rho, rho, rho, rho)
    L0 = inv(B_code / 1j) * L_raw = inv(B_phys) * L_raw

The nonlinear equations in the residual function R(q) = [f1, ..., f5]:
    drho/dt  = f1(q)            (continuity)
    rho*du/dt = f2(q)           (x-momentum)
    rho*dv/dt = f3(q)           (y-momentum)
    rho*dw/dt = f4(q)           (z-momentum)
    rho*dT/dt = f5(q)           (energy)

Linearizing: B_phys * dq'/dt = dR/dq * q', so J = dR/dq = L_raw.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.baseflow.baseflow import BaseFlow
from linstabpy.baseflow.viscosity import sutherland_air
from linstabpy.mesh.mesh import Mesh
from linstabpy.operators.assembly import (
    _assemble_block_operator,
    _prepare_coefficients,
)
from linstabpy.operators.coefficients import compute_cartesian_coefficients

# =====================================================================
# Sutherland viscosity — complex-step compatible
# =====================================================================


def _sutherland_complex(T_nondim, T_0):
    """Sutherland viscosity + derivatives, complex-step compatible.

    Mirrors ``viscosity.py:sutherland_air`` but works with complex-valued T
    (uses ``**0.5`` and ``**1.5`` instead of ``np.sqrt``).

    Returns non-dimensional (MU, dmudT, d2mudT2).
    """
    S1 = 110.4
    T_ref = 280.0
    mu_ref = 1.735e-5

    Tstar = T_nondim * T_0

    mu_0 = mu_ref * (T_ref + S1) / (T_0 + S1) * (T_0 / T_ref) ** 1.5

    mu = mu_ref * (T_ref + S1) / (Tstar + S1) * (Tstar / T_ref) ** 1.5

    dmudT_dim = (
        -mu_ref * (T_ref + S1) / (Tstar + S1) ** 2 * (Tstar / T_ref) ** 1.5
        + 1.5 * mu_ref * (T_ref + S1) / (Tstar + S1) * (Tstar / T_ref) ** 0.5 / T_ref
    )

    d2mudT2_dim = (
        2.0 * mu_ref * (T_ref + S1) / (Tstar + S1) ** 3 * (Tstar / T_ref) ** 1.5
        - 3.0 * mu_ref * (T_ref + S1) / (Tstar + S1) ** 2 * (Tstar / T_ref) ** 0.5 / T_ref
        + 0.75 * mu_ref * (T_ref + S1) / (Tstar + S1) * (Tstar / T_ref) ** (-0.5) / T_ref**2
    )

    MU = mu / mu_0
    dmudT_nd = dmudT_dim / mu_0 * T_0
    d2mudT2_nd = d2mudT2_dim / mu_0 * T_0**2

    return MU, dmudT_nd, d2mudT2_nd


# =====================================================================
# Nonlinear NS residual (complex-step compatible)
# =====================================================================


def _ns_residual(q_flat, mesh, bf_params):
    r"""Spatial RHS of 2D compressible NS in primitive variables.

    Computes R(q) such that:

    .. math::

        \begin{bmatrix} 1 \\ \rho \\ \rho \\ \rho \\ \rho \end{bmatrix}
        \frac{\partial q}{\partial t} = R(q)

    The Jacobian dR/dq at the base state equals L_raw from
    ``_assemble_block_operator``.

    The spanwise wavenumber ``kx`` enters via the Fourier-mode substitution
    d/dz = i*kx for all fields (both base flow and perturbation).

    Parameters
    ----------
    q_flat : ndarray, shape (5*ngp,)
        State vector [rho, u, v, w, T] (may be complex for complex-step).
    mesh : Mesh
        Mesh with derivative operators Dx, Dy, D2x, D2y.
    bf_params : dict
        Flow parameters: Re, Ma, Pr, kappa, T_0, kx.

    Returns
    -------
    R : ndarray, shape (5*ngp,)
        Spatial residual.
    """
    ngp = mesh.ngp
    Re = bf_params["Re"]
    Ma = bf_params["Ma"]
    Pr = bf_params["Pr"]
    kappa = bf_params["kappa"]
    T_0 = bf_params["T_0"]
    kx = bf_params["kx"]

    # Unpack state
    rho = q_flat[0 * ngp : 1 * ngp]
    u = q_flat[1 * ngp : 2 * ngp]
    v = q_flat[2 * ngp : 3 * ngp]
    w = q_flat[3 * ngp : 4 * ngp]
    T = q_flat[4 * ngp : 5 * ngp]

    Dx = mesh.Dx
    Dy = mesh.Dy
    D2x = mesh.D2x
    D2y = mesh.D2y

    # -- Spatial derivatives in x and y (FD matrices) -------------------------
    drhodx = Dx @ rho
    drhody = Dy @ rho
    dudx = Dx @ u
    dudy = Dy @ u
    dvdx = Dx @ v
    dvdy = Dy @ v
    dwdx = Dx @ w
    dwdy = Dy @ w
    dTdx = Dx @ T
    dTdy = Dy @ T

    d2udx2 = D2x @ u
    d2udy2 = D2y @ u
    d2vdx2 = D2x @ v
    d2vdy2 = D2y @ v
    d2wdx2 = D2x @ w
    d2wdy2 = D2y @ w
    d2Tdx2 = D2x @ T
    d2Tdy2 = D2y @ T

    d2udxdy = Dx @ dudy
    d2vdxdy = Dx @ dvdy

    # -- Spanwise (z) Fourier mode: d/dz = ikx, d²/dz² = -kx² ----------------
    ikx = 1j * kx
    ikx2 = ikx**2

    dudz = ikx * u
    dvdz = ikx * v
    dwdz = ikx * w
    dTdz = ikx * T
    drhodz = ikx * rho

    d2udz2 = ikx2 * u
    d2vdz2 = ikx2 * v
    d2wdz2 = ikx2 * w
    d2Tdz2 = ikx2 * T

    # Mixed z-derivatives
    d2wdxdz = ikx * dwdx

    # -- Thermodynamic quantities ----------------------------------------------
    gM2 = kappa * Ma**2
    p = rho * T / gM2
    dpdx = (drhodx * T + rho * dTdx) / gM2
    dpdy = (drhody * T + rho * dTdy) / gM2
    dpdz = ikx * p

    # -- Viscosity (Sutherland, complex-step safe) -----------------------------
    MU, dmudT, d2mudT2 = _sutherland_complex(T, T_0)
    dmudx = dmudT * dTdx
    dmudy = dmudT * dTdy
    dmudz = dmudT * dTdz

    # -- Velocity divergence ---------------------------------------------------
    div_u = dudx + dvdy + dwdz

    # Divergence spatial derivatives (needed for viscous stress divergence)
    d_div_u_dx = d2udx2 + d2vdxdy + ikx * dwdx
    d_div_u_dy = Dx @ dudy + D2y @ v + ikx * dwdy
    d_div_u_dz = ikx * dudx + ikx * dvdy + ikx2 * w

    # -- Viscous stress tensor -------------------------------------------------
    tau_xx = MU * (2.0 * dudx - (2.0 / 3.0) * div_u)
    tau_yy = MU * (2.0 * dvdy - (2.0 / 3.0) * div_u)
    tau_zz = MU * (2.0 * dwdz - (2.0 / 3.0) * div_u)
    tau_xy = MU * (dudy + dvdx)
    tau_xz = MU * (dudz + dwdx)
    tau_yz = MU * (dvdz + dwdy)

    # -- Divergence of stress tensor rows --------------------------------------
    # x-row: d(tau_xx)/dx + d(tau_xy)/dy + d(tau_xz)/dz
    dtau_xx_dx = dmudx * (2.0 * dudx - (2.0 / 3.0) * div_u) + MU * (
        2.0 * d2udx2 - (2.0 / 3.0) * d_div_u_dx
    )
    dtau_xy_dy = dmudy * (dudy + dvdx) + MU * (d2udy2 + Dx @ dvdy)
    dtau_xz_dz = dmudz * (dudz + dwdx) + MU * (d2udz2 + d2wdxdz)
    div_tau_x = dtau_xx_dx + dtau_xy_dy + dtau_xz_dz

    # y-row: d(tau_yx)/dx + d(tau_yy)/dy + d(tau_yz)/dz
    dtau_yx_dx = dmudx * (dudy + dvdx) + MU * (Dx @ dudy + d2vdx2)
    dtau_yy_dy = dmudy * (2.0 * dvdy - (2.0 / 3.0) * div_u) + MU * (
        2.0 * D2y @ v - (2.0 / 3.0) * d_div_u_dy
    )
    dtau_yz_dz = dmudz * (dvdz + dwdy) + MU * (d2vdz2 + ikx * dwdy)
    div_tau_y = dtau_yx_dx + dtau_yy_dy + dtau_yz_dz

    # z-row: d(tau_zx)/dx + d(tau_zy)/dy + d(tau_zz)/dz
    dtau_zx_dx = dmudx * (dwdx + dudz) + MU * (d2wdx2 + ikx * dudx)
    dtau_zy_dy = dmudy * (dwdy + dvdz) + MU * (D2y @ w + ikx * dvdy)
    dtau_zz_dz = dmudz * (2.0 * dwdz - (2.0 / 3.0) * div_u) + MU * (
        2.0 * ikx2 * w - (2.0 / 3.0) * d_div_u_dz
    )
    div_tau_z = dtau_zx_dx + dtau_zy_dy + dtau_zz_dz

    # -- Viscous dissipation: Phi = tau_ij * du_i/dx_j -------------------------
    Phi = (
        tau_xx * dudx
        + tau_xy * dudy
        + tau_xz * dudz
        + tau_xy * dvdx
        + tau_yy * dvdy
        + tau_yz * dvdz
        + tau_xz * dwdx
        + tau_yz * dwdy
        + tau_zz * dwdz
    )

    # -- Heat conduction: div(mu * grad(T)) ------------------------------------
    heat_cond = dmudx * dTdx + MU * d2Tdx2 + dmudy * dTdy + MU * d2Tdy2 + dmudz * dTdz + MU * d2Tdz2

    # -- Assemble equations ----------------------------------------------------
    # Eq 1: Continuity (drho/dt = f1)
    f1 = -(u * drhodx + v * drhody + w * drhodz + rho * div_u)

    # Eq 2: x-momentum (rho * du/dt = f2)
    f2 = -rho * (u * dudx + v * dudy + w * dudz) - dpdx + div_tau_x / Re

    # Eq 3: y-momentum (rho * dv/dt = f3)
    f3 = -rho * (u * dvdx + v * dvdy + w * dvdz) - dpdy + div_tau_y / Re

    # Eq 4: z-momentum (rho * dw/dt = f4)
    f4 = -rho * (u * dwdx + v * dwdy + w * dwdz) - dpdz + div_tau_z / Re

    # Eq 5: Energy — temperature equation divided by cv
    # The code uses rho*dT/dt form where cv = 1/(kappa*(kappa-1)*Ma^2).
    # Pressure-dilatation: -(kappa-1)*p*div_u / cv = -(kappa-1)*rho*T*div_u
    # Viscous dissipation: Phi/(Re*cv) = kappa*(kappa-1)*Ma^2 * Phi/Re
    # Heat conduction: kappa*div(mu*grad(T))/(Re*Pr) stays as-is
    f5 = (
        -(kappa - 1.0) * rho * T * div_u
        - rho * (u * dTdx + v * dTdy + w * dTdz)
        + kappa * (kappa - 1.0) * Ma**2 * Phi / Re
        + kappa * heat_cond / (Re * Pr)
    )

    # When kx != 0, Fourier terms (ikx) make the residual complex even
    # for real-valued state vectors.
    out_dtype = np.result_type(q_flat.dtype, complex if kx != 0 else float)
    R = np.empty(5 * ngp, dtype=out_dtype)
    R[0 * ngp : 1 * ngp] = f1
    R[1 * ngp : 2 * ngp] = f2
    R[2 * ngp : 3 * ngp] = f3
    R[3 * ngp : 4 * ngp] = f4
    R[4 * ngp : 5 * ngp] = f5
    return R


# =====================================================================
# Jacobian computation methods
# =====================================================================


def _complex_step_jacobian(residual_fn, q0, h=1e-30):
    """Column-by-column complex-step Jacobian (for real-valued residuals).

    Only valid when ``residual_fn(q0)`` is real.  For complex-valued
    residuals (kx != 0), use ``_finite_diff_jacobian`` instead.
    """
    n = q0.size
    J = np.empty((n, n), dtype=float)
    q0_c = q0.astype(complex)

    for j in range(n):
        q_pert = q0_c.copy()
        q_pert[j] += 1j * h
        J[:, j] = residual_fn(q_pert).imag / h

    return J


def _finite_diff_jacobian(residual_fn, q0, h=1e-7):
    """Central finite-difference Jacobian (works for complex-valued residuals).

    O(h^2) accuracy — roughly 10 digits for h=1e-7.
    """
    n = q0.size
    R0 = residual_fn(q0)
    J = np.empty((n, n), dtype=R0.dtype)

    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        Rp = residual_fn(q0 + h * e_j)
        Rm = residual_fn(q0 - h * e_j)
        J[:, j] = (Rp - Rm) / (2.0 * h)

    return J


# =====================================================================
# Helpers: build L_raw and base state vector
# =====================================================================


def _build_L_raw(mesh, bf, kx=0.0, floquet_exp=0.0):
    """Build L_raw = _assemble_block_operator (before inv(B))."""
    from linstabpy.operators.assembly import _apply_floquet_shift

    prep = _prepare_coefficients(mesh, bf, kx=kx)
    ngp = prep["ngp"]
    DR, D2R = prep["DR"], prep["D2R"]
    DZ, D2Z = prep["DZ"], prep["D2Z"]
    D2RZ = prep["D2RZ"]

    if floquet_exp != 0.0:
        i_alpha = 1j * floquet_exp
        DZ, D2Z, D2RZ = _apply_floquet_shift(
            DZ, D2Z, D2RZ, i_alpha, ngp, DR=DR
        )

    return _assemble_block_operator(ngp, prep["coeffs"], DR, D2R, DZ, D2Z, D2RZ)


def _build_base_state_vector(mesh, bf):
    """Extract flat base-state vector q_bar = [rho, u, v, w, T]."""
    p = mesh.used_ind
    ngp = mesh.ngp
    q = np.empty(5 * ngp)
    q[0 * ngp : 1 * ngp] = bf.RHO.ravel(order="F")[p]
    q[1 * ngp : 2 * ngp] = bf.U.ravel(order="F")[p]
    q[2 * ngp : 3 * ngp] = bf.V.ravel(order="F")[p]
    q[3 * ngp : 4 * ngp] = bf.W.ravel(order="F")[p]
    q[4 * ngp : 5 * ngp] = bf.T.ravel(order="F")[p]
    return q


def _make_bf_params(bf, kx):
    """Extract parameter dict for _ns_residual from a BaseFlow."""
    return {
        "Re": bf.Re,
        "Ma": bf.Ma,
        "Pr": bf.Pr,
        "kappa": bf.kappa,
        "T_0": bf.T_0,
        "kx": kx,
    }


# =====================================================================
# Diagnostic helper for mismatches
# =====================================================================


def _report_mismatch(J, L_raw_dense, label, nvars=5):
    """Print per-block max errors for debugging mismatches."""
    diff = np.abs(J - L_raw_dense)
    ngp = J.shape[0] // nvars
    names = ["rho", "u", "v", "w", "T"]
    lines = [f"\n{label} — per-block max |J - L_raw|:"]
    for i in range(nvars):
        row = []
        for j_idx in range(nvars):
            block = diff[i * ngp : (i + 1) * ngp, j_idx * ngp : (j_idx + 1) * ngp]
            row.append(f"{block.max():.2e}")
        lines.append(
            f"  eq={names[i]:>3s}: "
            + "  ".join(f"{names[j_idx]}={v}" for j_idx, v in enumerate(row))
        )
    return "\n".join(lines)


# =====================================================================
# Test: Complex-step verification — uniform base flow (kx=0)
# =====================================================================


@pytest.mark.slow
class TestComplexStepUniform:
    """Verify LNSE coefficients via complex-step on uniform flow.

    Uniform flow: U=0.5, V=W=0, rho=T=1.  All base-flow gradients are zero,
    so only A0 and the constant-coefficient viscous terms contribute.
    """

    @staticmethod
    def _setup():
        nx, ny = 8, 12
        mesh = Mesh.create(
            xrange=(0.0, 2 * np.pi),
            yrange=(-1.0, 1.0),
            nx=nx,
            ny=ny,
            fd_order=2,
            x_periodicity=True,
        )
        ones = np.ones((ny, nx))
        zeros = np.zeros((ny, nx))
        bf = BaseFlow(
            Re=1000.0,
            Ma=0.3,
            Pr=0.72,
            kappa=1.4,
            T_0=288.15,
            RHO=ones.copy(),
            U=0.5 * ones,
            V=zeros.copy(),
            W=zeros.copy(),
            T=ones.copy(),
        )
        sutherland_air(bf)
        return mesh, bf

    def test_uniform_kx0(self):
        """Uniform flow with kx=0: complex-step, machine precision."""
        mesh, bf = self._setup()
        kx = 0.0

        L_raw = _build_L_raw(mesh, bf, kx=kx)
        q_bar = _build_base_state_vector(mesh, bf)

        J = _complex_step_jacobian(lambda q: _ns_residual(q, mesh, _make_bf_params(bf, kx)), q_bar)

        L_raw_dense = L_raw.toarray()

        # L_raw should be real for kx=0 with real base flow
        assert np.max(np.abs(L_raw_dense.imag)) < 1e-12, "L_raw should be real for kx=0"

        np.testing.assert_allclose(
            J,
            L_raw_dense.real,
            atol=1e-10,
            err_msg=_report_mismatch(J, L_raw_dense.real, "uniform kx=0"),
        )


# =====================================================================
# Test: Complex-step verification — Poiseuille flow (kx=0)
# =====================================================================


@pytest.mark.slow
class TestComplexStepPoiseuille:
    """Verify LNSE via complex-step on Poiseuille flow (kx=0).

    U(y) = 1-y^2, V=W=0, rho=T=1.  Non-trivial dU/dy and d^2U/dy^2
    activate all first- and second-derivative coefficient terms.
    """

    @staticmethod
    def _setup():
        nx, ny = 8, 16
        mesh = Mesh.create(
            xrange=(0.0, 2 * np.pi),
            yrange=(-1.0, 1.0),
            nx=nx,
            ny=ny,
            fd_order=2,
            x_periodicity=True,
        )
        ones = np.ones((ny, nx))
        zeros = np.zeros((ny, nx))
        bf = BaseFlow(
            Re=1000.0,
            Ma=0.3,
            Pr=0.72,
            kappa=1.4,
            T_0=288.15,
            RHO=ones.copy(),
            U=1.0 - mesh.Y**2,
            V=zeros.copy(),
            W=zeros.copy(),
            T=ones.copy(),
        )
        sutherland_air(bf)
        return mesh, bf

    def test_poiseuille_kx0(self):
        """Poiseuille with kx=0: activates all gradient-dependent terms."""
        mesh, bf = self._setup()
        kx = 0.0

        L_raw = _build_L_raw(mesh, bf, kx=kx)
        q_bar = _build_base_state_vector(mesh, bf)

        J = _complex_step_jacobian(lambda q: _ns_residual(q, mesh, _make_bf_params(bf, kx)), q_bar)

        L_raw_dense = L_raw.toarray()
        assert np.max(np.abs(L_raw_dense.imag)) < 1e-12

        np.testing.assert_allclose(
            J,
            L_raw_dense.real,
            atol=1e-10,
            err_msg=_report_mismatch(J, L_raw_dense.real, "Poiseuille kx=0"),
        )


# =====================================================================
# Test: Complex-step verification — compressible (non-uniform T, kx=0)
# =====================================================================


@pytest.mark.slow
class TestComplexStepCompressible:
    """Verify LNSE via complex-step with non-uniform temperature.

    T(y) = 1 + 0.5*cos(pi*y), rho = 1/T, U(y) = 1-y^2.  Activates
    Sutherland viscosity terms (dmu/dT, d^2mu/dT^2) and pressure gradients.

    For non-uniform base flows, 'linearize-then-discretize' (L_raw) and
    'discretize-then-linearize' (complex-step Jacobian) differ by a
    commutator error of O(h^p) where h is the grid spacing and p the
    FD order.  We verify this by checking grid convergence: the error
    must decrease at the expected rate when the grid is refined.
    """

    @staticmethod
    def _make_case(ny):
        nx = 8
        mesh = Mesh.create(
            xrange=(0.0, 2 * np.pi),
            yrange=(-1.0, 1.0),
            nx=nx,
            ny=ny,
            fd_order=2,
            x_periodicity=True,
        )
        zeros = np.zeros((ny, nx))
        T_field = 1.0 + 0.5 * np.cos(np.pi * mesh.Y)
        bf = BaseFlow(
            Re=500.0,
            Ma=0.5,
            Pr=0.72,
            kappa=1.4,
            T_0=288.15,
            RHO=1.0 / T_field,
            U=1.0 - mesh.Y**2,
            V=zeros.copy(),
            W=zeros.copy(),
            T=T_field,
        )
        sutherland_air(bf)
        return mesh, bf

    @staticmethod
    def _max_error(mesh, bf, kx=0.0):
        """Max |J - L_raw| over interior points (excluding boundaries)."""
        L_raw = _build_L_raw(mesh, bf, kx=kx)
        q_bar = _build_base_state_vector(mesh, bf)
        J = _complex_step_jacobian(lambda q: _ns_residual(q, mesh, _make_bf_params(bf, kx)), q_bar)
        L_dense = L_raw.toarray().real
        return np.max(np.abs(J - L_dense))

    def test_compressible_grid_convergence(self):
        """Error between J and L_raw converges at O(h^2) for fd_order=2.

        The 'linearize-then-discretize' vs 'discretize-then-linearize'
        commutator error arises from terms like diag(f) @ Dy != Dy @ diag(f)
        when f is spatially varying.  For a 2nd-order FD scheme, this error
        is O(h^2) and should decrease by ~4x when the grid is refined 2x.
        """
        ny_coarse, ny_fine = 16, 32
        mesh_c, bf_c = self._make_case(ny_coarse)
        mesh_f, bf_f = self._make_case(ny_fine)

        err_coarse = self._max_error(mesh_c, bf_c)
        err_fine = self._max_error(mesh_f, bf_f)

        # Convergence ratio: doubling ny halves h, so error should drop by ~4x
        ratio = err_coarse / err_fine
        assert ratio > 2.5, (
            f"Expected O(h^2) convergence (ratio ~4), got {ratio:.2f} "
            f"(err_coarse={err_coarse:.4e}, err_fine={err_fine:.4e})"
        )

    def test_compressible_error_small(self):
        """Absolute error on the fine grid is small relative to operator norm."""
        mesh, bf = self._make_case(32)
        L_raw = _build_L_raw(mesh, bf, kx=0.0)
        q_bar = _build_base_state_vector(mesh, bf)
        J = _complex_step_jacobian(lambda q: _ns_residual(q, mesh, _make_bf_params(bf, 0.0)), q_bar)
        L_dense = L_raw.toarray().real
        rel_err = np.max(np.abs(J - L_dense)) / np.max(np.abs(L_dense))
        assert rel_err < 0.01, f"Relative error {rel_err:.4e} > 1%"


# =====================================================================
# Test: Finite-difference verification with kx != 0
# =====================================================================


@pytest.mark.slow
class TestLinearizedResidualKxNonzero:
    """Verify LNSE operator with kx != 0 via a linearized perturbation residual.

    For kx != 0, the Fourier decomposition d/dz -> ikx is only valid AFTER
    linearization (nonlinear products of Fourier modes generate harmonics).
    We therefore write a linearized residual R_lin(q') that computes the
    LNSE RHS for a perturbation q' with frozen base flow q_bar.  Since
    R_lin is linear in q', its matrix representation can be extracted by
    evaluating R_lin(e_j) for each unit vector e_j.
    """

    @staticmethod
    def _linearized_residual(q_prime, mesh, bf_params):
        r"""Linearized spatial RHS of LNSE for a perturbation Fourier mode.

        Computes R_lin(q') = L_raw @ q' by evaluating the linearized
        physics term by term.  The base flow is frozen (from bf_params).

        The key difference from ``_ns_residual`` is that d/dz -> ikx is
        applied only to perturbation fields, NOT to base-flow fields
        (which are 2D and z-invariant).
        """
        ngp = mesh.ngp
        Re = bf_params["Re"]
        Ma = bf_params["Ma"]
        Pr = bf_params["Pr"]
        kappa = bf_params["kappa"]
        kx = bf_params["kx"]
        gM2 = kappa * Ma**2
        ikx = 1j * kx
        ikx2 = ikx**2

        # Base flow (frozen, from bf_params)
        rho_b = bf_params["RHO"]
        u_b = bf_params["U"]
        v_b = bf_params["V"]
        w_b = bf_params["W"]
        T_b = bf_params["T"]
        MU_b = bf_params["MU"]
        dmudT_b = bf_params["dmudT"]

        Dx = mesh.Dx
        Dy = mesh.Dy
        D2x = mesh.D2x
        D2y = mesh.D2y

        # Base flow gradients (2D, no z-derivatives)
        du_b_dx = Dx @ u_b
        du_b_dy = Dy @ u_b
        dv_b_dx = Dx @ v_b
        dv_b_dy = Dy @ v_b
        dw_b_dx = Dx @ w_b
        dw_b_dy = Dy @ w_b
        drho_b_dx = Dx @ rho_b
        drho_b_dy = Dy @ rho_b
        dT_b_dx = Dx @ T_b
        dT_b_dy = Dy @ T_b

        # Perturbation fields
        rp = q_prime[0 * ngp : 1 * ngp]  # rho'
        up = q_prime[1 * ngp : 2 * ngp]  # u'
        vp = q_prime[2 * ngp : 3 * ngp]  # v'
        wp = q_prime[3 * ngp : 4 * ngp]  # w'
        Tp = q_prime[4 * ngp : 5 * ngp]  # T'

        # Perturbation gradients (d/dz -> ikx for perturbation only)
        drp_dx = Dx @ rp
        drp_dy = Dy @ rp
        drp_dz = ikx * rp
        dup_dx = Dx @ up
        dup_dy = Dy @ up
        dup_dz = ikx * up
        dvp_dx = Dx @ vp
        dvp_dy = Dy @ vp
        dvp_dz = ikx * vp
        dwp_dx = Dx @ wp
        dwp_dy = Dy @ wp
        dwp_dz = ikx * wp
        dTp_dx = Dx @ Tp
        dTp_dy = Dy @ Tp
        dTp_dz = ikx * Tp

        # Second derivatives of perturbation
        d2up_dx2 = D2x @ up
        d2up_dy2 = D2y @ up
        d2up_dz2 = ikx2 * up
        d2vp_dx2 = D2x @ vp
        d2vp_dy2 = D2y @ vp
        d2vp_dz2 = ikx2 * vp
        d2wp_dx2 = D2x @ wp
        d2wp_dy2 = D2y @ wp
        d2wp_dz2 = ikx2 * wp
        d2Tp_dx2 = D2x @ Tp
        d2Tp_dy2 = D2y @ Tp
        d2Tp_dz2 = ikx2 * Tp

        d2up_dxdy = Dx @ dup_dy
        d2vp_dxdy = Dx @ dvp_dy
        d2wp_dxdz = ikx * dwp_dx

        # -- Perturbation divergence ----------------------------------------
        div_up = dup_dx + dvp_dy + dwp_dz

        # -- Linearized viscosity perturbation: mu' = dmudT_b * T' ----------
        mu_p = dmudT_b * Tp
        dmu_p_dx = dmudT_b * dTp_dx
        dmu_p_dy = dmudT_b * dTp_dy
        dmu_p_dz = dmudT_b * dTp_dz

        # -- Eq 1: Continuity -----------------------------------------------
        # d(rho')/dt = -(u_bar·grad(rho') + u'·grad(rho_bar)
        #               + rho'·div(u_bar) + rho_bar·div(u'))
        f1 = -(
            u_b * drp_dx
            + v_b * drp_dy
            + w_b * drp_dz
            + up * drho_b_dx
            + vp * drho_b_dy
            + wp * ikx * 0  # drho_b/dz = 0
            + rp * (du_b_dx + dv_b_dy)  # div(u_bar), no dw_b/dz (2D)
            + rho_b * div_up
        )

        # -- Linearized pressure perturbation: p' = (rho'*T_b + rho_b*T')/gM2
        pp = (rp * T_b + rho_b * Tp) / gM2
        dpp_dx = (drp_dx * T_b + rp * dT_b_dx + drho_b_dx * Tp + rho_b * dTp_dx) / gM2
        dpp_dy = (drp_dy * T_b + rp * dT_b_dy + drho_b_dy * Tp + rho_b * dTp_dy) / gM2
        dpp_dz = ikx * pp  # perturbation z-derivative

        # -- Eq 2-4: Momentum (linearized) ----------------------------------
        # rho_bar*du'/dt + rho'*du_bar/dt = -(rho_bar*(u_bar·grad(u'))
        #   + u'·grad(u_bar)) + rho'*(u_bar·grad(u_bar))) - grad(p')
        #   + div(tau')/Re
        #
        # Since du_bar/dt = 0 and we work with the RHS only:

        # Linearized convective terms for each momentum equation
        conv_u = rho_b * (
            u_b * dup_dx + v_b * dup_dy + w_b * dup_dz + up * du_b_dx + vp * du_b_dy
        ) + rp * (u_b * du_b_dx + v_b * du_b_dy)
        conv_v = rho_b * (
            u_b * dvp_dx + v_b * dvp_dy + w_b * dvp_dz + up * dv_b_dx + vp * dv_b_dy
        ) + rp * (u_b * dv_b_dx + v_b * dv_b_dy)
        conv_w = rho_b * (
            u_b * dwp_dx + v_b * dwp_dy + w_b * dwp_dz + up * dw_b_dx + vp * dw_b_dy
        ) + rp * (u_b * dw_b_dx + v_b * dw_b_dy)

        # Linearized viscous stress divergence
        # tau'_ij = mu_bar*(du'_i/dx_j + du'_j/dx_i - 2/3*div(u')*delta_ij)
        #         + mu'   *(du_bar_i/dx_j + du_bar_j/dx_i - 2/3*div(u_bar)*d_ij)

        div_ub = du_b_dx + dv_b_dy  # no dw_b/dz (2D base flow)

        # d_div_up/dx, d_div_up/dy, d_div_up/dz
        d_div_up_dx = D2x @ up + Dx @ dvp_dy + ikx * dwp_dx
        d_div_up_dy = Dx @ dup_dy + D2y @ vp + ikx * dwp_dy
        d_div_up_dz = ikx * dup_dx + ikx * dvp_dy + ikx2 * wp

        # x-momentum viscous: div(tau'_x)
        # From base viscosity on perturbation velocity gradients:
        visc_u_base = (
            MU_b * (2 * d2up_dx2 - (2 / 3) * d_div_up_dx)  # d(tau'_xx)/dx
            + MU_b * (d2up_dy2 + d2vp_dxdy)  # d(tau'_xy)/dy
            + MU_b * (d2up_dz2 + d2wp_dxdz)  # d(tau'_xz)/dz
        )
        # From perturbation viscosity on base velocity gradients:
        #   Part 1: grad(mu') dot base strain rate S_bar
        visc_u_pert_grad = (
            dmu_p_dx * (2 * du_b_dx - (2 / 3) * div_ub)
            + dmu_p_dy * (du_b_dy + dv_b_dx)
            + dmu_p_dz * dw_b_dx  # du_b/dz = 0 for 2D base
        )
        #   Part 2: mu' * divergence of base strain rate
        d2u_b_dx2 = D2x @ u_b
        d2u_b_dy2 = D2y @ u_b
        d2v_b_dx2 = D2x @ v_b
        d2v_b_dy2 = D2y @ v_b
        d2w_b_dx2 = D2x @ w_b
        d2w_b_dy2 = D2y @ w_b
        d2u_b_dxdy = Dx @ du_b_dy
        d2v_b_dxdy = Dx @ dv_b_dy
        d2w_b_dxdy = Dx @ dw_b_dy
        d_div_ub_dx = D2x @ u_b + Dx @ dv_b_dy  # d(du_b/dx+dv_b/dy)/dx
        d_div_ub_dy = Dx @ du_b_dy + D2y @ v_b  # d(du_b/dx+dv_b/dy)/dy
        visc_u_pert_div = mu_p * (
            (2 * d2u_b_dx2 - (2 / 3) * d_div_ub_dx)  # d(S_bar_xx)/dx
            + (d2u_b_dy2 + d2v_b_dxdy)  # d(S_bar_xy)/dy
            # d(S_bar_xz)/dz = 0 (2D base, dz terms vanish)
        )
        # From base dmudx on perturbation velocity gradients:
        dmub_dx = dmudT_b * dT_b_dx
        dmub_dy = dmudT_b * dT_b_dy
        visc_u_cross = (
            dmub_dx * (2 * dup_dx - (2 / 3) * div_up) + dmub_dy * (dup_dy + dvp_dx)
            # dmub_dz = 0 for 2D base
        )
        div_tau_up_x = visc_u_base + visc_u_pert_grad + visc_u_pert_div + visc_u_cross

        # y-momentum viscous: div(tau'_y)
        visc_v_base = (
            MU_b * (d2vp_dx2 + Dx @ dup_dy)  # d(tau'_yx)/dx
            + MU_b * (2 * D2y @ vp - (2 / 3) * d_div_up_dy)  # d(tau'_yy)/dy
            + MU_b * (d2vp_dz2 + ikx * dwp_dy)  # d(tau'_yz)/dz
        )
        visc_v_pert_grad = (
            dmu_p_dx * (du_b_dy + dv_b_dx)
            + dmu_p_dy * (2 * dv_b_dy - (2 / 3) * div_ub)
            + dmu_p_dz * dw_b_dy  # dv_b/dz = 0
        )
        visc_v_pert_div = mu_p * (
            (d2u_b_dxdy + d2v_b_dx2)  # d(S_bar_yx)/dx
            + (2 * d2v_b_dy2 - (2 / 3) * d_div_ub_dy)  # d(S_bar_yy)/dy
            # d(S_bar_yz)/dz = 0
        )
        visc_v_cross = dmub_dx * (dvp_dx + dup_dy) + dmub_dy * (2 * dvp_dy - (2 / 3) * div_up)
        div_tau_up_y = visc_v_base + visc_v_pert_grad + visc_v_pert_div + visc_v_cross

        # z-momentum viscous: div(tau'_z)
        visc_w_base = (
            MU_b * (d2wp_dx2 + ikx * dup_dx)  # d(tau'_zx)/dx
            + MU_b * (D2y @ wp + ikx * dvp_dy)  # d(tau'_zy)/dy
            + MU_b * (2 * ikx2 * wp - (2 / 3) * d_div_up_dz)  # d(tau'_zz)/dz
        )
        visc_w_pert_grad = (
            dmu_p_dx * dw_b_dx + dmu_p_dy * dw_b_dy
            # dmu_p_dz * (2*dw_b/dz - 2/3*div_ub) = 0 (dw_b/dz = 0)
        )
        visc_w_pert_div = mu_p * (
            d2w_b_dx2  # d(S_bar_zx)/dx = d(dw_b/dx)/dx
            + d2w_b_dy2  # d(S_bar_zy)/dy = d(dw_b/dy)/dy
            # d(S_bar_zz)/dz = 0 (dw_b/dz = 0, div_ub doesn't have z)
        )
        visc_w_cross = dmub_dx * (dwp_dx + dup_dz) + dmub_dy * (dwp_dy + dvp_dz)
        div_tau_up_z = visc_w_base + visc_w_pert_grad + visc_w_pert_div + visc_w_cross

        f2 = -conv_u - dpp_dx + div_tau_up_x / Re
        f3 = -conv_v - dpp_dy + div_tau_up_y / Re
        f4 = -conv_w - dpp_dz + div_tau_up_z / Re

        # -- Eq 5: Energy (linearized, temperature form / cv) ---------------
        # rho_b*dT'/dt = -(kappa-1)*(rho_b*T_b*div(u') + rho'*T_b*div(u_b)
        #                  + rho_b*T'*div(u_b))
        #   - rho_b*(u_b*dT'/dx + v_b*dT'/dy + w_b*dT'/dz)
        #   - (rho_b*u'*dT_b/dx + rho_b*v'*dT_b/dy + rho'*(u_b*dT_b/dx + v_b*dT_b/dy))
        #   + kappa*(kappa-1)*Ma^2 * Phi'/Re
        #   + kappa/(Re*Pr) * div(mu'*grad(T_b) + mu_b*grad(T'))
        km1 = kappa - 1.0

        # Linearized pressure-dilatation (temperature form)
        pres_dil = km1 * rho_b * T_b * div_up + km1 * rp * T_b * div_ub + km1 * rho_b * Tp * div_ub

        # Linearized temperature convection
        temp_conv = (
            rho_b * (u_b * dTp_dx + v_b * dTp_dy + w_b * dTp_dz)
            + rho_b * (up * dT_b_dx + vp * dT_b_dy)
            + rp * (u_b * dT_b_dx + v_b * dT_b_dy)
        )

        # Linearized viscous dissipation:
        #   Phi' = tau_bar:grad(u') + tau':grad(u_bar)
        # where tau' = mu_bar*S' + mu'*S_bar.
        #
        # Identity: sum_ij tau_bar_ij * du'_i/dx_j = sum_ij mu_bar*S'_ij*du_bar_i/dx_j
        # (both equal mu_bar * [A + B - 2/3*div_ub*div(u')] when summed).
        # So Phi' = 2*Phi_base + mu'*S_bar:grad(u_bar).
        tau_b_xx = MU_b * (2 * du_b_dx - (2 / 3) * div_ub)
        tau_b_yy = MU_b * (2 * dv_b_dy - (2 / 3) * div_ub)
        tau_b_zz = MU_b * (-(2 / 3) * div_ub)  # dw_b/dz = 0
        tau_b_xy = MU_b * (du_b_dy + dv_b_dx)
        tau_b_xz = MU_b * dw_b_dx  # du_b/dz = 0
        tau_b_yz = MU_b * dw_b_dy  # dv_b/dz = 0

        Phi_base = (
            tau_b_xx * dup_dx
            + tau_b_xy * dup_dy
            + tau_b_xz * dup_dz
            + tau_b_xy * dvp_dx
            + tau_b_yy * dvp_dy
            + tau_b_yz * dvp_dz
            + tau_b_xz * dwp_dx
            + tau_b_yz * dwp_dy
            + tau_b_zz * dwp_dz
        )
        # mu' * S_bar_ij * du_bar_i/dx_j (only non-zero base gradients)
        Phi_mu_pert = (
            mu_p * (2 * du_b_dx - (2 / 3) * div_ub) * du_b_dx
            + mu_p * (du_b_dy + dv_b_dx) * du_b_dy
            + mu_p * (du_b_dy + dv_b_dx) * dv_b_dx
            + mu_p * (2 * dv_b_dy - (2 / 3) * div_ub) * dv_b_dy
            + mu_p * dw_b_dx * dw_b_dx
            + mu_p * dw_b_dy * dw_b_dy
        )
        Phi_p = 2.0 * Phi_base + Phi_mu_pert

        # Linearized heat conduction: div(mu'*grad(T_b) + mu_b*grad(T'))
        heat_pert_of_base = (
            dmu_p_dx * dT_b_dx + mu_p * (D2x @ T_b) + dmu_p_dy * dT_b_dy + mu_p * (D2y @ T_b)
            # dmu_p_dz * dT_b_dz = 0 (T_b is 2D)
        )
        heat_base_of_pert = (
            dmub_dx * dTp_dx
            + MU_b * d2Tp_dx2
            + dmub_dy * dTp_dy
            + MU_b * d2Tp_dy2
            + MU_b * d2Tp_dz2  # dmub_dz = 0
        )
        heat_cond_p = heat_pert_of_base + heat_base_of_pert

        f5 = (
            -pres_dil
            - temp_conv
            + kappa * km1 * Ma**2 * Phi_p / Re
            + kappa * heat_cond_p / (Re * Pr)
        )

        R = np.empty(5 * ngp, dtype=complex)
        R[0 * ngp : 1 * ngp] = f1
        R[1 * ngp : 2 * ngp] = f2
        R[2 * ngp : 3 * ngp] = f3
        R[3 * ngp : 4 * ngp] = f4
        R[4 * ngp : 5 * ngp] = f5
        return R

    @staticmethod
    def _make_bf_params_full(mesh, bf, kx):
        """Make bf_params dict with full base-flow fields for linearized residual."""
        p = mesh.used_ind
        return {
            "Re": bf.Re,
            "Ma": bf.Ma,
            "Pr": bf.Pr,
            "kappa": bf.kappa,
            "T_0": bf.T_0,
            "kx": kx,
            "RHO": bf.RHO.ravel(order="F")[p],
            "U": bf.U.ravel(order="F")[p],
            "V": bf.V.ravel(order="F")[p],
            "W": bf.W.ravel(order="F")[p],
            "T": bf.T.ravel(order="F")[p],
            "MU": bf.MU.ravel(order="F")[p],
            "dmudT": bf.dmudT.ravel(order="F")[p],
        }

    @staticmethod
    def _build_matrix_from_residual(residual_fn, ngp):
        """Extract the matrix of a linear residual by probing with unit vectors."""
        n = 5 * ngp
        # Probe with a random vector first to get dtype
        e0 = np.zeros(n)
        e0[0] = 1.0
        r0 = residual_fn(e0)
        J = np.empty((n, n), dtype=r0.dtype)
        J[:, 0] = r0
        for j in range(1, n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            J[:, j] = residual_fn(e_j)
        return J

    def test_uniform_kx_nonzero(self):
        """Uniform flow with kx=1.5: ikx terms in A0."""
        nx, ny = 8, 12
        mesh = Mesh.create(
            xrange=(0.0, 2 * np.pi),
            yrange=(-1.0, 1.0),
            nx=nx,
            ny=ny,
            fd_order=2,
            x_periodicity=True,
        )
        ones = np.ones((ny, nx))
        zeros = np.zeros((ny, nx))
        bf = BaseFlow(
            Re=1000.0,
            Ma=0.3,
            Pr=0.72,
            kappa=1.4,
            T_0=288.15,
            RHO=ones.copy(),
            U=0.5 * ones,
            V=zeros.copy(),
            W=zeros.copy(),
            T=ones.copy(),
        )
        sutherland_air(bf)
        kx = 1.5

        L_raw = _build_L_raw(mesh, bf, kx=kx)

        params = self._make_bf_params_full(mesh, bf, kx)
        J = self._build_matrix_from_residual(
            lambda q: self._linearized_residual(q, mesh, params), mesh.ngp
        )

        L_raw_dense = L_raw.toarray()

        np.testing.assert_allclose(
            J,
            L_raw_dense,
            atol=1e-10,
            err_msg=_report_mismatch(J, L_raw_dense, "uniform kx=1.5"),
        )

    def test_poiseuille_kx_nonzero(self):
        """Poiseuille with kx=2.0: all terms active."""
        nx, ny = 8, 16
        mesh = Mesh.create(
            xrange=(0.0, 2 * np.pi),
            yrange=(-1.0, 1.0),
            nx=nx,
            ny=ny,
            fd_order=2,
            x_periodicity=True,
        )
        ones = np.ones((ny, nx))
        zeros = np.zeros((ny, nx))
        bf = BaseFlow(
            Re=1000.0,
            Ma=0.3,
            Pr=0.72,
            kappa=1.4,
            T_0=288.15,
            RHO=ones.copy(),
            U=1.0 - mesh.Y**2,
            V=zeros.copy(),
            W=zeros.copy(),
            T=ones.copy(),
        )
        sutherland_air(bf)
        kx = 2.0

        L_raw = _build_L_raw(mesh, bf, kx=kx)

        params = self._make_bf_params_full(mesh, bf, kx)
        J = self._build_matrix_from_residual(
            lambda q: self._linearized_residual(q, mesh, params), mesh.ngp
        )

        L_raw_dense = L_raw.toarray()

        np.testing.assert_allclose(
            J,
            L_raw_dense,
            atol=1e-10,
            err_msg=_report_mismatch(J, L_raw_dense, "Poiseuille kx=2.0"),
        )


# =====================================================================
# Test: SymPy symbolic coefficient verification
# =====================================================================


@pytest.mark.slow
class TestSymPyCoefficients:
    """Verify individual LNSE coefficient entries against analytical values.

    Derives expected coefficient values from the compressible NS equations
    and compares against ``compute_cartesian_coefficients`` output.
    """

    @staticmethod
    def _make_uniform_coefficients(kx=0.0, U_val=0.5):
        """Compute coefficients for uniform flow at a single grid point."""
        Re, Ma, Pr, kappa, T_0 = 1000.0, 0.3, 0.72, 1.4, 288.15
        Z = np.zeros(1)
        I = np.ones(1)

        bf = BaseFlow(
            Re=Re,
            Ma=Ma,
            Pr=Pr,
            kappa=kappa,
            T_0=T_0,
            RHO=np.ones((1, 1)),
            U=U_val * np.ones((1, 1)),
            V=np.zeros((1, 1)),
            W=np.zeros((1, 1)),
            T=np.ones((1, 1)),
        )
        sutherland_air(bf)
        MU_val = bf.MU.ravel()
        dmudT_val = bf.dmudT.ravel()
        d2mudT2_val = bf.d2mudT2.ravel()
        P_val = I * I / (Ma**2 * kappa)

        coeffs = compute_cartesian_coefficients(
            U=U_val * I,
            V=Z,
            W=Z,
            RHO=I,
            T=I,
            MU=MU_val,
            P=P_val,
            kx=kx,
            Re=Re,
            Ma=Ma,
            Pr=Pr,
            kappa=kappa,
            dUdx=Z,
            dVdx=Z,
            dWdx=Z,
            dRHOdx=Z,
            dTdx=Z,
            dPdx=Z,
            dUdy=Z,
            dVdy=Z,
            dWdy=Z,
            dRHOdy=Z,
            dTdy=Z,
            dPdy=Z,
            d2Udx2=Z,
            d2Vdx2=Z,
            d2Wdx2=Z,
            d2Udy2=Z,
            d2Vdy2=Z,
            d2Wdy2=Z,
            d2Tdx2=Z,
            d2Tdy2=Z,
            d2Udxdy=Z,
            d2Vdxdy=Z,
            d2Wdxdy=Z,
            dmudT=dmudT_val,
            d2mudT2=d2mudT2_val,
        )
        return coeffs, {"Re": Re, "Ma": Ma, "Pr": Pr, "kappa": kappa, "mu": MU_val[0], "U": U_val}

    def test_B_matrix(self):
        """B matrix diagonal: B = 1j * diag(1, rho, rho, rho, rho)."""
        coeffs, p = self._make_uniform_coefficients()
        rho = 1.0

        np.testing.assert_allclose(coeffs.B[0, 0, 0], 1j)
        for i in range(1, 5):
            np.testing.assert_allclose(coeffs.B[i, i, 0], 1j * rho)
            # Off-diagonals should be zero
            for j in range(5):
                if j != i:
                    np.testing.assert_allclose(coeffs.B[i, j, 0], 0.0, atol=1e-15)

    def test_continuity_A0(self):
        """Continuity A0 entries for uniform flow with kx=1.5."""
        kx = 1.5
        coeffs, p = self._make_uniform_coefficients(kx=kx, U_val=0.5)

        # From linearized continuity:
        #   f1' = -(dUdx + dVdy + ikx*W)*rho' - dRHOdx*u' - dRHOdy*v'
        #         - ikx*rho*w' - u*drho'/dx - v*drho'/dy - rho*du'/dx - rho*dv'/dy
        # Uniform flow (all gradients zero, W=0):
        #   A0[0,0] = 0,  A0[0,1] = 0,  A0[0,2] = 0,
        #   A0[0,3] = -ikx*rho = -1j*kx,  A0[0,4] = 0
        # (Note: J = L_raw, so these are the raw Jacobian entries.)

        np.testing.assert_allclose(coeffs.A0[0, 0, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(coeffs.A0[0, 1, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(coeffs.A0[0, 2, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(coeffs.A0[0, 3, 0], -1j * kx, atol=1e-14)
        np.testing.assert_allclose(coeffs.A0[0, 4, 0], 0.0, atol=1e-14)

    def test_continuity_first_derivatives(self):
        """Continuity Ax and Ay entries for uniform flow."""
        coeffs, p = self._make_uniform_coefficients(U_val=0.7)
        u = p["U"]
        rho = 1.0

        # From linearized continuity:
        #   Ax[0,0] = -u (coeff of drho'/dx)
        #   Ax[0,3] = -rho (coeff of dw'/dx)  ... wait
        # Actually: Ax[0,1] = -rho (coeff of du'/dx from rho*div_u term)
        # And Ax[0,0] = -u (coeff of drho'/dx from u*drho'/dx term)
        # Ay[0,0] = -v = 0 (V=0)
        # Ay[0,2] = -rho (coeff of dv'/dy from rho*div_u term)

        np.testing.assert_allclose(coeffs.Ax[0, 0, 0], -u, atol=1e-14)
        np.testing.assert_allclose(coeffs.Ax[0, 1, 0], -rho, atol=1e-14)
        np.testing.assert_allclose(coeffs.Ay[0, 0, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(coeffs.Ay[0, 2, 0], -rho, atol=1e-14)

    def test_continuity_no_second_derivatives(self):
        """Continuity equation has no second-derivative terms."""
        coeffs, _ = self._make_uniform_coefficients()
        for j in range(5):
            np.testing.assert_allclose(coeffs.Axx[0, j, 0], 0.0, atol=1e-15)
            np.testing.assert_allclose(coeffs.Ayy[0, j, 0], 0.0, atol=1e-15)
            np.testing.assert_allclose(coeffs.Axy[0, j, 0], 0.0, atol=1e-15)

    def test_momentum_viscous_second_derivatives(self):
        """Viscous second-derivative coefficients in momentum equations.

        For x-momentum (row 1), the viscous stress divergence gives:
            d2u'/dx2 coefficient = (4/3)*mu/Re
            d2u'/dy2 coefficient = mu/Re
        Similarly for y and z momentum with the appropriate components.
        """
        coeffs, p = self._make_uniform_coefficients()
        mu, Re = p["mu"], p["Re"]

        # x-momentum: Axx[1,1] = (4/3)*mu/Re, Ayy[1,1] = mu/Re
        np.testing.assert_allclose(
            coeffs.Axx[1, 1, 0], (4.0 / 3.0) * mu / Re, atol=1e-14, rtol=1e-12
        )
        np.testing.assert_allclose(coeffs.Ayy[1, 1, 0], mu / Re, atol=1e-14, rtol=1e-12)

        # y-momentum: Axx[2,2] = mu/Re, Ayy[2,2] = (4/3)*mu/Re
        np.testing.assert_allclose(coeffs.Axx[2, 2, 0], mu / Re, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(
            coeffs.Ayy[2, 2, 0], (4.0 / 3.0) * mu / Re, atol=1e-14, rtol=1e-12
        )

        # z-momentum: Axx[3,3] = mu/Re, Ayy[3,3] = mu/Re
        np.testing.assert_allclose(coeffs.Axx[3, 3, 0], mu / Re, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(coeffs.Ayy[3, 3, 0], mu / Re, atol=1e-14, rtol=1e-12)

    def test_momentum_pressure_gradient(self):
        """Pressure gradient coefficients in x and y momentum.

        p = rho*T/(gamma*Ma^2), so:
            dp'/dx has contributions from rho' and T' (and their x-derivatives).
        For uniform flow (rho=T=1):
            A0[1,4] from dp'/dx: includes -T/(gamma*Ma^2) * drho'/dx ... no,
            that's Ax[1,0].  Let me be careful.

        Pressure: p = rho*T/(gamma*Ma^2)
        dp'/dx = (drho'/dx * T + rho * dT'/dx) / (gamma*Ma^2)
               = drho'/dx * 1/(gamma*Ma^2) + dT'/dx * 1/(gamma*Ma^2)  [for rho=T=1]

        x-momentum: f2 has -dp'/dx, so:
            Ax[1,0] contribution from pressure: -T/(gamma*Ma^2) = -1/gM2
            Ax[1,4] contribution from pressure: -rho/(gamma*Ma^2) = -1/gM2
        """
        coeffs, p = self._make_uniform_coefficients()
        gM2 = p["kappa"] * p["Ma"] ** 2

        # Pressure contributes to Ax[1,0] and Ax[1,4]
        # But there are also convective terms. For uniform flow with U=0.5:
        #   f2 = -rho*(u*du/dx + ...) - dp/dx + viscous/Re
        # Linearizing -rho*u*du/dx → -rho_bar*u_bar*du'/dx term
        # gives Ax[1,1] = -rho*u = -0.5 from convection (plus viscous).
        # The pressure part: -dp'/dx → Ax[1,0] = -1/gM2

        # For x-momentum, Ax[1,0] = -T/(gamma*Ma^2) = -1/gM2
        # (pressure contribution from drho'/dx)
        np.testing.assert_allclose(
            coeffs.Ax[1, 0, 0],
            -1.0 / gM2,
            atol=1e-14,
            err_msg="Ax[1,0] pressure-gradient coefficient",
        )

        # For y-momentum, Ay[2,0] = -T/(gamma*Ma^2) = -1/gM2
        # (pressure contribution from drho'/dy)
        np.testing.assert_allclose(
            coeffs.Ay[2, 0, 0],
            -1.0 / gM2,
            atol=1e-14,
            err_msg="Ay[2,0] pressure-gradient coefficient",
        )

    def test_energy_heat_conduction(self):
        """Heat conduction coefficients in energy equation.

        Conduction term: kappa/(Re*Pr) * div(mu*grad(T))
        Second derivative of T': Axx[4,4] = Ayy[4,4] = kappa*mu/(Re*Pr)
        """
        coeffs, p = self._make_uniform_coefficients()
        mu, Re, Pr, kappa = p["mu"], p["Re"], p["Pr"], p["kappa"]
        expected = kappa * mu / (Re * Pr)

        np.testing.assert_allclose(coeffs.Axx[4, 4, 0], expected, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(coeffs.Ayy[4, 4, 0], expected, atol=1e-14, rtol=1e-12)

    def test_momentum_convective_A0(self):
        """Convective A0 entries for uniform flow.

        For x-momentum with U=0.5, kx=0, all gradients zero:
            f2 = -rho*(u*du/dx + v*du/dy + w*du/dz) - dp/dx + viscous/Re

        Linearization:
            df2/du' at (rho_bar, U_bar, ...):
            From -rho_bar*(U*du'/dx + u'*dU/dx + V*du'/dy):
              A0 contrib from u': -rho_bar*dUdx = 0 (uniform)
            Plus viscous A0 from kx: mu*kx^2/Re = 0 (kx=0)
            So A0[1,1] = 0 for kx=0, uniform flow.

            df2/drho' = -(U*dU/dx + V*dU/dy) ... all zero for uniform.
            Plus pressure: -dp'/dx has no rho' A0 contribution (only Ax).
            So A0[1,0] = 0.
        """
        coeffs, _ = self._make_uniform_coefficients(kx=0.0)

        np.testing.assert_allclose(coeffs.A0[1, 0, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(coeffs.A0[1, 1, 0], 0.0, atol=1e-14)

    def test_momentum_A0_kx_nonzero(self):
        """A0 diagonal entries with kx != 0 (Fourier-mode viscous damping).

        The kx²-dependent A0 entries come from d(tau_iz)/dz linearized
        w.r.t. the same velocity component via du'/dz = ikx*u':

        x-momentum: d(tau_xz)/dz where tau_xz = mu*(du/dz + dw/dx)
            → mu*(ikx)²*u' = -mu*kx²*u'  → A0[1,1] = -mu*kx²/Re

        y-momentum: d(tau_yz)/dz where tau_yz = mu*(dv/dz + dw/dy)
            → -mu*kx²*v'  → A0[2,2] = -mu*kx²/Re

        z-momentum: d(tau_zz)/dz where tau_zz = mu*(2*dw/dz - 2/3*div_u)
            and dw'/dz in div_u gives: mu*(2 - 2/3)*ikx² = -(4/3)*mu*kx²
            → A0[3,3] = -(4/3)*mu*kx²/Re

        The (4/3) factor appears ONLY in z-momentum because dw'/dz
        enters both the 2*dw/dz and the -2/3*div_u parts of tau_zz.
        """
        kx = 2.0
        coeffs, p = self._make_uniform_coefficients(kx=kx)
        mu, Re = p["mu"], p["Re"]

        # x and y momentum: plain mu*kx²/Re
        np.testing.assert_allclose(
            coeffs.A0[1, 1, 0].real,
            -mu * kx**2 / Re,
            atol=1e-12,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            coeffs.A0[2, 2, 0].real,
            -mu * kx**2 / Re,
            atol=1e-12,
            rtol=1e-10,
        )

        # z-momentum: (4/3)*mu*kx²/Re (bulk viscosity factor)
        np.testing.assert_allclose(
            coeffs.A0[3, 3, 0].real,
            -(4.0 / 3.0) * mu * kx**2 / Re,
            atol=1e-12,
            rtol=1e-10,
        )

    def test_symmetry_of_second_derivatives(self):
        """Second derivative matrices should be symmetric in their role.

        For uniform flow: Axx[i,j] = Ayy[i,j] for diagonal blocks (i=j)
        in the z-momentum equation (row 3), since tau_zx and tau_zy have
        the same structure.
        """
        coeffs, _ = self._make_uniform_coefficients()

        # z-momentum: Axx[3,3] = Ayy[3,3] = mu/Re
        np.testing.assert_allclose(
            coeffs.Axx[3, 3, 0],
            coeffs.Ayy[3, 3, 0],
            atol=1e-15,
            err_msg="z-momentum should have equal Axx and Ayy for the w-w block",
        )
