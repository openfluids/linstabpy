#!/usr/bin/env python3
"""
Example 5: Paraboloid resolvent analysis.

Translates example_5_Paraboloid.m using a curvilinear transformation to a
parabolic physical grid and an axisymmetric model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from scipy.ndimage import zoom

from linstab2d.baseflow import load_paraboloid, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.resolvent import resolvent_analysis, compute_resolvent_svd_residuals
from linstab2d.utils.plotting import plot_flow

from ._helpers import env_bool, env_float, env_int

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")


def compute():
    """Run resolvent analysis on the paraboloid-mapped cavity.

    Uses m = 5 azimuthal Fourier mode and axy model.
    """
    omega = env_float("LINSTAB2D_PARABOLOID_OMEGA", 0.0)
    omega_shift_fallback = env_float("LINSTAB2D_PARABOLOID_OMEGA_SHIFT", 1e-4)
    use_shift_fallback = env_bool(
        "LINSTAB2D_PARABOLOID_USE_OMEGA_SHIFT_FALLBACK", default=True
    )
    n_modes = env_int("LINSTAB2D_PARABOLOID_N_MODES", 3)
    m = 5
    fd_order = 4

    Nx = env_int("LINSTAB2D_PARABOLOID_NX", 100)
    Ny = env_int("LINSTAB2D_PARABOLOID_NY", 100)
    Lx = 2.25
    Ly = 1.5

    xrange = (-1.0, 0.0)
    yrange = (0.0, 1.0)
    cmesh = Mesh.create(
        xrange,
        yrange,
        Nx,
        Ny,
        fd_order,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )

    X = cmesh.X.copy()
    x_nonzero = X != 0.0
    X_mapped = np.zeros_like(X)
    X_mapped[x_nonzero] = 1 / (
        1 / (X[x_nonzero] ** 4) * np.sign(X[x_nonzero]) + 1 / X[x_nonzero]
    )
    X = X_mapped
    X[np.isnan(X)] = 0.0
    X = X / np.max(np.abs(X))
    Y = cmesh.Y.copy()
    Y = 1 - (1 - Y) ** 1.4
    Y = 1 - (1 - Y) ** 1.4
    Y = np.maximum(Y, 1e-10)

    # Scale to physical domain before Z² conformal mapping (MATLAB convention)
    X = X * Lx - 0.5
    Y = Y * Ly

    Z = X + 1j * Y
    Xp = -np.real(Z**2)
    Yp = -np.imag(Z**2)
    mesh = cmesh.deform(X_new=Xp, Y_new=Yp)

    baseflow = load_paraboloid(mesh)
    sutherland_air(baseflow)

    xs = -0.5 - Lx + 0.1
    xs_trans = Lx / 10.0
    ys = Ly - 0.1
    ys_trans = 0.1
    sponge_amp = 5.0
    # MATLAB defines the sponge on the pre-conformal (x, y) grid, before the
    # final Z^2 mapping to the physical mesh.
    sponge_x = np.tanh(-(X - xs) / xs_trans) + 1
    sponge_y = np.tanh((Y - ys) / ys_trans) + 1
    mesh.sponge = sponge_amp / 2 * np.maximum(sponge_x, sponge_y)

    L0, idx = build_lnse_operator(mesh, baseflow, model="axy", m=m)
    borders = "ltr"
    vars_ = "uvwT"
    L0, bc_indices = apply_dirichlet(L0, idx, borders, vars_, mesh.ngp)

    used_ind = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[used_ind]
    rho = baseflow.RHO.ravel(order="F")[used_ind]
    T = baseflow.T.ravel(order="F")[used_ind]
    R = mesh.Y.ravel(order="F")[used_ind]
    W_diag, _ = compute_chu_norm(
        W_int,
        rho,
        T,
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
        model="axy",
        R=R,
    )

    B_diag = np.ones(5 * mesh.ngp)
    B_diag[idx["T_j"]] = 0.0
    B_diag[idx["rho_j"]] = 0.0
    B_diag[bc_indices] = 0.0
    B_mat = sp.diags(B_diag).tocsc()
    C_mat = B_mat.copy()

    # Paraboloid at ω=0 can be singular depending on discretization and
    # boundary conditions; a small imaginary shift is a conservative fallback
    # to keep this example numerically robust in reduced-memory settings.
    solve_freqs = [complex(omega)]
    if use_shift_fallback and abs(omega) < 1e-12 and omega_shift_fallback > 0.0:
        solve_freqs.append(1j * omega_shift_fallback)
    last_error: Exception | None = None
    gains: np.ndarray
    responses: np.ndarray
    forcings: np.ndarray
    omega_used = solve_freqs[0]

    for candidate in solve_freqs:
        try:
            gains, responses, forcings = resolvent_analysis(
                L0,
                candidate,
                n_modes,
                W_diag,
                B_mat=B_mat,
                C_mat=C_mat,
                filter_fn=mesh.filters.filter if mesh.filters is not None else None,
                filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
            )
            omega_used = candidate
            if candidate != solve_freqs[0]:
                print(f"Using fallback frequency ω={candidate:.3g} to avoid singular/ill-conditioned solve at ω={solve_freqs[0]:.3g}.")
            break
        except RuntimeError as exc:
            last_error = exc
            if candidate == solve_freqs[0] and len(solve_freqs) > 1:
                print(
                    "Paraboloid resolvent failed at ω=0; retrying with small imaginary"
                    f" shift ω={solve_freqs[1]:.3g}."
                )
                continue
            raise

    if "gains" not in locals():
        raise RuntimeError("Paraboloid resolvent failed for all frequency attempts.") from last_error

    # ---- SVD self-residuals -----------------------------------------------
    # MATLAB used 5-output lu(A) (UMFPACK) + ARPACK eigs at tol=eps=2.2e-16,
    # default subspace opts.p = 2*nEig+1.  Python uses MUMPS/METIS + SLEPc EPS
    # at tol=1e-12, ncv = max(2*n_modes+3, 20).  Different fill-reducing
    # orderings yield gain mismatch ~0.04% for this case; structurally
    # unavoidable.  omega_used (not the nominal omega=0) is passed so the
    # correct shifted operator is rebuilt for the residual check.
    svd_residuals_fwd, svd_residuals_adj = compute_resolvent_svd_residuals(
        L0,
        omega_used,
        B_mat,
        C_mat,
        gains,
        forcings,
        responses,
        W_diag,
        filter_fn=mesh.filters.filter if mesh.filters is not None else None,
        filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        svd_residuals_fwd=svd_residuals_fwd,
        svd_residuals_adj=svd_residuals_adj,
        idx_u_j=idx["u_j"],
        omega=omega,
        omega_used_real=float(np.real(omega_used)),
        omega_used_imag=float(np.imag(omega_used)),
        m=m,
        X=mesh.X,
        Y=mesh.Y,
        n_modes=n_modes,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def plot(data):
    """Plot forcing and response modes for the first three modes."""
    import matplotlib.pyplot as plt

    gains = data["gains"]
    responses = data["responses"]
    forcings = data["forcings"]
    idx_u = data["idx_u_j"]
    X = data["X"]
    Y = data["Y"]
    n_modes = int(data["n_modes"])
    refine = env_int("LINSTAB2D_PARABOLOID_PLOT_REFINED", 3)

    n_plot = min(n_modes, 3)
    fig, axes = plt.subplots(max(1, n_plot), 2, figsize=(12, 4 * max(1, n_plot)))
    if n_plot == 1:
        axes = np.array([[axes[0], axes[1]]]) if axes.ndim == 1 else axes

    def _upscale_field(field: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if factor <= 1:
            return X, Y, field
        return (
            zoom(X, factor, order=1),
            zoom(Y, factor, order=1),
            zoom(field, factor, order=1),
        )

    for i in range(n_plot):
        f_u = forcings[idx_u, i].reshape(X.shape, order="F")
        r_u = responses[idx_u, i].reshape(X.shape, order="F")
        X_plot_f, Y_plot_f, f_u = _upscale_field(np.real(f_u), refine)
        X_plot_r, Y_plot_r, r_u = _upscale_field(np.real(r_u), refine)
        ax_f = axes[i, 0]
        plot_flow(X_plot_f, Y_plot_f, f_u, ax=ax_f, levels=201, cmap="RdBu_r")
        ax_f.set_title(f"f_u^{i+1}, sigma={gains[i]:.2f}")
        ax_r = axes[i, 1]
        plot_flow(X_plot_r, Y_plot_r, r_u, ax=ax_r, levels=201, cmap="RdBu_r")
        ax_r.set_title(f"u^{i+1}")

    for i in range(n_plot):
        axes[i, 0].set_xlabel("x")
        axes[i, 1].set_xlabel("x")

    fig.suptitle("Paraboloid Resolvent Analysis (m=5, axy)")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_PNG}")


def main():
    try:
        data = np.load(OUTPUT_NPZ)
    except FileNotFoundError:
        compute()
        data = np.load(OUTPUT_NPZ)

    plot(data)


if __name__ == "__main__":
    main()
