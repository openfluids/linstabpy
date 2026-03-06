#!/usr/bin/env python3
"""
Example 4: Cavity global/adjoint stability analysis.

Translates example_4_Cavity.m using a masked cavity geometry, deformed mesh,
and direct + adjoint temporal eigenvalue problems.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from linstab2d.baseflow import load_cavity, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.eigenvalue import adjoint_eigenvalues, temporal_eigenvalues
from linstab2d.utils.plotting import plot_flow

from ._helpers import env_complex, env_int, to_grid

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")


def compute():
    """Run cavity direct/adjoint temporal eigenproblems."""
    omega_target = env_complex("LINSTAB2D_CAVITY_OMEGA_TARGET", 1.5 + 1.0j)
    n_modes = env_int("LINSTAB2D_CAVITY_N_MODES", 50)
    n_plot = min(3, n_modes)
    beta = 0.0
    arnoldi_ncv = env_int("LINSTAB2D_CAVITY_NCV", 300)

    Nx = env_int("LINSTAB2D_CAVITY_NX", 241)
    Ny = env_int("LINSTAB2D_CAVITY_NY", 101)
    FDorder = 4

    x_cavity_start = 0.34029
    L_cavity = 1.9995
    xrange = (
        x_cavity_start - L_cavity,
        x_cavity_start + 2 * L_cavity,
    )
    yrange = (-1.0, 1.0)

    cmesh = Mesh.create(
        xrange,
        yrange,
        Nx,
        Ny,
        FDorder,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )
    # MATLAB convention: mask=True means "remove this point".
    # Python Mesh.mask() convention: mask=True means "keep this point".
    # Negate to convert.
    remove = (cmesh.Y < 0.0) & ((cmesh.X < 0.34) | (cmesh.X > 2.34))
    mesh = cmesh.mask(~remove)

    # --- y-deformation (mesh concentration + top stretch) ------------------
    Y = mesh.Y.copy()
    a = 0.5
    Y = (Y**3 + a * Y) / (1 + a)
    strech_n = 3
    p = Y > 0.0
    Y[p] = Y[p] + 2.0 * ((Y[p] - 0.0) / (np.max(Y[p]) - 0.0)) ** strech_n
    mesh = mesh.deform(Y_new=Y)

    # --- x-stretch on the right ------------------------------------------
    X = mesh.X.copy()
    p = X > 2.8
    X[p] = X[p] + 1.0 * ((X[p] - 2.8) / (np.max(X[p]) - 2.8)) ** strech_n
    mesh = mesh.deform(X_new=X)

    baseflow = load_cavity(mesh)
    sutherland_air(baseflow)

    xs_trans = 0.5
    xs_offset = 0.5
    ys_trans = 0.5
    ys_offset = 1.0
    sponge_amp = 5.0
    sponge = np.zeros(mesh.X.shape, dtype=float)
    ind = mesh.used_ind
    x = mesh.X.ravel(order="F")[ind]
    y = mesh.Y.ravel(order="F")[ind]
    term_x = np.tanh(-(x - np.min(x) - xs_offset) / xs_trans) + 1
    term_y = np.tanh((x - np.max(x) + xs_offset) / xs_trans) + 1
    term_t = np.tanh((y - np.max(y) + ys_offset) / ys_trans) + 1
    sponge_vals = sponge_amp / 2 * np.maximum(term_t, term_x + term_y)
    sponge.ravel(order="F")[ind] = sponge_vals
    mesh.sponge = sponge

    L0, idx = build_lnse_operator(mesh, baseflow, model="2D", kx=beta)
    borders = "lbtrm"
    vars_ = "uvwT"
    L0, bc_indices = apply_dirichlet(L0, idx, borders, vars_, mesh.ngp)

    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO = baseflow.RHO.ravel(order="F")[p]
    T = baseflow.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(
        W_int,
        RHO,
        T,
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
        model="2D",
    )

    # Direct and adjoint temporal spectra
    omega, direct_modes = temporal_eigenvalues(
        L0,
        n_modes,
        omega_target,
        W_diag=W_diag,
        ncv=arnoldi_ncv,
    )
    omega_adj_target = 1j * np.conj(omega_target)
    omega_adj, adjoint_modes = adjoint_eigenvalues(
        L0,
        n_modes,
        omega_adj_target,
        W_diag,
        invW_diag,
        ncv=arnoldi_ncv,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        omega=omega,
        omega_adj=omega_adj,
        direct_modes=direct_modes,
        adjoint_modes=adjoint_modes,
        X=mesh.X,
        Y=mesh.Y,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
        used_ind=mesh.used_ind,
        bc_indices=bc_indices,
        n_plot=n_plot,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def plot(data):
    """Plot temporal spectra plus direct/adjoint leading modes."""
    import matplotlib.pyplot as plt

    output_direct = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_direct.png")
    output_adjoint = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_adjoint.png")
    output_spectrum = OUTPUT_PNG

    omega = data["omega"]
    omega_adj = data["omega_adj"]
    direct_modes = data["direct_modes"]
    adjoint_modes = data["adjoint_modes"]
    X = data["X"]
    Y = data["Y"]
    used_ind = data["used_ind"]
    idx_rho = data["idx_rho_j"]
    idx_u = data["idx_u_j"]
    idx_v = data["idx_v_j"]
    idx_w = data["idx_w_j"]
    idx_T = data["idx_T_j"]
    n_plot = int(data["n_plot"])

    direct_order = np.argsort(np.imag(omega))[::-1][:n_plot]

    # Convert adjoint eigenvalues to direct convention for plotting/sorting.
    # Python adjoint_eigenvalues returns eig(W⁻¹ L0ᴴ W) = i·conj(omega_d).
    # MATLAB returns eig(W⁻¹ (iL0)ᴴ W) = conj(omega_d).
    # Recover direct eigenvalues: omega_d_from_adj = conj(-i·omega_adj).
    omega_d_from_adj = np.conj(-1j * omega_adj)
    adj_order = np.argsort(np.imag(omega_d_from_adj))[::-1][:n_plot]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.real(omega), np.imag(omega), "ob", ms=7, mew=1, label="Direct")
    ax.plot(
        np.real(omega_d_from_adj),
        np.imag(omega_d_from_adj),
        "x",
        color="orange",
        ms=14,
        mew=2.2,
        label="Adjoint",
        zorder=4,
    )
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.set_title("Temporal spectra")
    ax.grid(True)
    ax.legend()

    for i, j in enumerate(direct_order):
        mode = direct_modes[:, int(j)]
        fig_mode, axes = plt.subplots(3, 2, figsize=(12, 10))
        fields = [
            (idx_rho, r"$\rho$"),
            (idx_u, r"$u$"),
            (idx_v, r"$v$"),
            (idx_w, r"$w$"),
            (idx_T, r"$T$"),
        ]
        for k, (idx_var, title_) in enumerate(fields):
            ax_k = axes.flat[k]
            field = to_grid(mode[idx_var], used_ind, X.shape)
            plot_flow(X, Y, np.real(field), ax=ax_k, levels=51, cmap="RdBu_r")
            ax_k.set_title(f"Direct {title_}, ω={omega[j]:.3f}")
        axes.flat[-1].set_visible(False)

        fig_mode.suptitle(f"Direct mode {int(j) + 1}")
        fig_mode.tight_layout()
        if i == 0:
            fig_mode.savefig(output_direct, dpi=600, bbox_inches="tight")
            plt.close(fig_mode)
            print(f"Direct mode figure saved to {output_direct}")

    for i, j in enumerate(adj_order):
        mode = adjoint_modes[:, int(j)]
        fig_mode, axes = plt.subplots(3, 2, figsize=(12, 10))
        fields = [
            (idx_rho, r"$\rho$"),
            (idx_u, r"$u$"),
            (idx_v, r"$v$"),
            (idx_w, r"$w$"),
            (idx_T, r"$T$"),
        ]
        for k, (idx_var, title_) in enumerate(fields):
            ax_k = axes.flat[k]
            field = to_grid(mode[idx_var], used_ind, X.shape)
            plot_flow(X, Y, np.real(field), ax=ax_k, levels=51, cmap="RdBu_r")
            ax_k.set_title(f"Adjoint ω*={omega_d_from_adj[j]:.3f} : {title_}")
        axes.flat[-1].set_visible(False)

        fig_mode.suptitle(f"Adjoint mode {int(j) + 1}")
        fig_mode.tight_layout()
        if i == 0:
            fig_mode.savefig(output_adjoint, dpi=600, bbox_inches="tight")
            plt.close(fig_mode)
            print(f"Adjoint mode figure saved to {output_adjoint}")

    fig.savefig(output_spectrum, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(
        "Figures saved to:\n"
        f"- Spectrum: {output_spectrum}\n"
        f"- Direct mode: {output_direct}\n"
        f"- Adjoint mode: {output_adjoint}"
    )


def main():
    try:
        data = np.load(OUTPUT_NPZ)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        compute()
        data = np.load(OUTPUT_NPZ)

    plot(data)


if __name__ == "__main__":
    main()
