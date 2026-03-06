#!/usr/bin/env python3
"""
Example 3: Streaky boundary-layer temporal and spatial analysis.

Translates example_3_Streaky_BL.m with:
- periodic x-domain and Hanifi-like y stretch
- temporal eigenvalues
- spatial eigenvalues in generalized (alpha) form
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from linstab2d.baseflow import load_streaky_bl
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.operators.spatial import build_spatial_operator
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet, apply_dirichlet_spatial
from linstab2d.solvers.eigenvalue import temporal_eigenvalues
from linstab2d.solvers.spatial_eig import spatial_eigenvalues
from linstab2d.utils.floquet import reconstruct_floquet
from linstab2d.utils.plotting import plot_flow

from ._helpers import env_float, env_int, leading_mode_real_frequency

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")


def _compute():
    """Run temporal and spatial eigensolutions."""
    Lx = np.pi
    Ly = 5.0
    Nx = env_int("LINSTAB2D_STREAKY_NX", 125)
    Ny = env_int("LINSTAB2D_STREAKY_NY", 250)
    print(f"Streaky BL grid points: Nx={Nx}, Ny={Ny}")
    fd_order = 4

    n_periods = 2
    floquet_exp = 0.0 if n_periods == 0 else 2 * np.pi / (Lx * n_periods)
    alpha_temp = 0.6283
    omega_tar = 0.5498 + 1.1j
    alpha_target = alpha_temp
    n_modes = env_int("LINSTAB2D_STREAKY_N_MODES", 50)
    # Hamada's exported Example 3 bundle was generated with opts.p = 150
    # rather than the original 1500 to keep the solve tractable.
    arnoldi_ncv = env_int("LINSTAB2D_STREAKY_NCV", 150)
    spatial_omega_imag_shift = env_float(
        "LINSTAB2D_STREAKY_SPATIAL_OMEGA_IMAG_SHIFT",
        0.0,
    )

    xrange = (0.0, Lx)
    yrange = (-1.0, 1.0)
    cmesh = Mesh.create(
        xrange,
        yrange,
        Nx,
        Ny,
        fd_order,
        y_symmetry=False,
        x_periodicity=True,
        alpha_filter=0.0,
    )

    # Hanifi-like y-mapping
    yi = 2.0
    a = yi * Ly / (Ly - 2 * yi)
    b = 1.0 + 2.0 * a / Ly
    Y = a * (1.0 + cmesh.Y) / (b - cmesh.Y)
    mesh = cmesh.deform(Y_new=Y)

    baseflow = load_streaky_bl(mesh)

    ys = Ly - 1.0
    ys_trans = 0.5
    sponge_amp = 0.1
    mesh.sponge = sponge_amp / 2 * (np.tanh((mesh.Y - ys) / ys_trans) + 1.0)

    # Temporal eigenproblem
    L0, idx = build_lnse_operator(
        mesh,
        baseflow,
        kx=alpha_temp,
        model="2D",
        floquet_exp=floquet_exp,
    )
    borders = "tb"
    variables = "uvwT"
    L0, _ = apply_dirichlet(L0, idx, borders, variables, mesh.ngp)

    used_ind = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[used_ind]
    rho_p = baseflow.RHO.ravel(order="F")[used_ind]
    T_p = baseflow.T.ravel(order="F")[used_ind]
    W_diag, _ = compute_chu_norm(W_int, rho_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp)

    temporal_omega, temporal_modes = temporal_eigenvalues(
        L0,
        n_modes,
        omega_tar,
        W_diag=W_diag,
        ncv=arnoldi_ncv,
    )

    # MATLAB's spatial branch validates against the real part of the
    # least-damped temporal mode, optionally with an added imag sweep.
    omega_spatial = leading_mode_real_frequency(
        temporal_omega,
        imag_shift=spatial_omega_imag_shift,
    )

    # Spatial generalized EVP in alpha
    Lc, R, Lw, _ = build_spatial_operator(
        mesh,
        baseflow,
        model="2D",
        floquet_exp=floquet_exp,
    )
    Lc, R, _ = apply_dirichlet_spatial(Lc, R, idx, borders, variables, mesh.ngp)
    spatial_alpha, spatial_modes = spatial_eigenvalues(
        Lc,
        Lw,
        R,
        omega_spatial,
        n_modes,
        alpha_target,
        ncv=arnoldi_ncv,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        temporal_omega=temporal_omega,
        temporal_modes=temporal_modes,
        spatial_alpha=spatial_alpha,
        spatial_modes=spatial_modes,
        X=mesh.X,
        Y=mesh.Y,
        used_ind=used_ind,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
        floquet_exp=floquet_exp,
        n_modes=n_modes,
        alpha_temp=alpha_temp,
        omega_tar_real=float(np.real(omega_tar)),
        omega_tar_imag=float(np.imag(omega_tar)),
        omega_spatial=omega_spatial,
        alpha_target=float(alpha_target),
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def _plot():
    """Plot selected temporal and spatial modes."""
    import matplotlib.pyplot as plt

    output_png_temporal = OUTPUT_PNG
    output_png_spatial = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_spatial.png")
    output_png_spectra = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_spectra.png")

    data = np.load(OUTPUT_NPZ)
    temporal_omega = data["temporal_omega"]
    temporal_modes = data["temporal_modes"]
    spatial_alpha = data["spatial_alpha"]
    spatial_modes = data["spatial_modes"]
    X = data["X"]
    Y = data["Y"]
    used_ind = data["used_ind"]
    floquet_exp = float(data["floquet_exp"])
    idx_rho = data["idx_rho_j"]
    idx_u = data["idx_u_j"]
    idx_v = data["idx_v_j"]
    idx_w = data["idx_w_j"]
    idx_T = data["idx_T_j"]
    n_modes = int(data["n_modes"])

    # Spectra figure.
    fig_spec, ax_spec = plt.subplots(1, 2, figsize=(12, 5))
    ax_spec[0].plot(np.real(temporal_omega), np.imag(temporal_omega), "o", label="Temporal spectra")
    for i in np.argsort(np.imag(temporal_omega))[-4:]:
        ax_spec[0].plot(np.real(temporal_omega[i]), np.imag(temporal_omega[i]), "x", label=f"{int(i)}")
    ax_spec[0].set_title("Temporal spectra")
    ax_spec[0].set_xlabel(r"$\Re(\omega)$")
    ax_spec[0].set_ylabel(r"$\Im(\omega)$")
    ax_spec[0].grid(True)

    ax_spec[1].plot(np.real(spatial_alpha), np.imag(spatial_alpha), "ob", label="Spatial spectra")
    for i in np.argsort(np.abs(spatial_alpha))[:5]:
        ax_spec[1].plot(np.real(spatial_alpha[i]), np.imag(spatial_alpha[i]), "x", label=f"{int(i)}")
    ax_spec[1].set_title("Spatial spectra")
    ax_spec[1].set_xlabel(r"$\Re(\alpha)$")
    ax_spec[1].set_ylabel(r"$\Im(\alpha)$")
    ax_spec[1].grid(True)

    # Temporal modes (first four unstable modes by imaginary part).
    iplot_temporal = np.argsort(np.imag(temporal_omega))[::-1][:4]
    for i, p in enumerate(iplot_temporal):
        if i >= 1:
            break
        mode = temporal_modes[:, int(p)]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, idx_block, label in [
            (axes.flat[0], idx_rho, r"$\rho$"),
            (axes.flat[1], idx_u, r"$u$"),
            (axes.flat[2], idx_v, r"$v$"),
            (axes.flat[3], idx_w, r"$w$"),
            (axes.flat[4], idx_T, r"$T$"),
        ]:
            block = mode[np.asarray(idx_block)]
            field = reconstruct_floquet(
                X,
                Y,
                block,
                {},
                used_ind=used_ind,
                floquet_exp=floquet_exp,
            ).reshape(X.shape, order="F")
            plot_flow(X, Y, np.real(field), ax=ax, levels=51, cmap="RdBu_r")
            ax.set_title(f"{label}, mode {int(p)}")
        fig.suptitle(f"Temporal mode index {int(p)}")
        fig.tight_layout()
        if i == 0:
            fig.savefig(output_png_temporal, dpi=600, bbox_inches="tight")
            plt.close(fig)
    print(f"Temporal mode figure saved to {output_png_temporal}")

    # Spatial modes (first five smallest |alpha|).
    iplot_spatial = np.argsort(np.abs(spatial_alpha))[:min(5, n_modes)]
    n_states = int(np.max(np.hstack([idx_u, idx_v, idx_w, idx_rho, idx_T])) + 1)
    for i, p in enumerate(iplot_spatial):
        if i >= 1:
            break
        mode = spatial_modes[:n_states, int(p)]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, idx_block, label in [
            (axes.flat[0], idx_rho, r"$\rho$"),
            (axes.flat[1], idx_u, r"$u$"),
            (axes.flat[2], idx_v, r"$v$"),
            (axes.flat[3], idx_w, r"$w$"),
            (axes.flat[4], idx_T, r"$T$"),
        ]:
            mode_var = mode[np.asarray(idx_block)]
            field = reconstruct_floquet(
                X,
                Y,
                mode_var,
                {},
                used_ind=used_ind,
                floquet_exp=floquet_exp,
            )
            plot_flow(X, Y, np.real(field), ax=ax, levels=51, cmap="RdBu_r")
            ax.set_title(f"{label}, α={spatial_alpha[int(p)]:.3f}")
        fig.suptitle(f"Spatial mode index {int(p)}")
        fig.tight_layout()
        if i == 0:
            fig.savefig(output_png_spatial, dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"Spatial mode figure saved to {output_png_spatial}")

    fig_spec.tight_layout()
    fig_spec.savefig(output_png_spectra, dpi=600, bbox_inches="tight")
    plt.close(fig_spec)
    print(
        "Figures saved to:\n"
        f"- Temporal: {output_png_temporal}\n"
        f"- Spatial: {output_png_spatial}\n"
        f"- Spectra: {output_png_spectra}"
    )


def main():
    try:
        np.load(OUTPUT_NPZ)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        _compute()
        np.load(OUTPUT_NPZ)

    _plot()


if __name__ == "__main__":
    main()
