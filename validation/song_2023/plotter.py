#!/usr/bin/env python3
"""Plot eigenvalues and resolvent modes from a song_2023 .npz output.

Reads the compressed .npz saved by song_2023.py and generates:
  - <output_dir>/eigvals/eig_0.png, eig_1.png, ... (spectrum + eigenmode contours)
  - <output_dir>/resolvent/gains.png, mode_0.png, mode_1.png, ...

Usage:
  python validation/song_2023/plotter.py --npz validation/song_2023/song_2023.npz
  python validation/song_2023/plotter.py --npz path/to/output.npz --output-dir my_plots --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Font settings
matplotlib.rcParams.update({
    "font.size": 22,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.usetex": True,
})


# Contour levels for scalar fields (symmetric about 0)
LEVELS = np.linspace(-1, 1, 11) * 0.5
LEVELS = np.delete(LEVELS, np.where(LEVELS == 0)[0])
SPONGE_CONTOUR = 1e-6


def _build_idx(ndof: int):
    """Build index arrays for rho, u, v, w, T from total dof count (5 * ngp)."""
    ngp = ndof // 5
    return {
        "idx_rho_j": np.arange(0, ngp),
        "idx_u_j": np.arange(ngp, 2 * ngp),
        "idx_v_j": np.arange(2 * ngp, 3 * ngp),
        "idx_w_j": np.arange(3 * ngp, 4 * ngp),
        "idx_T_j": np.arange(4 * ngp, 5 * ngp),
    }


def _get_2d_shape(X: np.ndarray) -> tuple[int, int]:
    """Return (ny, nx) for contour plots from mesh arrays."""
    return X.shape[0], X.shape[1]


def plot_spectrum(
    data: np.lib.npyio.NpzFile,
    save_dir: Path,
    *,
    beta: float = 0.0,
    St: float = 0.0,
    dpi: int = 600,
) -> None:
    """Plot eigenvalue spectrum and eigenmode contours (like plot_spectrum.py)."""
    omega = np.asarray(data["omega"]).flatten()
    if omega.size == 0:
        print("[plotter] No eigenvalues in npz, skipping spectrum plots.")
        return

    X = np.asarray(data["X"])
    Y = np.asarray(data["Y"])
    # run_case saves mesh.X, mesh.Y with shape (ny, nx)
    x = X
    y = Y
    ny, nx = _get_2d_shape(x)
    u_velocity = np.asarray(data["U"])
    sponge = np.asarray(data["sponge"])
    sponge_flag = float(np.max(sponge)) >= SPONGE_CONTOUR

    direct_modes = np.asarray(data["direct_modes"])
    ndof, n_modes = direct_modes.shape
    idx = _build_idx(ndof)
    eigenvectors = direct_modes.T  # (n_modes, ndof)

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plotter] Writing eigenvalue images to {save_dir} ...")

    for i in range(len(omega)):
        rho = np.asarray(eigenvectors[i, idx["idx_rho_j"]], dtype=complex)
        u = np.asarray(eigenvectors[i, idx["idx_u_j"]], dtype=complex)
        v = np.asarray(eigenvectors[i, idx["idx_v_j"]], dtype=complex)
        w = np.asarray(eigenvectors[i, idx["idx_w_j"]], dtype=complex)
        T = np.asarray(eigenvectors[i, idx["idx_T_j"]], dtype=complex)

        max_rho = np.max(np.abs(rho.real))
        max_u = np.max(np.abs(u.real))
        max_v = np.max(np.abs(v.real))
        max_T = np.max(np.abs(T.real))
        if max_rho > 0:
            rho = rho / max_rho
        if max_u > 0:
            u = u / max_u
        if max_v > 0:
            v = v / max_v
        if max_T > 0:
            T = T / max_T
        if beta > 1e-8:
            max_w = np.max(np.abs(w.real))
            if max_w > 0:
                w = w / max_w

        rho = rho.reshape((ny, nx), order="F")
        u = u.reshape((ny, nx), order="F")
        v = v.reshape((ny, nx), order="F")
        w = w.reshape((ny, nx), order="F")
        T = T.reshape((ny, nx), order="F")

        fig, ((ax1, ax11), (ax2, ax21), (ax3, ax31)) = plt.subplots(3, 2, figsize=(16, 10))
        axes_list = [ax1, ax11, ax2, ax21, ax3, ax31]

        ax1.scatter(omega.real, omega.imag, color="black", s=30)
        ax1.scatter(omega.real[i], omega.imag[i], color="blue", s=60)
        ax1.axhline(y=0, color="gray", linewidth=1)
        ax1.set_xlabel(r"$\omega_r$", labelpad=-10)
        ax1.set_ylabel(r"$\omega_i$")
        ax1.set_title(f"{omega[i]}")

        for ax, field in [(ax11, rho), (ax2, u), (ax21, v), (ax3, w), (ax31, T)]:
            cf = ax.contourf(x, y, field.real, cmap="coolwarm", levels=LEVELS, extend="both")
            ax.contour(x, y, u_velocity, levels=[-1e-6], colors=["blue"], linewidths=[1])
            ax.set_aspect("equal")
            if sponge_flag:
                ax.contour(x, y, sponge, levels=[SPONGE_CONTOUR], colors=["black"], linewidths=[1])
        ax11.set_ylabel(r"$R(\rho/\rho_{max})$")
        ax2.set_ylabel(r"$R(u/u_{max})$")
        ax21.set_ylabel(r"$R(v/v_{max})$")
        ax3.set_ylabel(r"$R(w/w_{max})$")
        ax31.set_ylabel(r"$R(T/T_{max})$")

        fig.suptitle(rf"$\beta = {beta}$, $St = {St}$")
        cbar = fig.colorbar(cf, ax=axes_list, orientation="vertical", location="right", shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label("Scale - real part, normalized by max(real)")

        fig.savefig(save_dir / f"eig_{i}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  eig_{i}.png")


def plot_resolvent(
    data: np.lib.npyio.NpzFile,
    save_dir: Path,
    *,
    beta: float = 0.0,
    St: float = 0.0,
    dpi: int = 600,
) -> None:
    """Plot resolvent gains and forcing/response mode contours (like plot_resolvent_modes.py)."""
    gains = np.ndarray.flatten(np.asarray(data["gains"]))
    if gains.size == 0:
        print("[plotter] No resolvent gains in npz, skipping resolvent plots.")
        return

    X = np.asarray(data["X"])
    Y = np.asarray(data["Y"])
    x = X
    y = Y
    ny, nx = _get_2d_shape(x)
    u_velocity = np.asarray(data["U"])
    sponge = np.asarray(data["sponge"])
    sponge_flag = float(np.max(sponge)) >= SPONGE_CONTOUR

    forcings = np.asarray(data["forcings"]).T   # (n_modes, ndof)
    responses = np.asarray(data["responses"]).T
    ndof = forcings.shape[1]
    idx = _build_idx(ndof)
    gains_orig = gains.copy()
    gain_total = np.sum(gains)
    if gain_total > 0:
        gains_norm = gains / gain_total
    else:
        gains_norm = gains
    gain_sum = np.cumsum(gains_norm)

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plotter] Writing resolvent images to {save_dir} ...")

    # Gains bar chart
    fig, ax = plt.subplots()
    ax.bar(range(1, len(gains) + 1), gains_norm, color="blue")
    ax.plot(range(1, len(gains) + 1), gain_sum, "-k")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Normalized Gain")
    fig.savefig(save_dir / "gains.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  gains.png")

    for i in range(len(gains)):
        rho_f = np.asarray(forcings[i, idx["idx_rho_j"]], dtype=complex)
        u_f = np.asarray(forcings[i, idx["idx_u_j"]], dtype=complex)
        v_f = np.asarray(forcings[i, idx["idx_v_j"]], dtype=complex)
        w_f = np.asarray(forcings[i, idx["idx_w_j"]], dtype=complex)
        T_f = np.asarray(forcings[i, idx["idx_T_j"]], dtype=complex)
        rho_r = np.asarray(responses[i, idx["idx_rho_j"]], dtype=complex)
        u_r = np.asarray(responses[i, idx["idx_u_j"]], dtype=complex)
        v_r = np.asarray(responses[i, idx["idx_v_j"]], dtype=complex)
        w_r = np.asarray(responses[i, idx["idx_w_j"]], dtype=complex)
        T_r = np.asarray(responses[i, idx["idx_T_j"]], dtype=complex)

        max_rho = max(np.max(np.abs(rho_f.real)), np.max(np.abs(rho_r.real))) or 1.0
        max_u = max(np.max(np.abs(u_f.real)), np.max(np.abs(u_r.real))) or 1.0
        max_v = max(np.max(np.abs(v_f.real)), np.max(np.abs(v_r.real))) or 1.0
        max_T = max(np.max(np.abs(T_f.real)), np.max(np.abs(T_r.real))) or 1.0
        rho_f, rho_r = rho_f / max_rho, rho_r / max_rho
        u_f, u_r = u_f / max_u, u_r / max_u
        v_f, v_r = v_f / max_v, v_r / max_v
        T_f, T_r = T_f / max_T, T_r / max_T
        if beta > 1e-8:
            max_w = max(np.max(np.abs(w_f.real)), np.max(np.abs(w_r.real))) or 1.0
            w_f, w_r = w_f / max_w, w_r / max_w

        def reshape_f(v):
            return v.reshape((ny, nx), order="F")
        rho_f = reshape_f(rho_f)
        u_f = reshape_f(u_f)
        v_f = reshape_f(v_f)
        w_f = reshape_f(w_f)
        T_f = reshape_f(T_f)
        rho_r = reshape_f(rho_r)
        u_r = reshape_f(u_r)
        v_r = reshape_f(v_r)
        w_r = reshape_f(w_r)
        T_r = reshape_f(T_r)

        fig, ((ax1, ax11), (ax2, ax21), (ax3, ax31), (ax4, ax41), (ax5, ax51)) = plt.subplots(5, 2, sharey=True, figsize=(12, 16))
        axes_list = [ax1, ax11, ax2, ax21, ax3, ax31, ax4, ax41, ax5, ax51]

        def add_contours(ax, field):
            cf = ax.contourf(x, y, field.real, cmap="coolwarm", levels=LEVELS, extend="both")
            ax.contour(x, y, u_velocity, levels=[-1e-6], colors=["blue"], linewidths=[1])
            if sponge_flag:
                ax.contour(x, y, sponge, levels=[SPONGE_CONTOUR], colors=["black"], linewidths=[1])
            ax.set_aspect("equal")
            return cf

        add_contours(ax1, rho_f)
        add_contours(ax11, rho_r)
        add_contours(ax2, u_f)
        add_contours(ax21, u_r)
        add_contours(ax3, v_f)
        add_contours(ax31, v_r)
        add_contours(ax4, w_f)
        add_contours(ax41, w_r)
        add_contours(ax5, T_f)
        cf = add_contours(ax51, T_r)

        ax1.set_ylabel(r"$\rho$")
        ax2.set_ylabel("$u$")
        ax3.set_ylabel("$v$")
        ax4.set_ylabel("$w$")
        ax5.set_ylabel("$T$")
        ax5.set_xlabel("Forcing")
        ax51.set_xlabel("Response")

        pct = gains_norm[i] * 100 if gain_total > 0 else 0.0
        fig.suptitle(rf"$\beta = {beta}$, $St = {St}$ — {pct:.3f}% of total gain, {gains_orig[i]:.7e}", y=0.925)
        # Leave right margin for colorbar so it does not overlap the response column
        plt.subplots_adjust(hspace=0.1, wspace=0.1, right=0.82)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
        cbar = fig.colorbar(cf, cax=cbar_ax)
        cbar.set_label("Scale - real part, normalized by max(forcing,response)")
        fig.savefig(save_dir / f"mode_{i}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  mode_{i}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot eigenvalues and resolvent modes from a run_case .npz file.",
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to the .npz file (e.g. song_2023.npz)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plots (default: <npz_stem>_plots/eigvals and .../resolvent)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI (default: 600)")
    parser.add_argument("--no-spectrum", action="store_true", help="Skip eigenvalue/spectrum plots")
    parser.add_argument("--no-resolvent", action="store_true", help="Skip resolvent mode plots")
    parser.add_argument("--beta", type=float, default=None, help="Spanwise wavenumber (overrides npz; use if npz was saved before beta was stored)")
    parser.add_argument("--St", type=float, default=None, help="Strouhal number (overrides npz)")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz file not found: {npz_path}")

    if args.output_dir is not None:
        out_root = Path(args.output_dir)
    else:
        out_root = Path(npz_path.stem + "_plots")

    data = np.load(npz_path, allow_pickle=False)
    beta = args.beta if args.beta is not None else (float(data["beta"]) if "beta" in data else 0.0)
    St = args.St if args.St is not None else (float(data["St"]) if "St" in data else (float(data["omega_target"]) / (2 * np.pi) if "omega_target" in data else 0.0))

    if not args.no_spectrum:
        plot_spectrum(data, out_root / "eigvals", beta=beta, St=St, dpi=args.dpi)
    if not args.no_resolvent:
        plot_resolvent(data, out_root / "resolvent", beta=beta, St=St, dpi=args.dpi)

    print(f"[plotter] Done. Plots under {out_root.resolve()}")


if __name__ == "__main__":
    main()
