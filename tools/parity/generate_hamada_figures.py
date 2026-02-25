#!/usr/bin/env python3
"""Generate Hamada parity figures from Python NPZ and MATLAB references."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def _diag_or_flat(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return np.diag(arr).astype(complex)
    return arr.reshape(-1).astype(complex)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-npz",
        type=Path,
        default=Path("example_6_hamada.npz"),
    )
    parser.add_argument(
        "--matlab-spectrum",
        type=Path,
        default=Path("example_6_hamada/MATLAB_exports/spectrum.mat"),
    )
    parser.add_argument(
        "--matlab-mode",
        type=Path,
        default=Path("example_6_hamada/MATLAB_exports/eigenvector_7.mat"),
    )
    args = parser.parse_args()

    root = Path(".")
    data = np.load(args.python_npz, allow_pickle = False)

    omega_py = np.asarray(data["omega"], dtype=complex).reshape(-1)
    omega_target = complex(np.asarray(data["omega_target"]).reshape(-1)[0])
    omega_mat = _diag_or_flat(np.asarray(loadmat(args.matlab_spectrum)["omega"]))

    n = int(min(100, omega_py.size, omega_mat.size))
    py_sel = omega_py[np.argsort(np.abs(omega_py - omega_target))[:n]]
    mat_sel = omega_mat[np.argsort(np.abs(omega_mat - omega_target))[:n]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        mat_sel.real,
        mat_sel.imag,
        s=72,
        marker="x",
        linewidths=1.3,
        color="#D55E00",
        label="MATLAB",
        alpha=0.8,
        zorder=6,
    )
    ax.scatter(
        py_sel.real,
        py_sel.imag,
        s=52,
        marker="o",
        facecolors="none",
        edgecolors="#0072B2",
        linewidths=1.7,
        label="Python",
        alpha=0.95,
        zorder=10,
    )
    ax.scatter(
        py_sel.real,
        py_sel.imag,
        s=12,
        marker="o",
        color="#0072B2",
        linewidths=0.0,
        alpha=0.95,
        zorder=11,
    )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Re(omega)")
    ax.set_ylabel("Im(omega)")
    ax.set_title(f"Hamada spectrum parity (n={n})")
    ax.legend()
    fig.tight_layout()
    spectrum_png = root / "example_6_hamada_vs_matlab_spectrum.png"
    fig.savefig(spectrum_png, dpi=220)
    plt.close(fig)

    # Leading-mode comparison by rank around target frequency.
    rank = np.arange(1, n + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(rank, mat_sel.real, "x-", markersize=5.5, linewidth=0.9, label="MATLAB")
    axes[0].plot(rank, py_sel.real, "o-", markersize=3.0, linewidth=0.8, label="Python")
    axes[0].set_ylabel("Re(omega)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(rank, mat_sel.imag, "x-", markersize=5.5, linewidth=0.9, label="MATLAB")
    axes[1].plot(rank, py_sel.imag, "o-", markersize=3.0, linewidth=0.8, label="Python")
    axes[1].set_xlabel("mode rank (nearest to target)")
    axes[1].set_ylabel("Im(omega)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    leading_png = root / "example_6_hamada_leading_modes.png"
    fig.savefig(leading_png, dpi=220)
    plt.close(fig)

    if "X" in data and "Y" in data and "U" in data:
        X = np.asarray(data["X"], dtype=float)
        Y = np.asarray(data["Y"], dtype=float)
        U = np.asarray(data["U"], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 4))
        c = ax.pcolormesh(X, Y, U, shading="auto")
        fig.colorbar(c, ax=ax, label="U")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Hamada baseflow preview (U)")
        fig.tight_layout()
        preview_png = root / "example_6_hamada_preview.png"
        fig.savefig(preview_png, dpi=220)
        plt.close(fig)

    direct = np.asarray(data["direct_modes"], dtype=complex)
    idx_u = np.asarray(data["idx_u_j"], dtype=int)
    mode_mat = np.asarray(loadmat(args.matlab_mode)["u"], dtype=complex).reshape(-1)

    mode_idx_mat = min(6, omega_mat.size - 1)
    omega_ref = omega_mat[mode_idx_mat]
    mode_idx_py = int(np.argmin(np.abs(omega_py - omega_ref)))
    u_py = direct[:, mode_idx_py][idx_u].reshape(-1)

    num = np.vdot(u_py, mode_mat)
    if num != 0:
        u_py = u_py * np.exp(-1j * np.angle(num))

    u_mat_n = mode_mat / max(np.linalg.norm(mode_mat), 1e-30)
    u_py_n = u_py / max(np.linalg.norm(u_py), 1e-30)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(np.real(u_mat_n), linewidth=0.9)
    axes[0].set_ylabel("Re(u)")
    axes[0].set_title("MATLAB mode 7 (normalized)")
    axes[1].plot(np.real(u_py_n), linewidth=0.9)
    axes[1].set_ylabel("Re(u)")
    axes[1].set_title(f"Python nearest mode ({mode_idx_py + 1})")
    axes[1].set_xlabel("grid dof index")
    fig.tight_layout()
    mode_png = root / "example_6_hamada_vs_matlab_mode7.png"
    fig.savefig(mode_png, dpi=220)
    plt.close(fig)

    modes2d_png = None
    mode7_2d_png = None
    if "X" in data and "Y" in data and direct.size and idx_u.size:
        X = np.asarray(data["X"], dtype=float)
        Y = np.asarray(data["Y"], dtype=float)
        n_modes_plot = int(min(4, omega_py.size, direct.shape[1]))
        mode_order = np.argsort(np.abs(omega_py - omega_target))[:n_modes_plot]

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)
        for ax, mode_idx in zip(axes, mode_order):
            u_mode = direct[:, mode_idx][idx_u].reshape(X.shape, order="F")
            vmax = np.max(np.abs(np.real(u_mode)))
            vmax = max(float(vmax), 1e-12)
            c = ax.pcolormesh(
                X,
                Y,
                np.real(u_mode),
                shading="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            om = omega_py[mode_idx]
            ax.set_title(
                f"Python mode {mode_idx + 1}\n"
                f"Re={om.real:.5f}, Im={om.imag:.5f}"
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(c, ax=ax, shrink=0.86)

        for ax in axes[n_modes_plot:]:
            ax.axis("off")

        fig.tight_layout()
        modes2d_png = root / "example_6_hamada_modes2d.png"
        fig.savefig(modes2d_png, dpi=220)
        plt.close(fig)

    if "X" in data and "Y" in data and mode_mat.size == int(np.asarray(data["X"]).size):
        X = np.asarray(data["X"], dtype=float)
        Y = np.asarray(data["Y"], dtype=float)
        u_mat_grid = u_mat_n.reshape(X.shape, order="F")
        u_py_grid = u_py_n.reshape(X.shape, order="F")
        iy = X.shape[0] // 2
        ix = X.shape[1] // 2

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        axes[0, 0].plot(X[iy, :], np.real(u_mat_grid[iy, :]), "x-", markersize=5.0, linewidth=0.9, label="MATLAB")
        axes[0, 0].plot(X[iy, :], np.real(u_py_grid[iy, :]), "o-", markersize=2.7, linewidth=0.8, label="Python")
        axes[0, 0].set_title("centerline cut in x: Re(u)")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("Re(u)")
        axes[0, 0].grid(alpha=0.25)
        axes[0, 0].legend()

        axes[0, 1].plot(Y[:, ix], np.real(u_mat_grid[:, ix]), "x-", markersize=5.0, linewidth=0.9, label="MATLAB")
        axes[0, 1].plot(Y[:, ix], np.real(u_py_grid[:, ix]), "o-", markersize=2.7, linewidth=0.8, label="Python")
        axes[0, 1].set_title("centerline cut in y: Re(u)")
        axes[0, 1].set_xlabel("y")
        axes[0, 1].set_ylabel("Re(u)")
        axes[0, 1].grid(alpha=0.25)
        axes[0, 1].legend()

        axes[1, 0].plot(X[iy, :], np.abs(u_mat_grid[iy, :]), "x-", markersize=5.0, linewidth=0.9, label="MATLAB")
        axes[1, 0].plot(X[iy, :], np.abs(u_py_grid[iy, :]), "o-", markersize=2.7, linewidth=0.8, label="Python")
        axes[1, 0].set_title("centerline cut in x: |u|")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("|u|")
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend()

        axes[1, 1].plot(Y[:, ix], np.abs(u_mat_grid[:, ix]), "x-", markersize=5.0, linewidth=0.9, label="MATLAB")
        axes[1, 1].plot(Y[:, ix], np.abs(u_py_grid[:, ix]), "o-", markersize=2.7, linewidth=0.8, label="Python")
        axes[1, 1].set_title("centerline cut in y: |u|")
        axes[1, 1].set_xlabel("y")
        axes[1, 1].set_ylabel("|u|")
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend()

        fig.tight_layout()
        cuts_png = root / "example_6_hamada_mode7_cuts.png"
        fig.savefig(cuts_png, dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
        vmax = max(
            float(np.max(np.abs(np.real(u_mat_grid)))),
            float(np.max(np.abs(np.real(u_py_grid)))),
            1e-12,
        )
        c0 = axes[0].pcolormesh(
            X,
            Y,
            np.real(u_mat_grid),
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[0].set_title("MATLAB mode 7: Re(u)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        c1 = axes[1].pcolormesh(
            X,
            Y,
            np.real(u_py_grid),
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[1].set_title(f"Python matched mode {mode_idx_py + 1}: Re(u)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        fig.colorbar(c1, ax=axes.ravel().tolist(), shrink=0.92, label="Re(u)")
        fig.tight_layout()
        mode7_2d_png = root / "example_6_hamada_mode7_2d_compare.png"
        fig.savefig(mode7_2d_png, dpi=220)
        plt.close(fig)
    else:
        cuts_png = None

    print(f"generated: {spectrum_png}")
    print(f"generated: {leading_png}")
    if "X" in data and "Y" in data and "U" in data:
        print(f"generated: {preview_png}")
    print(f"generated: {mode_png}")
    if cuts_png is not None:
        print(f"generated: {cuts_png}")
    if modes2d_png is not None:
        print(f"generated: {modes2d_png}")
    if mode7_2d_png is not None:
        print(f"generated: {mode7_2d_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
