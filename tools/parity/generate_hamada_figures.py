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
        default=Path("example_6_hamada/MATLAB_results/spectrum.mat"),
    )
    parser.add_argument(
        "--matlab-mode",
        type=Path,
        default=Path("example_6_hamada/MATLAB_results/eigenvector_7.mat"),
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
    ax.scatter(mat_sel.real, mat_sel.imag, s=16, label="MATLAB", alpha=0.8)
    ax.scatter(py_sel.real, py_sel.imag, s=16, label="Python", alpha=0.8)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Re(omega)")
    ax.set_ylabel("Im(omega)")
    ax.set_title(f"Hamada spectrum parity (n={n})")
    ax.legend()
    fig.tight_layout()
    spectrum_png = root / "example_6_hamada_vs_matlab_spectrum.png"
    fig.savefig(spectrum_png, dpi=220)
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

    print(f"generated: {spectrum_png}")
    if "X" in data and "Y" in data and "U" in data:
        print(f"generated: {preview_png}")
    print(f"generated: {mode_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
