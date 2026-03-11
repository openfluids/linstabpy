#!/usr/bin/env python3
"""
case1_dashboard: Build paper-ready Case 1 figures from exported MATLAB data.

OUTPUTS:
  case1_dashboard.npz
  case1_dashboard.png
  eigenvalue_spectrum.png
  mode_shapes.png
  resolvent_gains.png
  case1_modes.png

USAGE:
  python case1_dashboard.py                        # .npz exists -> regenerate plots
  rm case1_dashboard.npz && python case1_dashboard.py  # recompute cache + plots
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
CASE_DIR = ROOT / "results" / "case1"

OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")

OUT_SPECTRUM = SCRIPT.with_name("eigenvalue_spectrum.png")
OUT_MODE_SHAPES = SCRIPT.with_name("mode_shapes.png")
OUT_RES_GAINS = SCRIPT.with_name("resolvent_gains.png")
OUT_CASE1_MODES = SCRIPT.with_name("case1_modes.png")


def _compound_to_complex(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.fields and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return np.asarray(arr["real"], dtype=float) + 1j * np.asarray(arr["imag"], dtype=float)
    return np.asarray(arr, dtype=complex)


def _load_mat_spectrum(path: Path) -> np.ndarray:
    mat = loadmat(path)
    omega = np.asarray(mat["omega"])
    if omega.ndim == 2 and omega.shape[0] == omega.shape[1]:
        return np.diag(omega).astype(complex)
    return omega.reshape(-1).astype(complex)


def _load_mode_fields(path: Path) -> dict[str, np.ndarray]:
    mat = loadmat(path)
    return {
        "u": np.asarray(mat["u"]).reshape(-1).astype(complex),
        "v": np.asarray(mat["v"]).reshape(-1).astype(complex),
        "T": np.asarray(mat["T"]).reshape(-1).astype(complex),
    }


def _maybe_load_python_npz() -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        ROOT / "results" / "case1" / "case1.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = np.load(path, allow_pickle = False)
        omega_py = np.asarray(data.get("omega", np.array([], dtype=complex))).reshape(-1).astype(complex)
        gains_py = np.asarray(data.get("gains", np.array([], dtype=float))).reshape(-1).astype(float)
        return omega_py, gains_py
    return np.array([], dtype=complex), np.array([], dtype=float)


def _load_resolvent_data(svd_path: Path, idx_path: Path, cfg_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(svd_path, "r") as h:
        s_vals = np.asarray(h["S"]).reshape(-1).astype(float)
        u_modes = _compound_to_complex(np.asarray(h["U"]))
    with h5py.File(idx_path, "r") as h:
        idx_u = np.asarray(h["idx/u_j"]).reshape(-1).astype(int) - 1
    omega_forcing = np.nan
    with h5py.File(cfg_path, "r") as h:
        if "res_config/omega" in h:
            omega_forcing = float(np.asarray(h["res_config/omega"]).reshape(-1)[0])

    if u_modes.size == 0 or idx_u.size == 0:
        return s_vals, np.array([], dtype=complex), omega_forcing
    u_resp = np.asarray(u_modes[0, idx_u], dtype=complex).reshape(-1)
    return s_vals, u_resp, omega_forcing


def _safe_profile(field: np.ndarray, shape: tuple[int, int], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if field.size != shape[0] * shape[1]:
        idx = np.arange(field.size, dtype=float)
        mag = np.abs(field)
        denom = max(float(np.max(mag)), 1e-30)
        return idx, mag / denom
    f2d = field.reshape(shape, order="F")
    y_prof = np.mean(y, axis=1)
    val = np.mean(np.abs(f2d), axis=1)
    order = np.argsort(y_prof)
    y_sorted = y_prof[order]
    v_sorted = val[order]
    denom = max(float(np.max(v_sorted)), 1e-30)
    return y_sorted, v_sorted / denom


def _placeholder_axes(ax: plt.Axes, title: str, detail: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"{title}\n\n{detail}",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )


def compute() -> None:
    x_path = CASE_DIR / "x.npy"
    y_path = CASE_DIR / "y.npy"
    spectrum_path = CASE_DIR / "MATLAB_exports/spectrum.mat"
    mode_path = CASE_DIR / "MATLAB_exports/eigenvector_7.mat"
    res_svd_path = CASE_DIR / "MATLAB_exports_resolvent/res_svd_final.mat"
    idx_path = CASE_DIR / "MATLAB_exports_resolvent/indices_and_bcs.mat"
    cfg_path = CASE_DIR / "MATLAB_exports_resolvent/res_config.mat"

    X = np.load(x_path, allow_pickle = False) if x_path.exists() else np.empty((0, 0))
    Y = np.load(y_path, allow_pickle = False) if y_path.exists() else np.empty((0, 0))

    omega_mat = _load_mat_spectrum(spectrum_path) if spectrum_path.exists() else np.array([], dtype=complex)
    mode = _load_mode_fields(mode_path) if mode_path.exists() else {"u": np.array([], dtype=complex), "v": np.array([], dtype=complex), "T": np.array([], dtype=complex)}

    s_vals = np.array([], dtype=float)
    u_resp = np.array([], dtype=complex)
    omega_forcing = np.nan
    if res_svd_path.exists() and idx_path.exists() and cfg_path.exists():
        s_vals, u_resp, omega_forcing = _load_resolvent_data(res_svd_path, idx_path, cfg_path)

    omega_py, gains_py = _maybe_load_python_npz()

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        Y=Y,
        omega_mat=omega_mat,
        omega_py=omega_py,
        u_mode=mode["u"],
        v_mode=mode["v"],
        T_mode=mode["T"],
        s_vals=s_vals,
        gains_py=gains_py,
        u_resp=u_resp,
        omega_forcing=np.array([omega_forcing], dtype=float),
    )


def _plot_spectrum(data: np.lib.npyio.NpzFile) -> None:
    omega_mat = np.asarray(data["omega_mat"]).reshape(-1).astype(complex)
    omega_py = np.asarray(data["omega_py"]).reshape(-1).astype(complex)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if omega_mat.size:
        ax.scatter(
            omega_mat.real,
            omega_mat.imag,
            marker="x",
            s=46,
            linewidths=1.1,
            color="#D55E00",
            label="MATLAB",
        )
    if omega_py.size:
        n = min(omega_mat.size, omega_py.size) if omega_mat.size else omega_py.size
        sel = np.arange(n, dtype=int)
        ax.scatter(
            omega_py[sel].real,
            omega_py[sel].imag,
            marker="o",
            s=24,
            facecolors="none",
            edgecolors="#0072B2",
            linewidths=1.0,
            label="Python",
        )

    ax.axhline(0.0, color="0.55", linewidth=0.8)
    ax.set_xlabel(r"Re($\omega$)")
    ax.set_ylabel(r"Im($\omega$)")
    ax.set_title("Case 1 temporal spectrum")
    if omega_mat.size or omega_py.size:
        ax.legend(loc="best")
    else:
        _placeholder_axes(ax, "Spectrum unavailable", "Missing MATLAB/Python spectrum files")
    fig.tight_layout()
    fig.savefig(OUT_SPECTRUM, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_mode_shapes(data: np.lib.npyio.NpzFile) -> None:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    u = np.asarray(data["u_mode"]).reshape(-1).astype(complex)
    v = np.asarray(data["v_mode"]).reshape(-1).astype(complex)
    T = np.asarray(data["T_mode"]).reshape(-1).astype(complex)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if X.size and Y.size and u.size and v.size and T.size:
        yy, uu = _safe_profile(u, X.shape, Y)
        _, vv = _safe_profile(v, X.shape, Y)
        _, tt = _safe_profile(T, X.shape, Y)
        ax.plot(yy, uu, linewidth=1.6, label=r"$|u'|$")
        ax.plot(yy, vv, linewidth=1.6, label=r"$|v'|$")
        ax.plot(yy, tt, linewidth=1.6, label=r"$|T'|$")
        ax.set_xlabel(r"$y$")
        ax.set_ylabel("normalized magnitude")
        ax.set_title("Case 1 leading-mode profiles (MATLAB export)")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
    else:
        _placeholder_axes(ax, "Mode profiles unavailable", "Missing mode or grid files")
    fig.tight_layout()
    fig.savefig(OUT_MODE_SHAPES, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_resolvent_gains(data: np.lib.npyio.NpzFile) -> None:
    s_vals = np.asarray(data["s_vals"], dtype=float).reshape(-1)
    gains_py = np.asarray(data["gains_py"], dtype=float).reshape(-1)
    omega_forcing = float(np.asarray(data["omega_forcing"]).reshape(-1)[0])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if s_vals.size:
        idx = np.arange(1, s_vals.size + 1, dtype=int)
        ax.semilogy(idx, s_vals, "o-", markersize=4.0, linewidth=1.2, label="MATLAB")
        if gains_py.size:
            n = min(gains_py.size, s_vals.size)
            ax.semilogy(
                idx[:n],
                gains_py[:n],
                "s--",
                markersize=3.6,
                linewidth=1.0,
                label="Python",
            )
        ax.set_xlabel("mode index")
        ax.set_ylabel(r"singular value $\sigma_j$")
        if np.isfinite(omega_forcing):
            ax.set_title(rf"Case 1 resolvent gains ($\omega_f={omega_forcing:.4g}$)")
        else:
            ax.set_title("Case 1 resolvent gains")
        ax.legend(loc="best")
        ax.grid(alpha=0.25, which="both")
    else:
        _placeholder_axes(ax, "Resolvent gains unavailable", "Missing res_svd_final.mat")
    fig.tight_layout()
    fig.savefig(OUT_RES_GAINS, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_case1_modes(data: np.lib.npyio.NpzFile) -> None:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    u_mode = np.asarray(data["u_mode"]).reshape(-1).astype(complex)
    u_resp = np.asarray(data["u_resp"]).reshape(-1).astype(complex)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)

    if X.size and Y.size and u_mode.size == X.size:
        eig = np.real(u_mode.reshape(X.shape, order="F"))
        denom = max(float(np.max(np.abs(eig))), 1e-30)
        eig = eig / denom
        c0 = axes[0].pcolormesh(X, Y, eig, shading="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[0].set_title("Leading eigenvector Re($u'$)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(c0, ax=axes[0], fraction=0.046, pad=0.04)
    else:
        _placeholder_axes(axes[0], "Eigenvector map unavailable", "Missing mode or grid files")

    if X.size and Y.size and u_resp.size == X.size:
        resp = np.abs(u_resp.reshape(X.shape, order="F"))
        denom = max(float(np.max(resp)), 1e-30)
        resp = resp / denom
        c1 = axes[1].pcolormesh(X, Y, resp, shading="auto", cmap="viridis")
        axes[1].set_title("Resolvent response $|u'|$")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        _placeholder_axes(axes[1], "Resolvent map unavailable", "Missing resolvent mode or grid files")

    fig.savefig(OUT_CASE1_MODES, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_dashboard() -> None:
    panels = [
        (OUT_SPECTRUM, "Temporal spectrum"),
        (OUT_MODE_SHAPES, "Mode profiles"),
        (OUT_RES_GAINS, "Resolvent gains"),
        (OUT_CASE1_MODES, "Spatial mode maps"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (path, title) in zip(axes, panels):
        if path.exists():
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        else:
            _placeholder_axes(ax, title, "panel missing")
    fig.suptitle("Case 1 paper figure dashboard", fontsize=12)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot(data: np.lib.npyio.NpzFile) -> None:
    _plot_spectrum(data)
    _plot_mode_shapes(data)
    _plot_resolvent_gains(data)
    _plot_case1_modes(data)
    _plot_dashboard()


def main() -> None:
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    except FileNotFoundError:
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
