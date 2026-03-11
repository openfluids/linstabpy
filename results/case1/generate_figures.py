#!/usr/bin/env python3
"""Generate Hamada parity figures from Python NPZ and MATLAB references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

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


def _best_complex_scale(reference: np.ndarray, candidate: np.ndarray) -> complex:
    denom = np.vdot(candidate, candidate)
    if denom == 0:
        return complex(1.0)
    return np.vdot(candidate, reference) / denom


def _load_jsonc(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|[^:])//.*$", r"\1", text, flags=re.M)
    return json.loads(text)


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
    parser.add_argument(
        "--matlab-resolvent",
        type=Path,
        default=Path("example_6_hamada/MATLAB_exports_resolvent/res_svd_final.mat"),
    )
    parser.add_argument(
        "--parity-config",
        type=Path,
        default=Path("example_6_hamada/parity_config.jsonc"),
    )
    args = parser.parse_args()

    root = Path(".")
    data = np.load(args.python_npz, allow_pickle = False)
    stability_metrics: dict[str, float] = {}
    resolvent_metrics: dict[str, float] = {}

    use_matlab_sponge_aux = False
    if args.parity_config.exists():
        cfg = _load_jsonc(args.parity_config)
        use_matlab_sponge_aux = bool(
            cfg.get("physics", {}).get("use_matlab_sponge_aux", False)
        )

    omega_py = np.asarray(data["omega"], dtype=complex).reshape(-1)
    omega_target = complex(np.asarray(data["omega_target"]).reshape(-1)[0])
    omega_mat = _diag_or_flat(np.asarray(loadmat(args.matlab_spectrum)["omega"]))
    has_stability = omega_py.size > 0
    if not has_stability:
        print(
            "skipped: stability spectrum/mode figures "
            "(omega array is empty; run_stability is likely disabled in the solver run)."
        )
        stale = [
            "example_6_hamada_vs_matlab_spectrum.png",
            "example_6_hamada_leading_modes.png",
            "example_6_hamada_vs_matlab_mode7.png",
            "example_6_hamada_mode7_cuts.png",
            "example_6_hamada_mode7_2d_compare.png",
            "example_6_hamada_modes2d.png",
        ]
        for name in stale:
            path = root / name
            if path.exists():
                path.unlink()
                print(f"removed stale stability figure: {path}")

    spectrum_png = None
    leading_png = None
    if has_stability and not use_matlab_sponge_aux:
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
    elif has_stability and use_matlab_sponge_aux:
        print(
            "skipped: spectrum/leading parity figures because "
            "physics.use_matlab_sponge_aux=True in parity config; "
            "current MATLAB spectrum reference is from a different operator setup."
        )
        for name in (
            "example_6_hamada_vs_matlab_spectrum.png",
            "example_6_hamada_leading_modes.png",
        ):
            path = root / name
            if path.exists():
                path.unlink()
                print(f"removed stale spectrum figure: {path}")

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

    mode_png = None
    cuts_png = None
    modes2d_png = None
    mode7_2d_png = None
    if has_stability:
        direct = np.asarray(data["direct_modes"], dtype=complex)
        idx_u = np.asarray(data["idx_u_j"], dtype=int)
        mode_mat = np.asarray(loadmat(args.matlab_mode)["u"], dtype=complex).reshape(-1)

        mode_idx_mat = min(6, omega_mat.size - 1)
        omega_ref = omega_mat[mode_idx_mat]
        mode_idx_py = int(np.argmin(np.abs(omega_py - omega_ref)))
        u_py = direct[:, mode_idx_py][idx_u].reshape(-1)

        u_py = u_py * _best_complex_scale(mode_mat, u_py)

        norm_ref = max(np.linalg.norm(mode_mat), 1e-30)
        u_mat_n = mode_mat / norm_ref
        u_py_n = u_py / norm_ref

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
        axes[0].plot(np.real(u_mat_n), linewidth=0.9)
        axes[0].set_ylabel("Re(u)")
        axes[0].set_title("MATLAB mode 7 (normalized by MATLAB L2 norm)")
        axes[1].plot(np.real(u_py_n), linewidth=0.9)
        axes[1].set_ylabel("Re(u)")
        axes[1].set_title(f"Python nearest mode ({mode_idx_py + 1}), same normalization")
        axes[1].set_xlabel("grid dof index")

        ymax = max(
            float(np.max(np.abs(np.real(u_mat_n)))),
            float(np.max(np.abs(np.real(u_py_n)))),
            1e-12,
        )
        for ax in axes:
            ax.set_ylim(-1.05 * ymax, 1.05 * ymax)

        fig.tight_layout()
        mode_png = root / "example_6_hamada_vs_matlab_mode7.png"
        fig.savefig(mode_png, dpi=220)
        plt.close(fig)

    if has_stability and "X" in data and "Y" in data and direct.size and idx_u.size:
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

    if has_stability and "X" in data and "Y" in data and mode_mat.size == int(np.asarray(data["X"]).size):
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

        # Explicit error diagnostics for mode 7:
        # - absolute and relative field errors (u component)
        # - phase mismatch in energetic regions
        abs_ref = np.abs(u_mat_grid)
        abs_err = np.abs(u_py_grid - u_mat_grid)
        rel_err = abs_err / np.maximum(abs_ref, 1e-12)
        phase_diff = np.angle(np.exp(1j * (np.angle(u_py_grid) - np.angle(u_mat_grid))))
        energetic_mask = abs_ref >= np.percentile(abs_ref, 75.0)
        phase_plot = np.full_like(phase_diff, np.nan, dtype=float)
        phase_plot[energetic_mask] = phase_diff[energetic_mask]

        abs_err_log = np.log10(np.maximum(abs_err, 1e-30))
        rel_err_clip = np.clip(rel_err, 0.0, np.percentile(rel_err, 99.5))
        abs_ref_log = np.log10(np.maximum(abs_ref, 1e-30))
        mode7_rel_l2 = float(np.linalg.norm(u_py_grid - u_mat_grid) / max(np.linalg.norm(u_mat_grid), 1e-30))
        mode7_rel_max = float(np.max(rel_err))
        mode7_rel_p95 = float(np.percentile(rel_err, 95.0))
        phase_abs = np.abs(phase_plot[energetic_mask])
        mode7_phase_p95 = float(np.percentile(phase_abs, 95.0)) if phase_abs.size > 0 else 0.0
        stability_metrics["mode7_rel_l2"] = mode7_rel_l2
        stability_metrics["mode7_rel_max"] = mode7_rel_max
        stability_metrics["mode7_rel_p95"] = mode7_rel_p95
        stability_metrics["mode7_phase_p95_rad"] = mode7_phase_p95
        if spectrum_png is not None:
            spectrum_abs_err = np.abs(py_sel - mat_sel)
            stability_metrics["spectrum_max_abs"] = float(np.max(spectrum_abs_err))
            stability_metrics["spectrum_mean_abs"] = float(np.mean(spectrum_abs_err))

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
        c00 = axes[0, 0].pcolormesh(X, Y, abs_ref_log, shading="auto", cmap="magma")
        axes[0, 0].set_title("mode7 |u_mat| (log10)")
        c01 = axes[0, 1].pcolormesh(X, Y, abs_err_log, shading="auto", cmap="magma")
        axes[0, 1].set_title("mode7 |u_py - u_mat| (log10)")
        c10 = axes[1, 0].pcolormesh(X, Y, rel_err_clip, shading="auto", cmap="viridis")
        axes[1, 0].set_title("mode7 relative error")
        c11 = axes[1, 1].pcolormesh(X, Y, phase_plot, shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1, 1].set_title("mode7 phase error (energetic mask)")
        for ax in axes.ravel():
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.colorbar(c00, ax=axes[0, 0], shrink=0.9)
        fig.colorbar(c01, ax=axes[0, 1], shrink=0.9)
        fig.colorbar(c10, ax=axes[1, 0], shrink=0.9)
        fig.colorbar(c11, ax=axes[1, 1], shrink=0.9)
        fig.tight_layout()
        mode7_error_maps_png = root / "example_6_hamada_mode7_error_maps.png"
        fig.savefig(mode7_error_maps_png, dpi=220)
        plt.close(fig)

        # Pointwise parity scatter for real/imag parts in energetic region.
        idx_scatter = energetic_mask.ravel(order="F")
        mat_vec = u_mat_grid.ravel(order="F")[idx_scatter]
        py_vec = u_py_grid.ravel(order="F")[idx_scatter]
        if mat_vec.size > 0:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
            axes[0].scatter(np.real(mat_vec), np.real(py_vec), s=7, alpha=0.45)
            lo0 = min(float(np.min(np.real(mat_vec))), float(np.min(np.real(py_vec))))
            hi0 = max(float(np.max(np.real(mat_vec))), float(np.max(np.real(py_vec))))
            axes[0].plot([lo0, hi0], [lo0, hi0], "k--", linewidth=0.9)
            axes[0].set_title("mode7 parity scatter: Re(u)")
            axes[0].set_xlabel("MATLAB")
            axes[0].set_ylabel("Python")
            axes[0].grid(alpha=0.25)

            axes[1].scatter(np.imag(mat_vec), np.imag(py_vec), s=7, alpha=0.45)
            lo1 = min(float(np.min(np.imag(mat_vec))), float(np.min(np.imag(py_vec))))
            hi1 = max(float(np.max(np.imag(mat_vec))), float(np.max(np.imag(py_vec))))
            axes[1].plot([lo1, hi1], [lo1, hi1], "k--", linewidth=0.9)
            axes[1].set_title("mode7 parity scatter: Im(u)")
            axes[1].set_xlabel("MATLAB")
            axes[1].set_ylabel("Python")
            axes[1].grid(alpha=0.25)

            fig.tight_layout()
            mode7_scatter_png = root / "example_6_hamada_mode7_parity_scatter.png"
            fig.savefig(mode7_scatter_png, dpi=220)
            plt.close(fig)
            print(f"generated: {mode7_scatter_png}")
        print(f"generated: {mode7_error_maps_png}")

        # Stability dashboard: one compact view for quick report sharing.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        if spectrum_png is not None:
            axes[0, 0].scatter(mat_sel.real, mat_sel.imag, s=40, marker="x", label="MATLAB", alpha=0.8)
            axes[0, 0].scatter(py_sel.real, py_sel.imag, s=20, marker="o", facecolors="none", label="Python", alpha=0.9)
            axes[0, 0].axhline(0.0, color="0.45", linewidth=0.8)
            axes[0, 0].set_xlabel("Re(omega)")
            axes[0, 0].set_ylabel("Im(omega)")
            axes[0, 0].set_title("spectrum parity")
            axes[0, 0].grid(alpha=0.2)
            axes[0, 0].legend()
        else:
            axes[0, 0].axis("off")
            axes[0, 0].text(
                0.03,
                0.95,
                "spectrum parity figure skipped\n(reference/config mismatch mode)",
                va="top",
                ha="left",
                fontsize=10,
                transform=axes[0, 0].transAxes,
            )

        if mat_vec.size > 0:
            axes[0, 1].scatter(np.real(mat_vec), np.real(py_vec), s=5, alpha=0.45, label="Re(u)")
            axes[0, 1].scatter(np.imag(mat_vec), np.imag(py_vec), s=5, alpha=0.35, label="Im(u)")
            lo = min(
                float(np.min(np.real(mat_vec))),
                float(np.min(np.real(py_vec))),
                float(np.min(np.imag(mat_vec))),
                float(np.min(np.imag(py_vec))),
            )
            hi = max(
                float(np.max(np.real(mat_vec))),
                float(np.max(np.real(py_vec))),
                float(np.max(np.imag(mat_vec))),
                float(np.max(np.imag(py_vec))),
            )
            axes[0, 1].plot([lo, hi], [lo, hi], "k--", linewidth=0.9)
            axes[0, 1].set_xlabel("MATLAB")
            axes[0, 1].set_ylabel("Python")
            axes[0, 1].set_title("mode7 parity scatter (energetic)")
            axes[0, 1].grid(alpha=0.2)
            axes[0, 1].legend()
        else:
            axes[0, 1].axis("off")

        c = axes[1, 0].pcolormesh(X, Y, rel_err_clip, shading="auto", cmap="viridis")
        axes[1, 0].set_title("mode7 relative error map")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        fig.colorbar(c, ax=axes[1, 0], shrink=0.9)

        axes[1, 1].plot(X[iy, :], np.real(u_mat_grid[iy, :]), "x-", markersize=4.2, linewidth=0.9, label="MATLAB Re(u)")
        axes[1, 1].plot(X[iy, :], np.real(u_py_grid[iy, :]), "o-", markersize=2.4, linewidth=0.8, label="Python Re(u)")
        axes[1, 1].plot(X[iy, :], np.abs(u_mat_grid[iy, :]), "--", linewidth=1.0, label="MATLAB |u|")
        axes[1, 1].plot(X[iy, :], np.abs(u_py_grid[iy, :]), "-", linewidth=0.9, label="Python |u|")
        axes[1, 1].set_title("mode7 centerline-x cuts")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("value")
        axes[1, 1].grid(alpha=0.2)
        axes[1, 1].legend(fontsize=8)

        fig.suptitle("Hamada stability dashboard", fontsize=13)
        fig.tight_layout()
        stability_dash_png = root / "example_6_hamada_dashboard_stability.png"
        fig.savefig(stability_dash_png, dpi=220)
        plt.close(fig)
        print(f"generated: {stability_dash_png}")
    else:
        cuts_png = None

    if spectrum_png is not None:
        print(f"generated: {spectrum_png}")
    if leading_png is not None:
        print(f"generated: {leading_png}")
    if "X" in data and "Y" in data and "U" in data:
        print(f"generated: {preview_png}")
    if mode_png is not None:
        print(f"generated: {mode_png}")
    if cuts_png is not None:
        print(f"generated: {cuts_png}")
    if modes2d_png is not None:
        print(f"generated: {modes2d_png}")
    if mode7_2d_png is not None:
        print(f"generated: {mode7_2d_png}")

    if "gains" in data and args.matlab_resolvent.exists():
        import h5py

        def _as_complex(values: np.ndarray) -> np.ndarray:
            if values.dtype.fields and "real" in values.dtype.fields and "imag" in values.dtype.fields:
                return values["real"] + 1j * values["imag"]
            return values.astype(complex)

        def _best_complex_scale_res(reference: np.ndarray, candidate: np.ndarray) -> complex:
            denom = np.vdot(candidate, candidate)
            if denom == 0:
                return complex(1.0)
            return np.vdot(candidate, reference) / denom

        with h5py.File(args.matlab_resolvent, "r") as handle:
            gains_mat = np.asarray(handle["S"], dtype=float).reshape(-1)
            U_mat = _as_complex(np.asarray(handle["U"])).T if "U" in handle else None
            V_mat = _as_complex(np.asarray(handle["V"])).T if "V" in handle else None

        gains_py = np.asarray(data["gains"], dtype=float).reshape(-1)
        n_gain = int(min(gains_py.size, gains_mat.size))
        if n_gain > 0:
            rank = np.arange(1, n_gain + 1)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(rank, gains_mat[:n_gain], "x-", markersize=6.0, linewidth=1.0, label="MATLAB")
            ax.semilogy(rank, gains_py[:n_gain], "o-", markersize=3.2, linewidth=0.9, label="Python")
            ax.set_xlabel("mode rank")
            ax.set_ylabel("gain")
            ax.set_title(f"Hamada resolvent gains parity (n={n_gain})")
            ax.grid(alpha=0.25, which="both")
            ax.legend()
            fig.tight_layout()
            gains_png = root / "example_6_hamada_resolvent_gains.png"
            fig.savefig(gains_png, dpi=220)
            plt.close(fig)
            print(f"generated: {gains_png}")

        # Additional visual parity diagnostics for "physical feel":
        # - 2D response/forcing maps (u-component) for mode 1
        # - centerline cuts comparing MATLAB vs Python
        # - Python response mode gallery (first 3 modes)
        if (
            "responses" in data
            and "forcings" in data
            and "idx_u_j" in data
            and "X" in data
            and "Y" in data
            and U_mat is not None
            and V_mat is not None
        ):
            X = np.asarray(data["X"], dtype=float)
            Y = np.asarray(data["Y"], dtype=float)
            idx_u = np.asarray(data["idx_u_j"], dtype=int)

            responses_py = np.asarray(data["responses"], dtype=complex)
            forcings_py = np.asarray(data["forcings"], dtype=complex)
            n_mode = int(min(responses_py.shape[1], forcings_py.shape[1], U_mat.shape[1], V_mat.shape[1]))

            if n_mode > 0:
                # Mode-1 comparison with best complex scaling (amplitude + phase).
                r_ref = U_mat[:, 0]
                f_ref = V_mat[:, 0]
                r_py = responses_py[:, 0] * _best_complex_scale_res(r_ref, responses_py[:, 0])
                f_py = forcings_py[:, 0] * _best_complex_scale_res(f_ref, forcings_py[:, 0])

                r_ref_u = r_ref[idx_u].reshape(X.shape, order="F")
                r_py_u = r_py[idx_u].reshape(X.shape, order="F")
                f_ref_u = f_ref[idx_u].reshape(X.shape, order="F")
                f_py_u = f_py[idx_u].reshape(X.shape, order="F")

                def _robust_vmax(*arrays: np.ndarray) -> float:
                    vals = np.concatenate([np.abs(np.real(a)).ravel() for a in arrays])
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        return 1e-12
                    vmax = float(np.percentile(vals, 98.0))
                    return max(vmax, 1e-12)

                def _overlay_structure(ax: plt.Axes, field: np.ndarray) -> None:
                    mag = np.abs(field)
                    finite = mag[np.isfinite(mag)]
                    if finite.size == 0:
                        return
                    levels = np.unique(np.percentile(finite, [95.0, 99.0]))
                    levels = levels[levels > 0.0]
                    if levels.size == 0:
                        return
                    ax.contour(
                        X,
                        Y,
                        mag,
                        levels=levels,
                        colors="k",
                        linewidths=0.45,
                        alpha=0.35,
                    )

                vmax_r = _robust_vmax(r_ref_u, r_py_u)
                vmax_f = _robust_vmax(f_ref_u, f_py_u)

                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
                c00 = axes[0, 0].pcolormesh(
                    X, Y, np.real(r_ref_u), shading="auto", cmap="RdBu_r", vmin=-vmax_r, vmax=vmax_r
                )
                axes[0, 0].set_title("MATLAB response mode 1: Re(u)")
                _overlay_structure(axes[0, 0], r_ref_u)
                axes[0, 1].pcolormesh(
                    X, Y, np.real(r_py_u), shading="auto", cmap="RdBu_r", vmin=-vmax_r, vmax=vmax_r
                )
                axes[0, 1].set_title("Python response mode 1: Re(u)")
                _overlay_structure(axes[0, 1], r_py_u)
                axes[1, 0].pcolormesh(
                    X, Y, np.real(f_ref_u), shading="auto", cmap="RdBu_r", vmin=-vmax_f, vmax=vmax_f
                )
                axes[1, 0].set_title("MATLAB forcing mode 1: Re(u)")
                _overlay_structure(axes[1, 0], f_ref_u)
                axes[1, 1].pcolormesh(
                    X, Y, np.real(f_py_u), shading="auto", cmap="RdBu_r", vmin=-vmax_f, vmax=vmax_f
                )
                axes[1, 1].set_title("Python forcing mode 1: Re(u)")
                _overlay_structure(axes[1, 1], f_py_u)
                for ax in axes.ravel():
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                fig.colorbar(c00, ax=axes[0, :].tolist(), shrink=0.90, label="Re(u)")
                fig.tight_layout()
                mode1_maps_png = root / "example_6_hamada_resolvent_mode1_maps.png"
                fig.savefig(mode1_maps_png, dpi=220)
                plt.close(fig)
                print(f"generated: {mode1_maps_png}")

                # Mode-1 explicit error maps (response/forcing): abs and relative.
                r_abs_err = np.abs(r_py_u - r_ref_u)
                f_abs_err = np.abs(f_py_u - f_ref_u)
                r_rel_err = r_abs_err / np.maximum(np.abs(r_ref_u), 1e-12)
                f_rel_err = f_abs_err / np.maximum(np.abs(f_ref_u), 1e-12)

                r_abs_log = np.log10(np.maximum(r_abs_err, 1e-30))
                f_abs_log = np.log10(np.maximum(f_abs_err, 1e-30))
                r_rel_clip = np.clip(r_rel_err, 0.0, np.percentile(r_rel_err, 99.5))
                f_rel_clip = np.clip(f_rel_err, 0.0, np.percentile(f_rel_err, 99.5))

                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
                c00 = axes[0, 0].pcolormesh(X, Y, r_abs_log, shading="auto", cmap="magma")
                axes[0, 0].set_title("response mode1 |py - mat| (log10)")
                c01 = axes[0, 1].pcolormesh(X, Y, r_rel_clip, shading="auto", cmap="viridis")
                axes[0, 1].set_title("response mode1 relative error")
                c10 = axes[1, 0].pcolormesh(X, Y, f_abs_log, shading="auto", cmap="magma")
                axes[1, 0].set_title("forcing mode1 |py - mat| (log10)")
                c11 = axes[1, 1].pcolormesh(X, Y, f_rel_clip, shading="auto", cmap="viridis")
                axes[1, 1].set_title("forcing mode1 relative error")
                for ax in axes.ravel():
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                fig.colorbar(c00, ax=axes[0, 0], shrink=0.9)
                fig.colorbar(c01, ax=axes[0, 1], shrink=0.9)
                fig.colorbar(c10, ax=axes[1, 0], shrink=0.9)
                fig.colorbar(c11, ax=axes[1, 1], shrink=0.9)
                fig.tight_layout()
                mode1_error_png = root / "example_6_hamada_resolvent_mode1_error_maps.png"
                fig.savefig(mode1_error_png, dpi=220)
                plt.close(fig)
                print(f"generated: {mode1_error_png}")

                # Enhanced signed-log maps to avoid "empty-looking" plots when
                # structures are thin compared with full-domain extents.
                def _signed_log_real(field: np.ndarray) -> np.ndarray:
                    real_f = np.real(field)
                    finite = np.abs(real_f[np.isfinite(real_f)])
                    if finite.size == 0:
                        scale = 1e-12
                    else:
                        scale = max(float(np.percentile(finite, 90.0)), 1e-12)
                    return np.sign(real_f) * np.log10(1.0 + np.abs(real_f) / scale)

                r_ref_log = _signed_log_real(r_ref_u)
                r_py_log = _signed_log_real(r_py_u)
                f_ref_log = _signed_log_real(f_ref_u)
                f_py_log = _signed_log_real(f_py_u)
                vmax_log = max(
                    float(np.percentile(np.abs(r_ref_log), 99.5)),
                    float(np.percentile(np.abs(r_py_log), 99.5)),
                    float(np.percentile(np.abs(f_ref_log), 99.5)),
                    float(np.percentile(np.abs(f_py_log), 99.5)),
                    1e-12,
                )

                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
                axes[0, 0].pcolormesh(
                    X, Y, r_ref_log, shading="auto", cmap="RdBu_r", vmin=-vmax_log, vmax=vmax_log
                )
                axes[0, 0].set_title("MATLAB response mode 1: signed log Re(u)")
                axes[0, 1].pcolormesh(
                    X, Y, r_py_log, shading="auto", cmap="RdBu_r", vmin=-vmax_log, vmax=vmax_log
                )
                axes[0, 1].set_title("Python response mode 1: signed log Re(u)")
                axes[1, 0].pcolormesh(
                    X, Y, f_ref_log, shading="auto", cmap="RdBu_r", vmin=-vmax_log, vmax=vmax_log
                )
                axes[1, 0].set_title("MATLAB forcing mode 1: signed log Re(u)")
                axes[1, 1].pcolormesh(
                    X, Y, f_py_log, shading="auto", cmap="RdBu_r", vmin=-vmax_log, vmax=vmax_log
                )
                axes[1, 1].set_title("Python forcing mode 1: signed log Re(u)")
                for ax in axes.ravel():
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                fig.tight_layout()
                mode1_enhanced_png = root / "example_6_hamada_resolvent_mode1_maps_enhanced.png"
                fig.savefig(mode1_enhanced_png, dpi=220)
                plt.close(fig)
                print(f"generated: {mode1_enhanced_png}")

                # Log-magnitude + zoom view to make localized structures visible.
                mag_r = np.maximum(np.abs(r_py_u), 1e-30)
                mag_f = np.maximum(np.abs(f_py_u), 1e-30)
                log_r = np.log10(mag_r)
                log_f = np.log10(mag_f)

                # auto zoom box around energetic response region (99th percentile)
                thr = np.percentile(mag_r, 99.0)
                active = mag_r >= thr
                if np.any(active):
                    iy_act, ix_act = np.where(active)
                    iy0 = max(int(iy_act.min()) - 8, 0)
                    iy1 = min(int(iy_act.max()) + 8, X.shape[0] - 1)
                    ix0 = max(int(ix_act.min()) - 8, 0)
                    ix1 = min(int(ix_act.max()) + 8, X.shape[1] - 1)
                else:
                    iy0, iy1, ix0, ix1 = 0, X.shape[0] - 1, 0, X.shape[1] - 1

                # Zoomed linear maps (MATLAB vs Python) for response/forcing mode 1.
                x_zoom_min = float(np.min(X[iy0:iy1 + 1, ix0:ix1 + 1]))
                x_zoom_max = float(np.max(X[iy0:iy1 + 1, ix0:ix1 + 1]))
                y_zoom_min = float(np.min(Y[iy0:iy1 + 1, ix0:ix1 + 1]))
                y_zoom_max = float(np.max(Y[iy0:iy1 + 1, ix0:ix1 + 1]))

                r_ref_zoom = r_ref_u[iy0:iy1 + 1, ix0:ix1 + 1]
                r_py_zoom = r_py_u[iy0:iy1 + 1, ix0:ix1 + 1]
                f_ref_zoom = f_ref_u[iy0:iy1 + 1, ix0:ix1 + 1]
                f_py_zoom = f_py_u[iy0:iy1 + 1, ix0:ix1 + 1]
                vmax_r_zoom = _robust_vmax(r_ref_zoom, r_py_zoom)
                vmax_f_zoom = _robust_vmax(f_ref_zoom, f_py_zoom)

                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
                axes[0, 0].pcolormesh(
                    X, Y, np.real(r_ref_u), shading="auto", cmap="RdBu_r", vmin=-vmax_r_zoom, vmax=vmax_r_zoom
                )
                axes[0, 0].set_title("MATLAB response mode 1: Re(u), zoom")
                axes[0, 1].pcolormesh(
                    X, Y, np.real(r_py_u), shading="auto", cmap="RdBu_r", vmin=-vmax_r_zoom, vmax=vmax_r_zoom
                )
                axes[0, 1].set_title("Python response mode 1: Re(u), zoom")
                axes[1, 0].pcolormesh(
                    X, Y, np.real(f_ref_u), shading="auto", cmap="RdBu_r", vmin=-vmax_f_zoom, vmax=vmax_f_zoom
                )
                axes[1, 0].set_title("MATLAB forcing mode 1: Re(u), zoom")
                axes[1, 1].pcolormesh(
                    X, Y, np.real(f_py_u), shading="auto", cmap="RdBu_r", vmin=-vmax_f_zoom, vmax=vmax_f_zoom
                )
                axes[1, 1].set_title("Python forcing mode 1: Re(u), zoom")
                for ax in axes.ravel():
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_xlim(x_zoom_min, x_zoom_max)
                    ax.set_ylim(y_zoom_min, y_zoom_max)
                fig.tight_layout()
                mode1_zoom_compare_png = root / "example_6_hamada_resolvent_mode1_maps_zoomed.png"
                fig.savefig(mode1_zoom_compare_png, dpi=220)
                plt.close(fig)
                print(f"generated: {mode1_zoom_compare_png}")

                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
                c0 = axes[0].pcolormesh(X, Y, log_r, shading="auto", cmap="magma")
                axes[0].set_title("Python response mode 1: log10|u|")
                axes[0].set_xlabel("x")
                axes[0].set_ylabel("y")
                axes[0].set_xlim(x_zoom_min, x_zoom_max)
                axes[0].set_ylim(y_zoom_min, y_zoom_max)
                fig.colorbar(c0, ax=axes[0], shrink=0.9, label="log10|u|")

                c1 = axes[1].pcolormesh(X, Y, log_f, shading="auto", cmap="magma")
                axes[1].set_title("Python forcing mode 1: log10|u|")
                axes[1].set_xlabel("x")
                axes[1].set_ylabel("y")
                axes[1].set_xlim(x_zoom_min, x_zoom_max)
                axes[1].set_ylim(y_zoom_min, y_zoom_max)
                fig.colorbar(c1, ax=axes[1], shrink=0.9, label="log10|u|")
                fig.tight_layout()
                mode1_zoom_png = root / "example_6_hamada_resolvent_mode1_zoom_logmag.png"
                fig.savefig(mode1_zoom_png, dpi=220)
                plt.close(fig)
                print(f"generated: {mode1_zoom_png}")

                # Centerline cuts for mode 1 (response/forcing).
                iy = X.shape[0] // 2
                ix = X.shape[1] // 2
                fig, axes = plt.subplots(2, 2, figsize=(11, 7))
                axes[0, 0].plot(X[iy, :], np.real(r_ref_u[iy, :]), "x-", linewidth=0.9, markersize=5.0, label="MATLAB")
                axes[0, 0].plot(X[iy, :], np.real(r_py_u[iy, :]), "o-", linewidth=0.8, markersize=2.8, label="Python")
                axes[0, 0].set_title("response mode 1, centerline-x: Re(u)")
                axes[0, 0].set_xlabel("x")
                axes[0, 0].set_ylabel("Re(u)")
                axes[0, 0].grid(alpha=0.25)
                axes[0, 0].legend()

                axes[0, 1].plot(Y[:, ix], np.real(r_ref_u[:, ix]), "x-", linewidth=0.9, markersize=5.0, label="MATLAB")
                axes[0, 1].plot(Y[:, ix], np.real(r_py_u[:, ix]), "o-", linewidth=0.8, markersize=2.8, label="Python")
                axes[0, 1].set_title("response mode 1, centerline-y: Re(u)")
                axes[0, 1].set_xlabel("y")
                axes[0, 1].set_ylabel("Re(u)")
                axes[0, 1].grid(alpha=0.25)
                axes[0, 1].legend()

                axes[1, 0].plot(X[iy, :], np.abs(f_ref_u[iy, :]), "x-", linewidth=0.9, markersize=5.0, label="MATLAB")
                axes[1, 0].plot(X[iy, :], np.abs(f_py_u[iy, :]), "o-", linewidth=0.8, markersize=2.8, label="Python")
                axes[1, 0].set_title("forcing mode 1, centerline-x: |u|")
                axes[1, 0].set_xlabel("x")
                axes[1, 0].set_ylabel("|u|")
                axes[1, 0].grid(alpha=0.25)
                axes[1, 0].legend()

                axes[1, 1].plot(Y[:, ix], np.abs(f_ref_u[:, ix]), "x-", linewidth=0.9, markersize=5.0, label="MATLAB")
                axes[1, 1].plot(Y[:, ix], np.abs(f_py_u[:, ix]), "o-", linewidth=0.8, markersize=2.8, label="Python")
                axes[1, 1].set_title("forcing mode 1, centerline-y: |u|")
                axes[1, 1].set_xlabel("y")
                axes[1, 1].set_ylabel("|u|")
                axes[1, 1].grid(alpha=0.25)
                axes[1, 1].legend()
                fig.tight_layout()
                mode1_cuts_png = root / "example_6_hamada_resolvent_mode1_cuts.png"
                fig.savefig(mode1_cuts_png, dpi=220)
                plt.close(fig)
                print(f"generated: {mode1_cuts_png}")

                # Gallery of Python response modes (first 3), useful to assess
                # smoothness/structure and physical coherence.
                n_gallery = int(min(3, n_mode))
                fig, axes = plt.subplots(1, n_gallery, figsize=(5.2 * n_gallery, 4.2), sharex=True, sharey=True)
                if n_gallery == 1:
                    axes = np.array([axes])
                for j, ax in enumerate(axes):
                    mode_u = responses_py[:, j][idx_u].reshape(X.shape, order="F")
                    vmax = max(float(np.percentile(np.abs(np.real(mode_u)), 99.5)), 1e-12)
                    c = ax.pcolormesh(
                        X,
                        Y,
                        np.real(mode_u),
                        shading="auto",
                        cmap="RdBu_r",
                        vmin=-vmax,
                        vmax=vmax,
                    )
                    ax.set_title(f"Python response mode {j + 1}: Re(u)")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    _overlay_structure(ax, mode_u)
                    fig.colorbar(c, ax=ax, shrink=0.90)
                fig.tight_layout()
                gallery_png = root / "example_6_hamada_resolvent_response_gallery.png"
                fig.savefig(gallery_png, dpi=220)
                plt.close(fig)
                print(f"generated: {gallery_png}")

                # Mode-wise parity error summary (gains + forcing/response L2 errors).
                gain_rel = np.abs(gains_py[:n_mode] - gains_mat[:n_mode]) / np.maximum(np.abs(gains_mat[:n_mode]), 1e-30)
                forcing_rel = np.zeros(n_mode, dtype=float)
                response_rel = np.zeros(n_mode, dtype=float)
                for j in range(n_mode):
                    f_ref_j = V_mat[:, j]
                    r_ref_j = U_mat[:, j]
                    f_py_j = forcings_py[:, j] * _best_complex_scale_res(f_ref_j, forcings_py[:, j])
                    r_py_j = responses_py[:, j] * _best_complex_scale_res(r_ref_j, responses_py[:, j])
                    forcing_rel[j] = np.linalg.norm(f_py_j - f_ref_j) / max(np.linalg.norm(f_ref_j), 1e-30)
                    response_rel[j] = np.linalg.norm(r_py_j - r_ref_j) / max(np.linalg.norm(r_ref_j), 1e-30)

                rank = np.arange(1, n_mode + 1)
                fig, ax = plt.subplots(figsize=(8.5, 5.0))
                ax.semilogy(rank, gain_rel, "o-", markersize=3.5, linewidth=0.9, label="gain rel error")
                ax.semilogy(rank, forcing_rel, "s-", markersize=3.2, linewidth=0.9, label="forcing rel L2")
                ax.semilogy(rank, response_rel, "^-", markersize=3.2, linewidth=0.9, label="response rel L2")
                ax.set_xlabel("mode rank")
                ax.set_ylabel("relative error")
                ax.set_title("resolvent parity errors by mode")
                ax.grid(alpha=0.25, which="both")
                ax.legend()
                fig.tight_layout()
                parity_err_png = root / "example_6_hamada_resolvent_parity_errors.png"
                fig.savefig(parity_err_png, dpi=220)
                plt.close(fig)
                print(f"generated: {parity_err_png}")

                resolvent_metrics["gain_rel_max"] = float(np.max(gain_rel))
                resolvent_metrics["gain_rel_mean"] = float(np.mean(gain_rel))
                resolvent_metrics["forcing_rel_max"] = float(np.max(forcing_rel))
                resolvent_metrics["response_rel_max"] = float(np.max(response_rel))

                # Resolvent dashboard: compact summary for reporting.
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes[0, 0].semilogy(rank, gains_mat[:n_mode], "x-", markersize=5.2, linewidth=0.9, label="MATLAB")
                axes[0, 0].semilogy(rank, gains_py[:n_mode], "o-", markersize=3.2, linewidth=0.85, label="Python")
                axes[0, 0].set_title("gains parity")
                axes[0, 0].set_xlabel("mode rank")
                axes[0, 0].set_ylabel("gain")
                axes[0, 0].grid(alpha=0.2, which="both")
                axes[0, 0].legend()

                axes[0, 1].semilogy(rank, gain_rel, "o-", markersize=3.2, linewidth=0.9, label="gain rel")
                axes[0, 1].semilogy(rank, forcing_rel, "s-", markersize=3.0, linewidth=0.9, label="forcing rel L2")
                axes[0, 1].semilogy(rank, response_rel, "^-", markersize=3.0, linewidth=0.9, label="response rel L2")
                axes[0, 1].set_title("mode-wise parity errors")
                axes[0, 1].set_xlabel("mode rank")
                axes[0, 1].set_ylabel("relative error")
                axes[0, 1].grid(alpha=0.2, which="both")
                axes[0, 1].legend()

                vmax_rsp = max(float(np.percentile(np.abs(np.real(r_py_zoom)), 99.5)), 1e-12)
                vmax_frc = max(float(np.percentile(np.abs(np.real(f_py_zoom)), 99.5)), 1e-12)
                c0 = axes[1, 0].pcolormesh(X, Y, np.real(r_py_u), shading="auto", cmap="RdBu_r", vmin=-vmax_rsp, vmax=vmax_rsp)
                axes[1, 0].set_xlim(x_zoom_min, x_zoom_max)
                axes[1, 0].set_ylim(y_zoom_min, y_zoom_max)
                axes[1, 0].set_title("Python response mode1 (zoom)")
                axes[1, 0].set_xlabel("x")
                axes[1, 0].set_ylabel("y")
                fig.colorbar(c0, ax=axes[1, 0], shrink=0.9)

                c1 = axes[1, 1].pcolormesh(X, Y, np.real(f_py_u), shading="auto", cmap="RdBu_r", vmin=-vmax_frc, vmax=vmax_frc)
                axes[1, 1].set_xlim(x_zoom_min, x_zoom_max)
                axes[1, 1].set_ylim(y_zoom_min, y_zoom_max)
                axes[1, 1].set_title("Python forcing mode1 (zoom)")
                axes[1, 1].set_xlabel("x")
                axes[1, 1].set_ylabel("y")
                fig.colorbar(c1, ax=axes[1, 1], shrink=0.9)

                fig.suptitle("Hamada resolvent dashboard", fontsize=13)
                fig.tight_layout()
                resolvent_dash_png = root / "example_6_hamada_dashboard_resolvent.png"
                fig.savefig(resolvent_dash_png, dpi=220)
                plt.close(fig)
                print(f"generated: {resolvent_dash_png}")

    # Global overview dashboard (works for stability-only, resolvent-only, or both).
    if stability_metrics or resolvent_metrics:
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for key, val in stability_metrics.items():
            labels.append(f"stability: {key}")
            values.append(float(val))
            colors.append("#1f77b4")
        for key, val in resolvent_metrics.items():
            labels.append(f"resolvent: {key}")
            values.append(float(val))
            colors.append("#d62728")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1.7]})
        axes[0].axis("off")
        lines = [
            f"NPZ: {args.python_npz}",
            f"stability modes: {omega_py.size}",
            f"resolvent gains: {int(np.asarray(data.get('gains', [])).size)}",
            f"use_matlab_sponge_aux: {use_matlab_sponge_aux}",
            f"matlab spectrum ref: {args.matlab_spectrum}",
            f"matlab resolvent ref: {args.matlab_resolvent}",
        ]
        axes[0].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, transform=axes[0].transAxes)

        y = np.arange(len(values))
        axes[1].barh(y, values, color=colors, alpha=0.85)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=9)
        axes[1].set_xlabel("metric value")
        axes[1].grid(axis="x", alpha=0.25)
        if values and min(v for v in values if v > 0.0) > 0.0:
            vmin = min(v for v in values if v > 0.0)
            vmax = max(values)
            if vmax / vmin > 1e3:
                axes[1].set_xscale("log")
                axes[1].set_xlabel("metric value (log scale)")

        fig.suptitle("Hamada parity overview dashboard", fontsize=13)
        fig.tight_layout()
        overview_png = root / "example_6_hamada_dashboard_overview.png"
        fig.savefig(overview_png, dpi=220)
        plt.close(fig)
        print(f"generated: {overview_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
