#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Figures S4/S5 — Savarino vs linstabpy leading response (Song eigenmode layout).

Same panel template as the Savarino2026 case plotter: shared -1..1 colourbar,
``\\Re`` labels, gouraud shading, matched panel aspect.
Domain: full streamwise mesh with ``x`` from 0.2 to the outflow.
Wall-normal crop ``y/X_sh in [0, 0.1]`` (lift-up) and ``[0, 0.06]`` (Mack).

Requires shipped Savarino response primitives in ``reference/`` and local
linstabpy outputs from ``run_compare_points.py`` under ``output/compare/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from reference_paths import (  # noqa: E402
    OUTPUT_COMPARE_DIR,
    OUTPUT_FIGURES_DIR,
    SAVARINO_RESPONSE_LIFTUP_NPZ,
    SAVARINO_RESPONSE_MACK_NPZ,
)
from savarino_mode_to_primitive import PRIM  # noqa: E402

from examples.figure_style import apply, save, signed_cmap  # noqa: E402

FIELD_LABELS = {
    "rho": r"$\Re(\rho')$",
    "u": r"$\Re(u')$",
    "v": r"$\Re(v')$",
    "w": r"$\Re(w')$",
    "T": r"$\Re(T')$",
}
X_MIN = 0.2
_MACK_YMAX = 0.06
_MACK_ASPECT_SCALE = 9.0
_PANEL_YMAX_ASPECT = _MACK_YMAX * _MACK_ASPECT_SCALE
_LAYOUT = {"col_gap_in": 0.06, "row_gap_in": 0.32}

JOBS = (
    {
        "tag": "liftup",
        "ymax": 0.1,
        "yticks": (0.0, 0.05, 0.1),
        "yticklabels": ["0.0", "0.05", "0.1"],
        "aspect_scale": _PANEL_YMAX_ASPECT / 0.1,
        "response_npz": SAVARINO_RESPONSE_LIFTUP_NPZ,
        "output_name": "liftup",
        "output_stem": "sav_map_b163p0_f0p0001",
    },
    {
        "tag": "mack",
        "ymax": _MACK_YMAX,
        "yticks": (0.0, 0.03, 0.06),
        "yticklabels": ["0.00", "0.03", "0.06"],
        "aspect_scale": _MACK_ASPECT_SCALE,
        "response_npz": SAVARINO_RESPONSE_MACK_NPZ,
        "output_name": "mack",
        "output_stem": "sav_map_b45p0_f2",
    },
)


def _require_output_npz(outputs_dir: Path, name: str, stem: str) -> Path:
    path = outputs_dir / name / f"{stem}.npz"
    if not path.is_file():
        raise SystemExit(
            f"missing {path}\nRun the resolvent points first:  python run_compare_points.py"
        )
    return path


def load_savarino_response(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nIt is built from the Zenodo record, which is not shipped:\n"
            "  python fetch_zenodo.py baseflow\n"
            "  python fetch_zenodo.py modes --f 2 --beta 45\n"
            "  python fetch_zenodo.py modes --f 0.0001 --beta 163\n"
            "  python pack_reference_modes.py"
        )
    data = np.load(path)
    return {v: data[v] for v in PRIM}


def normalized_real(field: np.ndarray) -> np.ndarray:
    r = np.real(np.asarray(field, dtype=complex))
    peak = float(np.nanmax(np.abs(r)))
    if peak <= 0.0:
        return np.zeros_like(r)
    return r / peak


def _align_chu(
    q_ref: np.ndarray, q_py: np.ndarray, w_diag: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    q_ref = q_ref / np.sqrt(np.real(np.vdot(q_ref, w_diag * q_ref)))
    q_py = q_py / np.sqrt(np.real(np.vdot(q_py, w_diag * q_py)))
    c_plain = np.vdot(q_ref, w_diag * q_py)
    c_conj = np.vdot(q_ref, w_diag * np.conj(q_py))
    conjugated = abs(c_conj) > abs(c_plain)
    if conjugated:
        q_py = np.conj(q_py)
        c = c_conj
    else:
        c = c_plain
    q_py = q_py * np.conj(c) / abs(c)
    return q_py, float(abs(c)), conjugated


def _plot_pair(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    prim_ref: dict[str, np.ndarray],
    prim_py: dict[str, np.ndarray],
    out_path: Path,
    ymax: float,
    yticks: tuple[float, ...],
    yticklabels: list[str],
    col_gap_in: float,
    row_gap_in: float,
    aspect_scale: float,
) -> None:
    apply()
    x1 = np.asarray(X[0, :], dtype=float)
    y1 = np.asarray(Y[:, 0], dtype=float)
    xmin = X_MIN
    xmax = float(x1.max())
    xspan = max(xmax - xmin, 1e-12)
    aspect = (ymax / xspan) * aspect_scale
    n_rows = len(PRIM)
    fig_w = 6.6
    left_in = 0.62
    right_in = 0.10
    cbar_w_in = 0.12
    cbar_pad_in = 0.08
    bottom_in = 0.42
    top_in = 0.30
    usable_w = fig_w - left_in - right_in - cbar_w_in - cbar_pad_in - col_gap_in
    pw = usable_w / 2.0
    ph = pw * aspect
    fig_h = bottom_in + top_in + n_rows * ph + (n_rows - 1) * row_gap_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows,
        2,
        figure=fig,
        left=left_in / fig_w,
        right=(left_in + 2.0 * pw + col_gap_in) / fig_w,
        bottom=bottom_in / fig_h,
        top=1.0 - top_in / fig_h,
        wspace=col_gap_in / pw,
        hspace=row_gap_in / ph,
    )
    axes = np.empty((n_rows, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1], sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(1, n_rows):
        axes[i, 0] = fig.add_subplot(gs[i, 0], sharex=axes[0, 0], sharey=axes[0, 0])
        axes[i, 1] = fig.add_subplot(gs[i, 1], sharex=axes[0, 0], sharey=axes[0, 0])
    mesh = None
    for i, key in enumerate(PRIM):
        ref_f = normalized_real(prim_ref[key])
        py_f = normalized_real(prim_py[key])
        ax_s, ax_p = axes[i, 0], axes[i, 1]
        mesh = ax_s.pcolormesh(
            x1,
            y1,
            ref_f,
            shading="gouraud",
            cmap=signed_cmap(),
            vmin=-1.0,
            vmax=1.0,
            rasterized=True,
        )
        ax_p.pcolormesh(
            x1,
            y1,
            py_f,
            shading="gouraud",
            cmap=signed_cmap(),
            vmin=-1.0,
            vmax=1.0,
            rasterized=True,
        )
        ax_s.set_ylabel(r"$y/X_{\mathrm{sh}}$")
        ax_s.text(
            0.02,
            0.82,
            FIELD_LABELS[key],
            transform=ax_s.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        if i == 0:
            ax_s.set_title("Savarino")
            ax_p.set_title("linstabpy")
    axes[-1, 0].set_xlabel(r"$x/X_{\mathrm{sh}}$")
    axes[-1, 1].set_xlabel(r"$x/X_{\mathrm{sh}}$")
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    cax = fig.add_axes(
        [
            (left_in + 2.0 * pw + col_gap_in + cbar_pad_in) / fig_w,
            bottom_in / fig_h,
            cbar_w_in / fig_w,
            (n_rows * ph + (n_rows - 1) * row_gap_in) / fig_h,
        ]
    )
    fig.colorbar(mesh, cax=cax)
    for ax in np.asarray(axes).ravel():
        ax.set_ylim(0.0, ymax)
        ax.set_yticks(yticks)
        ax.minorticks_off()
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(xmin, xmax)
    for ax in axes[:, 0]:
        ax.set_yticklabels(yticklabels)
    save(fig, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUT_COMPARE_DIR,
        help="Directory with local linstabpy .npz outputs (default: output/compare/)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_FIGURES_DIR,
        help="Output directory for the PNGs",
    )
    args = ap.parse_args()
    outputs_dir = args.outputs_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for job in JOBS:
        output_npz = _require_output_npz(outputs_dir, job["output_name"], job["output_stem"])
        hp = load_savarino_response(job["response_npz"])
        with np.load(output_npz, allow_pickle=False) as data:
            X = np.asarray(data["X"], dtype=float)
            Y = np.asarray(data["Y"], dtype=float)
            w_diag = np.asarray(data["W_diag"], dtype=float)
            q_py = np.asarray(data["responses"][:, 0], dtype=complex)
            idx = {v: np.asarray(data[f"idx_{v}_j"]) for v in PRIM}
            gain = float(np.asarray(data["gains"]).reshape(-1)[0])

        q_ref = np.zeros_like(q_py)
        for v in PRIM:
            q_ref[idx[v]] = hp[v].T.ravel(order="F")
        q_py, align, conjugated = _align_chu(q_ref, q_py, w_diag)
        prim_ref = {}
        prim_py = {}
        for v in PRIM:
            prim_ref[v] = q_ref[idx[v]].reshape(X.shape, order="F")
            prim_py[v] = q_py[idx[v]].reshape(X.shape, order="F")

        out_path = out_dir / f"compare_mode_{job['tag']}.png"
        _plot_pair(
            X=X,
            Y=Y,
            prim_ref=prim_ref,
            prim_py=prim_py,
            out_path=out_path,
            ymax=float(job["ymax"]),
            yticks=job["yticks"],
            yticklabels=job["yticklabels"],
            col_gap_in=_LAYOUT["col_gap_in"],
            row_gap_in=_LAYOUT["row_gap_in"],
            aspect_scale=float(job["aspect_scale"]),
        )
        print(
            f"{job['tag']}: sigma={gain:.6f}  Chu |<q_ref,q_py>|={align:.4f}  "
            f"conjugated={conjugated}  ymax={job['ymax']}"
        )
        print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
