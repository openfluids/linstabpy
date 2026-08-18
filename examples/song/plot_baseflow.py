#!/usr/bin/env python3
"""Plot Song (2023) nodal mesh and mean-flow fields (θ = 32°).

Reads ``inputs/baseflow/*.npy`` (from ``prepare_baseflow.py``) with the same
``load_case_fields`` path used by the EVP driver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import ImageGrid

from linstabpy.baseflow import load_case_fields

from examples.figure_style import apply, check_plot_environment, save
from examples.song.reference_paths import FIGURES_BASEFLOW_DIR, INPUTS_BASEFLOW_DIR

BASEFLOW_LABEL_SIZE = 16
BASEFLOW_TICK_SIZE = 14
BASEFLOW_CBAR_LABEL_SIZE = 16
BASEFLOW_CBAR_TICK_SIZE = 14
BASEFLOW_FIGSIZE = (11.0, 4.9)
BASEFLOW_AXES_PAD_H = 1.0
BASEFLOW_AXES_PAD_V = 0.2
BASEFLOW_CBAR_PAD = 0.06
BASEFLOW_CBAR_SIZE = "2.4%"
BASEFLOW_CBAR_MAX_TICKS = 5
BASEFLOW_CBAR_LABELPAD = 8
BASEFLOW_CBAR_LABELPAD_T = 10
BASEFLOW_XLABEL_PAD = 6
BASEFLOW_MARGIN_LEFT = 0.06
BASEFLOW_MARGIN_RIGHT = 0.98
BASEFLOW_MARGIN_BOTTOM = 0.16
BASEFLOW_MARGIN_TOP = 0.97
BASEFLOW_SAVE_PAD_INCHES = 0.05
N_CONTOUR_LEVELS = 48
FIELD_CLIMS: dict[str, tuple[float, float]] = {
    "rho": (0.6, 1.4),
    "u": (0.0, 1.0),
    "v": (-0.08, 0.08),
    "T": (1.0, 1.8),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseflow-dir", type=Path, default=INPUTS_BASEFLOW_DIR)
    p.add_argument("--fig-dir", type=Path, default=FIGURES_BASEFLOW_DIR)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--mesh-stride", type=int, default=4)
    return p.parse_args()


def _configure_style() -> None:
    """Match validation figures: LaTeX when available, with baseflow font sizes."""
    apply()
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": BASEFLOW_LABEL_SIZE,
            "axes.labelsize": BASEFLOW_LABEL_SIZE,
            "axes.titlesize": BASEFLOW_LABEL_SIZE,
            "xtick.labelsize": BASEFLOW_TICK_SIZE,
            "ytick.labelsize": BASEFLOW_TICK_SIZE,
        }
    )


def _load_case(bf_dir: Path):
    return load_case_fields(
        bf_dir,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        transpose_input=True,
        preprocess={"enabled": False},
    )


def plot_mesh(*, X: np.ndarray, Y: np.ndarray, out_path: Path, dpi: int, stride: int) -> None:
    _configure_style()
    fig, (ax_grid, ax_dy) = plt.subplots(1, 2, figsize=(12.5, 4.6), width_ratios=[2.0, 1.0])
    xs = X[::stride, ::stride]
    ys = Y[::stride, ::stride]
    ax_grid.plot(xs, ys, color="0.25", linewidth=0.35)
    ax_grid.plot(xs.T, ys.T, color="0.25", linewidth=0.35)
    ax_grid.set_xlabel(r"$x/X_{\mathrm{sh}}$")
    ax_grid.set_ylabel(r"$y/X_{\mathrm{sh}}$")
    ax_grid.set_title(rf"Nodal mesh (every {stride}th point shown)")
    ax_grid.set_aspect("equal", adjustable="box")
    j_mid = X.shape[1] // 2
    dy = np.diff(Y[:, j_mid])
    ax_dy.semilogy(dy, np.arange(dy.size), marker=".", markersize=3.5, linewidth=1.0, color="C0")
    ax_dy.set_xlabel(r"$\Delta y$")
    ax_dy.set_ylabel(r"wall-normal index $j$")
    ax_dy.set_title("Wall-normal spacing")
    fig.tight_layout()
    save(fig, out_path, dpi=dpi, pad_inches=0.02)


def plot_baseflow_fields(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    T: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    _configure_style()
    box_aspect = float(Y.max() - Y.min()) / float(X.max() - X.min())
    fig = plt.figure(figsize=BASEFLOW_FIGSIZE)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 2),
        axes_pad=(BASEFLOW_AXES_PAD_H, BASEFLOW_AXES_PAD_V),
        cbar_mode="each",
        cbar_location="right",
        cbar_pad=BASEFLOW_CBAR_PAD,
        cbar_size=BASEFLOW_CBAR_SIZE,
        share_all=False,
    )
    panels = (
        (rho, r"$\bar{\rho}$", FIELD_CLIMS["rho"], BASEFLOW_CBAR_LABELPAD),
        (u, r"$\bar{u}$", FIELD_CLIMS["u"], BASEFLOW_CBAR_LABELPAD),
        (v, r"$\bar{v}$", FIELD_CLIMS["v"], BASEFLOW_CBAR_LABELPAD),
        (T, r"$\overline{T}$", FIELD_CLIMS["T"], BASEFLOW_CBAR_LABELPAD_T),
    )
    for ax, (field, label, (vmin, vmax), labelpad) in zip(grid, panels):
        levels = np.linspace(vmin, vmax, N_CONTOUR_LEVELS + 1)
        cf = ax.contourf(
            X, Y, field, levels=levels, cmap="turbo", vmin=vmin, vmax=vmax, extend="both"
        )
        ax.contour(X, Y, u, levels=[0.0], colors=["white"], linewidths=[0.8], alpha=0.85)
        ax.set_xlim(float(X.min()), float(X.max()))
        ax.set_ylim(float(Y.min()), float(Y.max()))
        ax.set_box_aspect(box_aspect)
        cbar = ax.cax.colorbar(cf, extend="both")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=BASEFLOW_CBAR_MAX_TICKS))
        cbar.set_label(label, rotation=90, va="center", labelpad=labelpad, fontsize=BASEFLOW_CBAR_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=BASEFLOW_CBAR_TICK_SIZE, length=2.5, width=0.6)
        ax.tick_params(axis="both", labelsize=BASEFLOW_TICK_SIZE)
    for ax in grid.axes_column[0]:
        ax.set_ylabel(r"$y/X_{\mathrm{sh}}$", fontsize=BASEFLOW_LABEL_SIZE)
    for ax in grid.axes_row[1]:
        ax.set_xlabel(r"$x/X_{\mathrm{sh}}$", labelpad=BASEFLOW_XLABEL_PAD, fontsize=BASEFLOW_LABEL_SIZE)
    fig.subplots_adjust(
        left=BASEFLOW_MARGIN_LEFT,
        right=BASEFLOW_MARGIN_RIGHT,
        bottom=BASEFLOW_MARGIN_BOTTOM,
        top=BASEFLOW_MARGIN_TOP,
    )
    save(fig, out_path, dpi=dpi, pad_inches=BASEFLOW_SAVE_PAD_INCHES)


def main() -> None:
    check_plot_environment(require_latex=True)
    args = _parse_args()
    bf_dir = args.baseflow_dir.expanduser().resolve()
    fig_dir = args.fig_dir.expanduser().resolve()
    X, Y, baseflow, _sponge, _info = _load_case(bf_dir)
    out_mesh = fig_dir / "song2023_mesh_theta32.png"
    out_fields = fig_dir / "song2023_baseflow_fields_theta32.png"
    plot_mesh(X=X, Y=Y, out_path=out_mesh, dpi=args.dpi, stride=args.mesh_stride)
    plot_baseflow_fields(
        X=X,
        Y=Y,
        rho=baseflow.RHO,
        u=baseflow.U,
        v=baseflow.V,
        T=baseflow.T,
        out_path=out_fields,
        dpi=args.dpi,
    )
    print(f"baseflow dir: {bf_dir}")
    print(f"Wrote {out_mesh}")
    print(f"Wrote {out_fields}")


if __name__ == "__main__":
    main()
