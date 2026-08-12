#!/usr/bin/env python3
"""Fig S3: Song leading mode and, when present, the packed linstabpy lead."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from examples.figure_style import apply, save, signed_cmap, y_limit_for_support

HERE = Path(__file__).resolve().parent
SRC_SONG = HERE / "song_32deg_leading_mode.npz"
SRC_OURS = HERE / "song_32deg_modes.npz"
DST = HERE / "song_fig3_eigenmode.png"
VARS = ("rho", "u", "v", "w", "t")
LABELS = {
    "rho": r"$\mathrm{Re}(\rho')$",
    "u": r"$\mathrm{Re}(u')$",
    "v": r"$\mathrm{Re}(v')$",
    "w": r"$\mathrm{Re}(w')$",
    "t": r"$\mathrm{Re}(T')$",
}
X_SH_MM = 80.0


def _phase_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    denom = np.vdot(candidate, candidate)
    if denom == 0:
        return candidate
    scale = np.vdot(candidate, reference) / denom
    return candidate * scale


def _interp_run(run: np.lib.npyio.NpzFile, name: str, x_dst, y_dst) -> np.ndarray:
    """Map one packed linstabpy field onto the Song plot grid."""
    xs = np.asarray(run["x"])[0, :]
    ys = np.asarray(run["y"])[:, 0]
    grid = np.real(np.asarray(run[name]))
    interp = RegularGridInterpolator((ys, xs), grid, bounds_error=False, fill_value=0.0)
    pts = np.column_stack([np.asarray(y_dst).ravel(), np.asarray(x_dst).ravel()])
    return interp(pts).reshape(np.asarray(x_dst).shape)


def _to_xsh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map millimetre Song arrays onto units of X_sh = 80 mm."""
    if float(np.max(x)) > 10.0:
        return x / X_SH_MM, y / X_SH_MM
    return x, y


def main() -> int:
    apply()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=Path, default=SRC_OURS)
    parser.add_argument("-o", "--output", type=Path, default=DST)
    args = parser.parse_args()

    song = np.load(SRC_SONG)
    x_s, y_s = _to_xsh(np.asarray(song["x"]).T, np.asarray(song["y"]).T)
    run = np.load(args.modes) if args.modes.exists() else None
    ncols = 2 if run is not None else 1
    width = 3.15 if ncols == 1 else 5.8
    fig, axes = plt.subplots(
        len(VARS),
        ncols,
        figsize=(width, 6.4),
        sharex=True,
        sharey=True,
    )
    if ncols == 1:
        axes = np.asarray(axes).reshape(len(VARS), 1)
    fig.subplots_adjust(wspace=0.08, hspace=0.12, left=0.12, right=0.96, top=0.97, bottom=0.06)

    song_fields = []
    for name in VARS:
        song_f = np.real(np.asarray(song[name])).T
        peak = np.max(np.abs(song_f))
        if peak > 0.0:
            song_f = song_f / peak
        song_fields.append(song_f)
    ymax = min(y_limit_for_support(y_s, *song_fields, frac=0.99), 0.45)

    for i, name in enumerate(VARS):
        song_f = song_fields[i]
        ax = axes[i, 0]
        mesh = ax.pcolormesh(
            x_s, y_s, song_f, shading="auto", cmap=signed_cmap(), vmin=-1.0, vmax=1.0, rasterized=True
        )
        ax.set_ylim(0.0, ymax)
        ax.set_ylabel(r"$y/X_{\mathrm{sh}}$")
        ax.text(
            0.03,
            0.88,
            LABELS[name],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        if i == 0:
            ax.set_title("Song")
        if run is not None:
            ours = _interp_run(run, name, x_s, y_s)
            ours = np.real(_phase_align(song_f.astype(complex), ours.astype(complex)))
            peak_o = np.max(np.abs(ours))
            if peak_o > 0.0:
                ours = ours / peak_o
            ax2 = axes[i, 1]
            ax2.pcolormesh(
                x_s,
                y_s,
                ours,
                shading="auto",
                cmap=signed_cmap(),
                vmin=-1.0,
                vmax=1.0,
                rasterized=True,
            )
            ax2.set_ylim(0.0, ymax)
            if i == 0:
                ax2.set_title("linstabpy")
    axes[-1, 0].set_xlabel(r"$x/X_{\mathrm{sh}}$")
    if ncols == 2:
        axes[-1, 1].set_xlabel(r"$x/X_{\mathrm{sh}}$")
        for ax in axes[:, 1]:
            ax.tick_params(labelleft=False)
    for ax in np.asarray(axes).ravel():
        ax.set_ylim(0.0, ymax)
    fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02)
    save(fig, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
