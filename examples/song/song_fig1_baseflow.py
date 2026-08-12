#!/usr/bin/env python3
"""Fig S1: Song 32 deg base-flow primitives from song_32deg.npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.figure_style import apply, save, signed_cmap, thin_colorbar

HERE = Path(__file__).resolve().parent
SRC = HERE / "song_32deg.npz"
DST = HERE / "song_fig1_baseflow.png"


def main() -> int:
    apply()
    data = np.load(SRC)
    x = np.asarray(data["x"]).T
    y = np.asarray(data["y"]).T
    fields = [
        (r"$\bar{\rho}$", np.asarray(data["rho"]).T, False),
        (r"$\bar{u}$", np.asarray(data["u"]).T, False),
        (r"$\bar{v}$", np.asarray(data["v"]).T, True),
        (r"$\bar{T}$", np.asarray(data["T"]).T, False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.28, hspace=0.18, left=0.09, right=0.98, top=0.97, bottom=0.12)
    for ax, (label, field, signed) in zip(axes.ravel(), fields, strict=True):
        if signed:
            lim = float(np.max(np.abs(field)))
            mesh = ax.pcolormesh(
                x, y, field, shading="auto", cmap=signed_cmap(), vmin=-lim, vmax=lim, rasterized=True
            )
        else:
            mesh = ax.pcolormesh(x, y, field, shading="auto", cmap="cividis", rasterized=True)
        ax.contour(x, y, np.asarray(data["u"]).T, levels=[0.0], colors="0.15", linewidths=0.4)
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.set_ylim(float(np.min(y)), float(np.max(y)))
        ax.tick_params(top=True, right=True)
        thin_colorbar(fig, mesh, ax, label)
    axes[1, 0].set_xlabel(r"$x/X_{\mathrm{sh}}$")
    axes[1, 1].set_xlabel(r"$x/X_{\mathrm{sh}}$")
    axes[0, 0].set_ylabel(r"$y/X_{\mathrm{sh}}$")
    axes[1, 0].set_ylabel(r"$y/X_{\mathrm{sh}}$")
    save(fig, DST)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
