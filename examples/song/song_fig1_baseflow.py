#!/usr/bin/env python3
"""Fig S1: Song 32 deg base-flow primitives from song_32deg.npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "song_32deg.npz"
DST = HERE / "song_fig1_baseflow.png"


def main() -> int:
    data = np.load(SRC)
    x = np.asarray(data["x"]).T
    y = np.asarray(data["y"]).T
    fields = [
        (r"$\bar{\rho}$", np.asarray(data["rho"]).T),
        (r"$\bar{u}$", np.asarray(data["u"]).T),
        (r"$\bar{v}$", np.asarray(data["v"]).T),
        (r"$\bar{T}$", np.asarray(data["T"]).T),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), constrained_layout=True)
    for ax, (label, field) in zip(axes.ravel(), fields, strict=True):
        mesh = ax.pcolormesh(x, y, field, shading="auto", cmap="viridis")
        ax.contour(x, y, np.asarray(data["u"]).T, levels=[0.0], colors="white", linewidths=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/X_{\mathrm{sh}}$")
        ax.set_ylabel(r"$y/X_{\mathrm{sh}}$")
        fig.colorbar(mesh, ax=ax, label=label)
    fig.savefig(DST, dpi=300)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
