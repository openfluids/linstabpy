# SPDX-License-Identifier: GPL-3.0-only
"""Fig 1: Lui stride-2 base-flow primitives from sawtooth.npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "sawtooth.npz"
DST = HERE / "sawtooth_fig1_baseflow.png"


def main() -> int:
    data = np.load(SRC)
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    fields = [
        (r"$\bar{\rho}$", np.asarray(data["rho"])),
        (r"$\bar{u}$", np.asarray(data["u"])),
        (r"$\bar{v}$", np.asarray(data["v"])),
        (r"$\bar{T}$", np.asarray(data["T"])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), constrained_layout=True)
    for ax, (label, field) in zip(axes.ravel(), fields, strict=True):
        mesh = ax.pcolormesh(x, y, field, shading="auto", cmap="viridis")
        ax.contour(x, y, np.asarray(data["u"]), levels=[0.0], colors="white", linewidths=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/c_x$")
        ax.set_ylabel(r"$y/c_x$")
        fig.colorbar(mesh, ax=ax, label=label)
    fig.savefig(DST, dpi=300)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
