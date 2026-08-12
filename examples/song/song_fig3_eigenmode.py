#!/usr/bin/env python3
"""Fig S3: Song leading eigenmode at 32 deg from song_32deg_leading_mode.npz.

The supplement figure is a side-by-side comparison. The linstabpy partner
field from that run is not in the repo, so this script draws the Song
panels only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "song_32deg_leading_mode.npz"
DST = HERE / "song_fig3_eigenmode.png"
VARS = ("rho", "u", "v", "w", "t")


def main() -> int:
    data = np.load(SRC)
    x = np.asarray(data["x"]).T
    y = np.asarray(data["y"]).T
    fig, axes = plt.subplots(len(VARS), 1, figsize=(6.4, 9.2), constrained_layout=True)
    for ax, name in zip(axes, VARS, strict=True):
        field = np.real(np.asarray(data[name])).T
        scale = np.max(np.abs(field))
        if scale > 0.0:
            field = field / scale
        mesh = ax.pcolormesh(x, y, field, shading="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_aspect("equal")
        ax.set_ylabel(r"$y/X_{\mathrm{sh}}$")
        ax.set_title(rf"$\Re({name})$ / $\max|\Re|$")
        fig.colorbar(mesh, ax=ax)
    axes[-1].set_xlabel(r"$x/X_{\mathrm{sh}}$")
    fig.savefig(DST, dpi=300)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
