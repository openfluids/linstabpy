#!/usr/bin/env python3
"""Plot |u| component of returned eigenmodes over the (x,y) domain.

Usage:
    .venv/bin/python studies/case2/phase1p5_plot_modes.py <npz_path> [<out_name>]

Writes to studies/case2/phase1p5/<out_name>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: phase1p5_plot_modes.py <npz_path> [<out_name>]")
        sys.exit(2)
    npz = Path(sys.argv[1])
    out_name = sys.argv[2] if len(sys.argv) >= 3 else npz.stem + "_modes"
    d = np.load(npz, allow_pickle=True)
    X = d["X"]
    Y = d["Y"]
    modes = d["direct_modes"]
    omegas = d["omega"]
    idx_u = d["idx_u_j"]
    ny, nx = X.shape

    n_modes = modes.shape[1]
    fig, axes = plt.subplots(
        n_modes, 2, figsize=(12, 4.5 * n_modes), constrained_layout=True,
        squeeze=False,
    )
    for j in range(n_modes):
        u_flat = modes[idx_u, j]
        u_grid = u_flat.reshape((ny, nx), order="F")
        mag = np.abs(u_grid)
        mag_norm = mag / mag.max()
        phase = np.angle(u_grid)

        ax = axes[j, 0]
        pc = ax.pcolormesh(X, Y, mag_norm, shading="auto", cmap="magma",
                            vmin=0, vmax=1)
        fig.colorbar(pc, ax=ax, label=r"$|u|/\max|u|$")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Mode {j}: ω = {omegas[j].real:+.3f} {omegas[j].imag:+.3f}j — |u|")

        ax = axes[j, 1]
        # Mask phase where |u| < 2% of peak — phase undefined in near-zero regions
        mask = mag_norm < 0.02
        phase_masked = np.ma.array(phase, mask=mask)
        pc = ax.pcolormesh(X, Y, phase_masked, shading="auto",
                           cmap="twilight", vmin=-np.pi, vmax=np.pi)
        fig.colorbar(pc, ax=ax, label=r"arg$(u)$ [rad]")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Mode {j}: phase (masked where |u|<0.02)")

    fig.suptitle(f"Eigenmodes of {npz.name}", fontsize=10)
    out = OUT_DIR / f"{out_name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
