#!/usr/bin/env python3
"""Plot optimal forcing and optimal response mode shapes from a
resolvent run.

Usage:
    .venv/bin/python studies/case2/phase1p5_plot_resolvent_modes.py \\
        <npz_path>

Shows |u| of the leading forcing and response modes.
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
        print("usage: phase1p5_plot_resolvent_modes.py <npz>")
        sys.exit(2)
    npz = Path(sys.argv[1])
    d = np.load(npz, allow_pickle=True)
    X = d["X"]
    Y = d["Y"]
    ny, nx = X.shape
    gains = d.get("gains", np.array([]))
    responses = d.get("responses")
    forcings = d.get("forcings")
    idx_u = d["idx_u_j"]
    omega = complex(d.get("omega_target", 0))

    if responses is None or responses.size == 0:
        print(f"No responses in {npz}")
        sys.exit(1)

    n_modes = min(responses.shape[1], 3)
    fig, axes = plt.subplots(n_modes, 2, figsize=(12, 4 * n_modes),
                              constrained_layout=True, squeeze=False)
    for j in range(n_modes):
        resp_u = np.abs(responses[idx_u, j]).reshape((ny, nx), order="F")
        resp_u /= resp_u.max()
        force_u = np.abs(forcings[idx_u, j]).reshape((ny, nx), order="F")
        force_u /= force_u.max()

        ax = axes[j, 0]
        pc = ax.pcolormesh(X, Y, force_u, cmap="magma", shading="auto",
                            vmin=0, vmax=1)
        fig.colorbar(pc, ax=ax, label=r"$|u|$ forcing / max")
        ax.set_aspect("equal")
        ax.set_title(f"Mode {j+1}: σ = {gains[j]:.3e} — forcing")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax = axes[j, 1]
        pc = ax.pcolormesh(X, Y, resp_u, cmap="viridis", shading="auto",
                            vmin=0, vmax=1)
        fig.colorbar(pc, ax=ax, label=r"$|u|$ response / max")
        ax.set_aspect("equal")
        ax.set_title(f"Mode {j+1}: σ = {gains[j]:.3e} — response")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"{npz.stem}\nω = {omega}", fontsize=10)
    out = OUT_DIR / f"{npz.stem}_fr_modes.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
