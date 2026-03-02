#!/usr/bin/env python3
"""
example_grid: Generate a representative stretched boundary-layer grid figure.

OUTPUTS: example_grid.npz, example_grid.png
USAGE:   python example_grid.py                        # .npz exists -> plot only
         rm example_grid.npz && python example_grid.py # recompute
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")


def _stretch_y(eta: np.ndarray, y_max: float, s: float) -> np.ndarray:
    t = np.tanh(s * (1.0 - eta)) / np.tanh(s)
    return y_max * (1.0 - t) / (1.0 + t)


def compute() -> None:
    nx = 220
    ny = 120
    x_max = 1.2
    y_max = 40.0
    s = 2.8

    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)

    x = x_max * xi
    y = _stretch_y(eta, y_max, s)

    X, Y = np.meshgrid(x, y, indexing="xy")

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        Y=Y,
        nx=nx,
        ny=ny,
        x_max=x_max,
        y_max=y_max,
        s=s,
    )


def plot(data: np.lib.npyio.NpzFile) -> None:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    nx = int(np.asarray(data["nx"]).item())
    ny = int(np.asarray(data["ny"]).item())
    s = float(np.asarray(data["s"]).item())

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    for j in range(0, ny, max(1, ny // 24)):
        ax.plot(X[j, :], Y[j, :], color="0.35", linewidth=0.55)
    for i in range(0, nx, max(1, nx // 28)):
        ax.plot(X[:, i], Y[:, i], color="0.35", linewidth=0.55)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(
        "Representative stretched grid\n"
        rf"$N_x={nx}$, $N_y={ny}$, stretch parameter $s={s:.1f}$"
    )
    ax.set_xlim(float(np.min(X)), float(np.max(X)))
    ax.set_ylim(float(np.min(Y)), float(np.max(Y)))
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    except FileNotFoundError:
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
