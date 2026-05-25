#!/usr/bin/env python3
"""Plot the Song baseflow fields (x, y, rho, u, v, w, T, sponge).

Usage (run from this directory):
  python scripts/validation/song_2023/scripts/plot_baseflow.py
  python scripts/validation/song_2023/scripts/plot_baseflow.py /path/to/another/baseflow

Reads x.npy, y.npy, rho.npy, u.npy, v.npy, w.npy, T.npy, sponge.npy and saves
baseflow_plot.png (and optionally separate panels) in the same directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
SONG_DIR = SCRIPT.parents[1]
DEFAULT_BASEFLOW_DIR = SONG_DIR / "data" / "baseflow"


def _load_field(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be 2-D, got shape {arr.shape}")
    return arr


def _mesh_for_plot(x: np.ndarray, y: np.ndarray):
    """Return (X_2d, Y_2d) for plotting. Handle both 1D and 2D x,y."""
    if x.ndim == 2 and y.ndim == 2:
        return x, y
    if x.ndim == 1 and y.ndim == 1:
        return np.meshgrid(x, y, indexing="ij")
    raise ValueError(f"x shape {x.shape}, y shape {y.shape} not supported")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseflow fields from .npy files.")
    parser.add_argument(
        "dir",
        nargs="?",
        default=DEFAULT_BASEFLOW_DIR,
        type=Path,
        help="Directory containing x.npy, y.npy, rho.npy, u.npy, v.npy, w.npy, T.npy, sponge.npy",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="baseflow_plot.png",
        help="Output filename (default: baseflow_plot.png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    required = ["x", "y", "rho", "u", "v", "w", "T", "sponge"]
    for name in required:
        if not (root / f"{name}.npy").exists():
            raise FileNotFoundError(f"Missing {root / f'{name}.npy'}")

    x = _load_field(root / "x.npy")
    y = _load_field(root / "y.npy")
    rho = _load_field(root / "rho.npy")
    u = _load_field(root / "u.npy")
    v = _load_field(root / "v.npy")
    w = _load_field(root / "w.npy")
    T = _load_field(root / "T.npy")
    sponge = _load_field(root / "sponge.npy")

    X_plot, Y_plot = _mesh_for_plot(x, y)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    fields = [
        (rho, r"$\rho$"),
        (u, "$u$"),
        (v, "$v$"),
        (w, "$w$"),
        (T, "$T$"),
        (sponge, "sponge"),
        (np.sqrt(u**2 + v**2 + w**2), r"$|U|$"),
        (np.where(sponge > 1e-10, sponge, np.nan), "sponge (active)"),
    ]
    for ax, (data, title) in zip(axes, fields):
        if np.all(np.isnan(data)):
            ax.set_title(title)
            ax.text(0.5, 0.5, "no active sponge", ha="center", va="center", transform=ax.transAxes)
            ax.set_aspect("equal")
            continue
        # Use contourf to avoid pcolormesh warning on non-monotonic (curvilinear) grids
        p = ax.contourf(X_plot, Y_plot, data, levels=32, cmap="viridis", extend="both")
        ax.set_aspect("equal")
        ax.set_title(title)
        plt.colorbar(p, ax=ax, shrink=0.7)

    plt.suptitle(f"Baseflow — {root.resolve().name}")
    plt.tight_layout()

    out_path = root / args.output
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
