#!/usr/bin/env python3
"""Overlay MATLAB and Python Song spectra across FD orders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
SCALING_SUMMARY = REPO / "validation/song_2023/scaling_runs/summary.json"
ORDER_SUMMARY = REPO / "validation/song_2023/order_convergence_runs/order_convergence_summary.json"
MATLAB_SPECTRUM = REPO / "validation/song_2023/matlab_reference/EigModes/spectrum.mat"
OUTPUT_PNG = SCRIPT.with_name("song_fd_spectrum_sensitivity.png")

FD_STYLES = {
    2: {"label": "Python fd2", "color": "#1f77b4", "marker": "o"},
    4: {"label": "Python fd4", "color": "#ff7f0e", "marker": "s"},
    6: {"label": "Python fd6", "color": "#2ca02c", "marker": "^"},
    8: {"label": "Python fd8", "color": "#d62728", "marker": "D"},
}


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matlab_omega(path: Path) -> np.ndarray:
    data = loadmat(path)
    omega = np.asarray(data["omega"], dtype=complex)
    if omega.ndim == 2 and omega.shape[0] == omega.shape[1]:
        return np.diag(omega)
    return omega.reshape(-1)


def _collect_npz_by_fd() -> dict[int, Path]:
    by_fd: dict[int, tuple[int, Path]] = {}
    sources = _load_json(SCALING_SUMMARY) + _load_json(ORDER_SUMMARY)
    for item in sources:
        if item.get("status") != "ok":
            continue
        fd = item.get("fd_order")
        if fd is None:
            name = str(item.get("variant", ""))
            if name.startswith("fd"):
                try:
                    fd = int(name.split("_", 1)[0][2:])
                except ValueError:
                    fd = None
        if fd not in FD_STYLES:
            continue
        ranks = int(item.get("ranks", 1))
        npz_rel = item.get("output_npz")
        if npz_rel is None:
            npz_rel = f"validation/song_2023/scaling_runs/{item['variant']}/song_2023.npz"
        npz_path = REPO / npz_rel
        if not npz_path.exists():
            continue
        # Prefer serial results when available, then lower-rank distributed runs.
        score = 0 if ranks == 1 else ranks
        current = by_fd.get(int(fd))
        if current is None or score < current[0]:
            by_fd[int(fd)] = (score, npz_path)
    return {fd: path for fd, (_, path) in by_fd.items()}


def main() -> None:
    matlab = _load_matlab_omega(MATLAB_SPECTRUM)
    npz_by_fd = _collect_npz_by_fd()
    if len(npz_by_fd) < 2:
        raise SystemExit("Need at least two successful FD-order spectra before plotting.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(np.real(matlab), np.imag(matlab), s=18, c="k", marker="x", alpha=0.65, label="MATLAB ref")

    for fd in sorted(npz_by_fd):
        with np.load(npz_by_fd[fd], allow_pickle=False) as data:
            omega = np.asarray(data["omega"], dtype=complex).reshape(-1)
        style = FD_STYLES[fd]
        ax.scatter(
            np.real(omega),
            np.imag(omega),
            s=28,
            c=style["color"],
            marker=style["marker"],
            alpha=0.75,
            label=style["label"],
        )

    ax.set_xlabel(r"$\Re(\omega)$")
    ax.set_ylabel(r"$\Im(\omega)$")
    ax.set_title("Song2023 Spectrum Sensitivity to FD Order")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
