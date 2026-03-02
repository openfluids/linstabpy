#!/usr/bin/env python3
"""
scaling_reference: Generate heuristic complexity-reference slopes for scaling plots.

OUTPUTS: scaling_reference.npz, scaling_reference.png
USAGE:   python scaling_reference.py                        # .npz exists -> plot only
         rm scaling_reference.npz && python scaling_reference.py # recompute
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


def compute() -> None:
    dof = np.array([2.0e4, 5.0e4, 1.0e5, 2.0e5, 5.0e5], dtype=float)
    n0 = float(dof[0])
    ref_n_15 = (dof / n0) ** 1.5
    ref_n_20 = (dof / n0) ** 2.0

    np.savez_compressed(
        OUTPUT_NPZ,
        dof=dof,
        ref_n_15=ref_n_15,
        ref_n_20=ref_n_20,
    )


def plot(data: np.lib.npyio.NpzFile) -> None:
    dof = np.asarray(data["dof"], dtype=float)
    ref_n_15 = np.asarray(data["ref_n_15"], dtype=float)
    ref_n_20 = np.asarray(data["ref_n_20"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    ax.loglog(dof, ref_n_15, "--", linewidth=1.8, color="#0072B2", label=r"$O(n^{1.5})$ reference")
    ax.loglog(dof, ref_n_20, "--", linewidth=1.8, color="#D55E00", label=r"$O(n^{2})$ reference")

    ax.set_xlabel("degrees of freedom (DOF)")
    ax.set_ylabel("normalized wall time")
    ax.set_title("Scaling-reference slopes (template)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.28)
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
