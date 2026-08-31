# SPDX-License-Identifier: GPL-3.0-only
"""Plot Padovan ODE harmonic resolvent singular values vs index."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_singular_values(
    npz_path: str = "padovan_toy.npz",
    out_path: str = "padovan_toy_fig1.png",
) -> None:
    """Load Padovan toy result and plot sigma_j vs j.

    Parameters
    ----------
    npz_path : str
        Path to padovan_toy.npz.
    out_path : str
        Output PNG path.
    """
    data = np.load(npz_path)
    N_h = int(data["N_h"])

    if "sigma_j" in data:
        sigma_j = data["sigma_j"]
        j_vals = np.arange(len(sigma_j))

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.semilogy(j_vals + 1, sigma_j, "ko-", markersize=6, linewidth=1.5)
        ax.set_xlabel(r"Index $j$", fontsize=11)
        ax.set_ylabel(r"Singular value $\sigma_j$", fontsize=11)
        ax.set_title(f"Padovan ODE Harmonic Resolvent: $N_h = {N_h}$", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        print("sigma_j not in npz; skipping plot (requires resolvent4py on c9)")


if __name__ == "__main__":
    plot_singular_values()
