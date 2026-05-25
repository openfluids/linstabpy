#!/usr/bin/env python3
"""Plot |u| and Chu-energy density for the Dussauge-range GLOBAL
eigenmodes — the candidate shock-breathing modes.

Picks specific modes by index from specific npz files and renders a
clean 2-row × N-col grid: top row |u|, bottom row Chu energy density.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
OUT_DIR = CASE_DIR / "artifacts" / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)

# Each entry: (label, npz_path, mode_index)
SELECTION = [
    (
        "GLOBAL ω≈(+0.235, -7.129)  St=0.037",
        "examples/case_studies/case2_lui/output/evp_stride2_m2_topology_probe_qraw/runs/case2_perf_probe_m2_0p4_m6p2_20260419T080329Z.npz",
        13,
    ),
    (
        "GLOBAL ω≈(+0.333, -7.270)  St=0.053",
        "examples/case_studies/case2_lui/output/evp_stride2_m2_topology_probe_qraw/runs/case2_perf_probe_m2_0p4_m6p2_20260419T080329Z.npz",
        18,
    ),
    (
        "GLOBAL ω≈(+1.742, -6.317)  St=0.277",
        "examples/case_studies/case2_lui/output/evp_stride2_m2_screening_qraw/runs/case2_perf_bstat_m2_qraw_nev20_ncv500_tol5em3_20260418T164130Z.npz",
        17,
    ),
]


def chu_energy_per_cell(d: dict, j: int) -> np.ndarray:
    modes = d["direct_modes"]
    W = d["W_diag"]
    X = d["X"]
    ny, nx = X.shape
    ec = W * np.abs(modes[:, j]) ** 2
    ec_per_cell = np.zeros(ny * nx)
    for name in ("rho", "u", "v", "w", "T"):
        ec_per_cell += ec[d[f"idx_{name}_j"]]
    return ec_per_cell.reshape((ny, nx), order="F")


def main() -> None:
    n = len(SELECTION)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 8.5), constrained_layout=True)
    for col, (label, npz_path, j) in enumerate(SELECTION):
        d_obj = np.load(npz_path, allow_pickle=True)
        d = {k: d_obj[k] for k in d_obj.files}
        X = d["X"]
        Y = d["Y"]
        ny, nx = X.shape
        idx_u = d["idx_u_j"]
        modes = d["direct_modes"]
        u_grid = modes[idx_u, j].reshape((ny, nx), order="F")
        mag_u = np.abs(u_grid)
        mag_u /= mag_u.max()
        ec = chu_energy_per_cell(d, j)
        ec_norm = ec / ec.max()

        ax = axes[0, col]
        pc = ax.pcolormesh(X, Y, mag_u, shading="auto", cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(pc, ax=ax, label=r"$|u|/\max$")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label, fontsize=9)

        ax = axes[1, col]
        pc = ax.pcolormesh(
            X, Y, np.log10(ec_norm + 1e-6), shading="auto", cmap="magma", vmin=-4, vmax=0
        )
        fig.colorbar(pc, ax=ax, label=r"$\log_{10}(E_\text{Chu}/\max)$")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Case 2 M2 Qraw — candidate GLOBAL oscillatory eigenmodes", fontsize=11)
    out = OUT_DIR / "global_oscillatory_candidates.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
