#!/usr/bin/env python3
"""Plot fd-order convergence for the Song2023 validation case.

Reads order_convergence_summary.json and produces:
  - validation/song_2023/order_convergence.png

Shows: (a) mean nearest-MATLAB eigenvalue distance vs fd_order,
       (b) leading-mode growth-rate error vs fd_order.

Usage (from repo root):
    .venv/bin/python validation/song_2023/plot_order_convergence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
SUMMARY_JSON = REPO / "validation/song_2023/order_convergence_runs/order_convergence_summary.json"
OUTPUT_PNG = SCRIPT.with_name("order_convergence.png")


def main() -> None:
    if not SUMMARY_JSON.exists():
        print(f"ERROR: {SUMMARY_JSON} not found. Run the order-convergence study first.")
        sys.exit(1)

    with SUMMARY_JSON.open() as f:
        results = json.load(f)

    # Collect data from successful runs
    fd_orders: list[int] = []
    mean_dists: list[float] = []
    max_dists: list[float] = []
    lead_im_err: list[float] = []
    lead_im_py: list[float] = []
    lead_im_ml: list[float] = []
    total_times: list[float] = []
    peak_mems: list[float] = []

    for item in results:
        if item.get("status") != "ok":
            continue
        comp = item.get("comparison", {})
        if not comp or comp.get("status"):
            continue

        fd_orders.append(item["fd_order"])
        mean_dists.append(comp["mean_nearest_matlab"])
        max_dists.append(comp["max_nearest_matlab"])
        lead = comp["leading_eigenvalue"]
        lead_im_py.append(lead["python_im"])
        lead_im_ml.append(lead["matlab_im"])
        lead_im_err.append(abs(lead["python_im"] - lead["matlab_im"]))
        total_times.append(item.get("summary", {}).get("total_time_s", 0))
        peak_mems.append(item.get("summary", {}).get("peak_memory_mib", 0))

    if len(fd_orders) < 2:
        print(f"Need at least 2 successful runs for a convergence plot, got {len(fd_orders)}.")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Mean nearest-MATLAB distance
    ax = axes[0]
    ax.semilogy(fd_orders, mean_dists, "ko-", markersize=7, label="mean")
    ax.semilogy(fd_orders, max_dists, "s--", color="0.5", markersize=5, label="max")
    ax.set_xlabel("FD order")
    ax.set_ylabel("Nearest-MATLAB distance")
    ax.set_title("(a) Spectrum agreement")
    ax.set_xticks(fd_orders)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Leading-mode growth-rate error
    ax = axes[1]
    ax.semilogy(fd_orders, lead_im_err, "ko-", markersize=7)
    ax.set_xlabel("FD order")
    ax.set_ylabel(r"$|\mathrm{Im}(\omega_1)_\mathrm{Py} - \mathrm{Im}(\omega_1)_\mathrm{ML}|$")
    ax.set_title(r"(b) Leading $\omega_1$ error")
    ax.set_xticks(fd_orders)
    ax.grid(True, alpha=0.3)

    # (c) Timing and memory
    ax = axes[2]
    ax_mem = ax.twinx()
    bars_time = ax.bar(
        [x - 0.2 for x in fd_orders], total_times, width=0.4,
        color="steelblue", alpha=0.8, label="Time [s]",
    )
    bars_mem = ax_mem.bar(
        [x + 0.2 for x in fd_orders], [m / 1024 for m in peak_mems], width=0.4,
        color="coral", alpha=0.8, label="Peak [GB]",
    )
    ax.set_xlabel("FD order")
    ax.set_ylabel("Wall time [s]", color="steelblue")
    ax_mem.set_ylabel("Peak memory [GB]", color="coral")
    ax.set_title("(c) Cost")
    ax.set_xticks(fd_orders)
    ax.legend(handles=[bars_time, bars_mem], fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
