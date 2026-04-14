#!/usr/bin/env python3
"""Plot Song2023 time-to-solution and memory scaling across FD orders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
SUMMARY_JSON = REPO / "validation/song_2023/scaling_runs/summary.json"
OUTPUT_PNG = SCRIPT.with_name("song_scaling_summary.png")

FD_STYLES = {
    2: {"label": "fd2", "color": "#1f77b4", "marker": "o"},
    4: {"label": "fd4", "color": "#ff7f0e", "marker": "s"},
    6: {"label": "fd6", "color": "#2ca02c", "marker": "^"},
    8: {"label": "fd8", "color": "#d62728", "marker": "D"},
}


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}; run the scaling study first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _fd_from_name(name: str) -> int | None:
    head = name.split("_", 1)[0]
    if not head.startswith("fd"):
        return None
    try:
        return int(head[2:])
    except ValueError:
        return None


def main() -> None:
    results = _load_results(SUMMARY_JSON)
    grouped: dict[int, list[dict]] = {}
    for item in results:
        if item.get("status") != "ok":
            continue
        fd = _fd_from_name(str(item.get("variant", "")))
        if fd not in FD_STYLES:
            continue
        grouped.setdefault(fd, []).append(item)

    if not grouped:
        raise SystemExit("No successful scaling results found.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    all_ranks: set[int] = set()

    for fd in sorted(grouped):
        style = FD_STYLES[fd]
        series = sorted(grouped[fd], key=lambda item: int(item.get("ranks", 0)))
        ranks = np.asarray([int(item["ranks"]) for item in series], dtype=int)
        times = np.asarray([float(item["summary"]["total_time_s"]) for item in series], dtype=float)
        peak_sum = np.asarray(
            [float(item["summary"].get("peak_memory_mib_sum_ranks", np.nan)) / 1024.0 for item in series],
            dtype=float,
        )
        all_ranks.update(int(r) for r in ranks)

        axes[0].plot(
            ranks,
            times,
            marker=style["marker"],
            markersize=8,
            color=style["color"],
            linewidth=2.2,
            label=style["label"],
        )
        axes[1].plot(
            ranks,
            peak_sum,
            marker=style["marker"],
            markersize=8,
            color=style["color"],
            linewidth=2.2,
            label=style["label"],
        )
        for x, y in zip(ranks, times):
            axes[0].annotate(
                f"{y/60.0:.1f}m",
                (x, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=style["color"],
            )
        for x, y in zip(ranks, peak_sum):
            axes[1].annotate(
                f"{y:.0f}",
                (x, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=style["color"],
            )

    rank_ticks = sorted(all_ranks)
    if rank_ticks:
        xmin = min(rank_ticks) - 1
        xmax = max(rank_ticks) + 1

    axes[0].set_xlabel("MPI ranks")
    axes[0].set_ylabel("Wall time [s]")
    axes[0].set_title("Time to Solution")
    axes[0].set_xticks(rank_ticks)
    axes[0].set_xlim(xmin, xmax)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=True, fontsize=9)

    axes[1].set_xlabel("MPI ranks")
    axes[1].set_ylabel("Peak memory sum [GiB]")
    axes[1].set_title("Memory Footprint")
    axes[1].set_xticks(rank_ticks)
    axes[1].set_xlim(xmin, xmax)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
