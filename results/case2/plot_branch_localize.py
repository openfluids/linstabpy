#!/usr/bin/env python3
"""Plot Case 2 branch-localization runs using explicit acceptance metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [row for row in csv.DictReader(fh) if row.get("status") == "ok"]


def _to_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw in {"", "None", "nan", "NaN"}:
        return float("nan")
    return float(raw)


def _to_int(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "")
    if raw in {"", "None", "nan", "NaN"}:
        return -1
    return int(float(raw))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default="Case 2 Branch Localization")
    args = parser.parse_args()

    rows = _load_rows(args.report_csv)
    if not rows:
        raise SystemExit(f"no successful rows in {args.report_csv}")

    labels = [row.get("case_id", f"case{i}") for i, row in enumerate(rows)]
    accepted = np.array([_to_int(row, "stability_accepted_count") for row in rows], dtype=float)
    min_dist = np.array([_to_float(row, "stability_postcheck_target_distance_min") for row in rows], dtype=float)
    median_post = np.array([_to_float(row, "stability_postcheck_residual_rel_median") for row in rows], dtype=float)
    elapsed_min = np.array([_to_float(row, "elapsed_seconds") / 60.0 for row in rows], dtype=float)
    peak_gib = np.array([_to_float(row, "peak_rss_gib") for row in rows], dtype=float)
    x = np.arange(len(rows), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    colors = ["#c76b00" if val <= 0 else "#287d3c" for val in accepted]
    axes[0, 0].bar(x, accepted, color=colors)
    axes[0, 0].set_title("Accepted Modes")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=25, ha="right")

    scatter = axes[0, 1].scatter(min_dist, median_post, c=accepted, cmap="viridis", s=90)
    axes[0, 1].set_title("Target Hit vs Postcheck")
    axes[0, 1].set_xlabel("minimum target distance")
    axes[0, 1].set_ylabel("median postcheck residual")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.25)
    for i, label in enumerate(labels):
        axes[0, 1].annotate(label, (min_dist[i], median_post[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    cbar = fig.colorbar(scatter, ax=axes[0, 1])
    cbar.set_label("accepted count")

    axes[1, 0].bar(x, elapsed_min, color="#00798c")
    axes[1, 0].set_title("Runtime")
    axes[1, 0].set_ylabel("minutes")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=25, ha="right")

    axes[1, 1].bar(x, peak_gib, color="#7a3e9d")
    axes[1, 1].set_title("Peak Memory")
    axes[1, 1].set_ylabel("GiB")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=25, ha="right")

    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(args.title)
    output = args.output or args.report_csv.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
