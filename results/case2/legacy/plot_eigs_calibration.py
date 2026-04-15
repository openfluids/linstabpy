#!/usr/bin/env python3
"""Plot Case 2 eigensolver calibration summaries from perf_memory CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


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
    parser.add_argument("--x", default="stability_eps_ncv", help="CSV column for x-axis")
    parser.add_argument("--title", default="Case 2 Stability Calibration")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [row for row in _load_rows(args.report_csv) if row.get("status") == "ok"]
    if not rows:
        raise SystemExit(f"no successful rows in {args.report_csv}")

    rows.sort(key=lambda row: _to_float(row, args.x))
    x = np.array([_to_float(row, args.x) for row in rows], dtype=float)
    elapsed_min = np.array([_to_float(row, "elapsed_seconds") / 60.0 for row in rows], dtype=float)
    peak_gib = np.array([_to_float(row, "peak_rss_gib") for row in rows], dtype=float)
    post_1e5 = np.array([_to_int(row, "stability_postcheck_residual_rel_count_le_1e4") for row in rows], dtype=float)
    post_1e6 = np.array([_to_int(row, "stability_postcheck_residual_rel_count_le_1e6") for row in rows], dtype=float)
    nconv = np.array([_to_int(row, "stability_eps_nconv") for row in rows], dtype=float)
    nev = np.array([_to_int(row, "n_modes_stability_requested") for row in rows], dtype=float)
    median_post = np.array([_to_float(row, "stability_postcheck_residual_rel_median") for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    axes[0, 0].plot(x, nev, marker="o", color="#9f9f9f", linewidth=2, label="requested nev")
    axes[0, 0].plot(x, nconv, marker="o", color="#00798c", linewidth=2, label="EPS nconv")
    axes[0, 0].plot(x, post_1e5, marker="s", color="#c76b00", linewidth=2, label="postcheck <= 1e-4")
    axes[0, 0].plot(x, post_1e6, marker="^", color="#7a3e9d", linewidth=2, label="postcheck <= 1e-6")
    axes[0, 0].set_title("Accepted Modes")
    axes[0, 0].set_xlabel(args.x)
    axes[0, 0].set_ylabel("Mode count")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(x, elapsed_min, marker="o", color="#1f3b4d", linewidth=2)
    axes[0, 1].set_title("Runtime")
    axes[0, 1].set_xlabel(args.x)
    axes[0, 1].set_ylabel("Minutes")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(x, peak_gib, marker="o", color="#287d3c", linewidth=2)
    axes[1, 0].set_title("Peak Memory")
    axes[1, 0].set_xlabel(args.x)
    axes[1, 0].set_ylabel("GiB")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(x, median_post, marker="o", color="#9a031e", linewidth=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Median Postcheck Residual")
    axes[1, 1].set_xlabel(args.x)
    axes[1, 1].set_ylabel("relative residual")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(args.title)
    output = args.output or args.report_csv.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
