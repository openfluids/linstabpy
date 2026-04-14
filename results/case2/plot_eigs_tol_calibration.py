#!/usr/bin/env python3
"""Plot Case 2 tolerance sweeps grouped by ncv."""

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
    parser.add_argument("--title", default="Case 2 Stability Tolerance Sweep")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _load_rows(args.report_csv)
    if not rows:
        raise SystemExit(f"no successful rows in {args.report_csv}")

    ncv_values = sorted({_to_int(row, "stability_eps_ncv") for row in rows})
    colors = ["#00798c", "#c76b00", "#7a3e9d", "#287d3c", "#9a031e"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for idx, ncv in enumerate(ncv_values):
        group = [row for row in rows if _to_int(row, "stability_eps_ncv") == ncv]
        group.sort(key=lambda row: _to_float(row, "stability_eps_tol"))
        x = np.array([_to_float(row, "stability_eps_tol") for row in group], dtype=float)
        nconv = np.array([_to_int(row, "stability_eps_nconv") for row in group], dtype=float)
        post_1e6 = np.array(
            [_to_int(row, "stability_postcheck_residual_rel_count_le_1e6") for row in group],
            dtype=float,
        )
        elapsed_min = np.array([_to_float(row, "elapsed_seconds") / 60.0 for row in group], dtype=float)
        peak_gib = np.array([_to_float(row, "peak_rss_gib") for row in group], dtype=float)
        median_post = np.array(
            [_to_float(row, "stability_postcheck_residual_rel_median") for row in group],
            dtype=float,
        )
        color = colors[idx % len(colors)]
        label = f"ncv={ncv}"

        axes[0, 0].plot(x, nconv, marker="o", linewidth=2, color=color, label=label)
        axes[0, 0].plot(x, post_1e6, marker="s", linewidth=2, linestyle="--", color=color)
        axes[0, 1].plot(x, elapsed_min, marker="o", linewidth=2, color=color, label=label)
        axes[1, 0].plot(x, peak_gib, marker="o", linewidth=2, color=color, label=label)
        axes[1, 1].plot(x, median_post, marker="o", linewidth=2, color=color, label=label)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("EPS tolerance")

    axes[0, 0].set_title("Converged vs Accepted")
    axes[0, 0].set_ylabel("Mode count")
    axes[0, 0].legend()
    axes[0, 0].text(
        0.02,
        0.02,
        "solid = EPS nconv\n dashed = postcheck <= 1e-6",
        transform=axes[0, 0].transAxes,
        fontsize=9,
        va="bottom",
    )

    axes[0, 1].set_title("Runtime")
    axes[0, 1].set_ylabel("Minutes")

    axes[1, 0].set_title("Peak Memory")
    axes[1, 0].set_ylabel("GiB")

    axes[1, 1].set_title("Median Postcheck Residual")
    axes[1, 1].set_ylabel("relative residual")
    axes[1, 1].set_yscale("log")

    fig.suptitle(args.title)
    output = args.output or args.report_csv.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
