#!/usr/bin/env python3
"""Populate paper/supp.tex tab:scaling from order-convergence JSON results.

Reads the order_convergence_summary.json produced by
run_song_2023_order_convergence.py and replaces the 'pending' entries
in supp.tex with actual numbers.

Usage (from repo root):
    .venv/bin/python validation/song_2023/scripts/populate_scaling_table.py

Reads:  validation/song_2023/order_convergence_runs/order_convergence_summary.json
Writes: paper/supp.tex (in-place edit of tab:scaling rows)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
SUMMARY_JSON = REPO / "validation/song_2023/order_convergence_runs/order_convergence_summary.json"
SUPP_TEX = REPO / "paper/supp.tex"


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "---"
    return f"{seconds:.0f}"


def _format_memory_gb(mib: float | None) -> str:
    if mib is None:
        return "---"
    return f"{mib / 1024:.1f}"


def main() -> None:
    if not SUMMARY_JSON.exists():
        print(f"ERROR: {SUMMARY_JSON} not found. Run the order-convergence study first.")
        sys.exit(1)

    with SUMMARY_JSON.open() as f:
        results = json.load(f)

    # Build replacement lines indexed by fd_order
    replacements: dict[int, str] = {}
    for item in results:
        if item.get("status") != "ok":
            continue
        fd = item["fd_order"]
        summary = item.get("summary", {})

        # Extract step timings
        step_summary = summary.get("step_summary", {})
        assemble_s = step_summary.get("assemble_operator_bc", {}).get("time_mean")
        stability_s = step_summary.get("stability", {}).get("time_mean")
        total_s = summary.get("total_time_s")
        peak_mib = summary.get("peak_memory_mib")

        line = (
            f"fd{fd} & 901\\,495 & "
            f"{_format_time(assemble_s)} & "
            f"{_format_time(stability_s)} & "
            f"{_format_time(total_s)} & "
            f"{_format_memory_gb(peak_mib)} \\\\"
        )
        replacements[fd] = line

    if not replacements:
        print("ERROR: No successful runs found in the summary JSON.")
        sys.exit(1)

    # Read supp.tex and replace pending lines
    tex = SUPP_TEX.read_text(encoding="utf-8")
    lines = tex.split("\n")
    changed = 0

    for i, line in enumerate(lines):
        # Match lines like: fd2 & 901\,495 & pending & pending & pending & pending \\
        m = re.match(r"^fd(\d+)\s*&\s*901", line)
        if m:
            fd = int(m.group(1))
            if fd in replacements:
                lines[i] = replacements[fd]
                changed += 1
                print(f"  Replaced fd{fd} row")

    if changed == 0:
        print("WARNING: No rows matched in supp.tex. Check table format.")
    else:
        SUPP_TEX.write_text("\n".join(lines), encoding="utf-8")
        print(f"Updated {changed} rows in {SUPP_TEX.relative_to(REPO)}")


if __name__ == "__main__":
    main()
