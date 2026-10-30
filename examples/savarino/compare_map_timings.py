#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Compare the 200-point MUMPS farm with the assemble-once line campaign.

Reads savarino_map3_200pt.csv (or another point-farm CSV) and every
savarino_line_b*.csv. Writes savarino_map_timing.csv with matched
(beta, f) gains and walls. No solver. No figure.

Usage:
    python compare_map_timings.py savarino_map3_200pt.csv campaign_line
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, str]) -> tuple[float, float]:
    return (float(row["beta_Xsh"]), float(row["f_Xsh_Uinf"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("point_csv", type=Path)
    ap.add_argument("line_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    points = {_key(r): r for r in _load(args.point_csv)}
    line_rows: list[dict[str, str]] = []
    for path in sorted(args.line_dir.glob("savarino_line_b*.csv")):
        line_rows.extend(_load(path))
    out = args.output or (args.line_dir / "savarino_map_timing.csv")
    fields = [
        "beta_Xsh",
        "f_Xsh_Uinf",
        "gain_point",
        "gain_line",
        "gain_rel",
        "wall_s_point",
        "wall_s_line",
        "assemble_s",
        "window_s",
        "factor_s",
        "gmres_s",
        "rsvd_s",
        "recycle_setup_s",
        "line_wall_s",
        "method_point",
        "method_line",
        "status_point",
        "status_line",
    ]
    n_match = 0
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in line_rows:
            key = _key(row)
            other = points.get(key)
            if other is None:
                continue
            n_match += 1
            gp = float(other["gain"])
            gl = float(row["gain"])
            rel = abs(gl - gp) / abs(gp) if gp != 0.0 else float("nan")
            writer.writerow(
                {
                    "beta_Xsh": row["beta_Xsh"],
                    "f_Xsh_Uinf": row["f_Xsh_Uinf"],
                    "gain_point": other["gain"],
                    "gain_line": row["gain"],
                    "gain_rel": f"{rel:.6e}",
                    "wall_s_point": other["wall_s"],
                    "wall_s_line": row["wall_s"],
                    "assemble_s": row.get("assemble_s", ""),
                    "window_s": row.get("window_s", ""),
                    "factor_s": row.get("factor_s", ""),
                    "gmres_s": row.get("gmres_s", ""),
                    "rsvd_s": row.get("rsvd_s", ""),
                    "recycle_setup_s": row.get("recycle_setup_s", ""),
                    "line_wall_s": row.get("line_wall_s", ""),
                    "method_point": other.get("backend", "mumps_point"),
                    "method_line": row.get("method", "line_mumps"),
                    "status_point": other.get("status", ""),
                    "status_line": row.get("status", ""),
                }
            )
    print(f"wrote {out} matched={n_match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
