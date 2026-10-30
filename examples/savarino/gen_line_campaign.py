#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Write one Slurm job per Savarino beta line.

The point campaign (gen_campaign.py) factors L0 at every (beta, f) in its
own job.
This generator keeps the same GRID and the same P(f).
It runs all frequencies of one beta in one process after a single assembly.

Do not submit these jobs on a node that still runs the 200-point farm.

Usage:
    python gen_line_campaign.py --out campaign_line --dry-run
    python gen_line_campaign.py --out campaign_line
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from examples.savarino.gen_campaign import GRID, tag  # noqa: E402

TEMPLATE = HERE / "savarino_line.sbatch.template"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--partition", default="LocalQ")
    ap.add_argument("--nranks", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--method",
        choices=("recycle", "line_mumps"),
        default="recycle",
        help="recycle first. line_mumps is the later timed MUMPS map.",
    )
    args = ap.parse_args()
    template = TEMPLATE.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    n_pts = sum(len(freqs) for freqs in GRID.values())
    print(f"{len(GRID)} beta lines, {n_pts} frequencies, dry_run={args.dry_run}")
    for beta, freqs in GRID.items():
        stem = f"savarino_line_b{tag(beta)}"
        workdir = args.out / stem
        workdir.mkdir(exist_ok=True)
        sb = args.out / f"{stem}.sbatch"
        sb.write_text(
            template.format(
                jobname=stem,
                workdir=workdir,
                partition=args.partition,
                nranks=args.nranks,
                repo_root=REPO,
                beta=beta,
                nfreq=len(freqs),
                outdir=args.out,
                method=args.method,
                recycle=int(args.method == "recycle"),
            ),
            encoding="utf-8",
        )
        if args.dry_run:
            print(f"{stem}: {len(freqs)} frequencies (dry run, wrote {sb})")
            continue
        result = subprocess.run(["sbatch", str(sb)], capture_output=True, text=True)
        print(f"{stem}: {result.stdout.strip()}{result.stderr.strip()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
