#!/usr/bin/env python3
"""Write one Slurm job per Savarino map-walk beta line.

The walker factors the first two frequencies of a beta line, then tries
to walk the rest. Each job times one beta line against the committed farm
walls in savarino_map2_200pt.csv.

The repository root comes from this file, so the job script has no
absolute path in it.

Usage:
    python gen_mapwalk_campaign.py --out . --beta 40 --dry-run
    python gen_mapwalk_campaign.py --out . --beta 40
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

TEMPLATE = HERE / "savarino_mapwalk.sbatch.template"
HOURS_TO_SECONDS = 3600


def tag(beta: float) -> str:
    """Return a file-name tag for a beta value."""
    return f"{beta:g}".replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beta", type=float, nargs="+", required=True)
    ap.add_argument("--partition", default="LocalQ")
    ap.add_argument("--nranks", type=int, default=4)
    ap.add_argument("--hours", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template = TEMPLATE.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    for beta in args.beta:
        stem = f"savarino_mapwalk_b{tag(beta)}"
        sb = args.out / f"{stem}.sbatch"
        sb.write_text(
            template.format(
                jobname=stem,
                partition=args.partition,
                nranks=args.nranks,
                repo_root=REPO,
                beta=beta,
                walltime=f"{args.hours:02d}:00:00",
                deadline_s=args.hours * HOURS_TO_SECONDS,
            ),
            encoding="utf-8",
        )
        if args.dry_run:
            print(f"{stem}: dry run, wrote {sb}")
            continue
        result = subprocess.run(["sbatch", str(sb)], capture_output=True, text=True, check=False)
        print(f"{stem}: {result.stdout.strip()}{result.stderr.strip()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
