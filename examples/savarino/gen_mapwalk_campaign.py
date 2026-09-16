#!/usr/bin/env python3
"""Write one Slurm job per Savarino map-walk beta line.

The walker factors the first two frequencies of a beta line, then tries
to walk the rest. Each job times one beta line against the committed farm
walls in savarino_map3_200pt.csv.

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

from examples.savarino.gen_campaign import env_lines, gpu_block  # noqa: E402

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
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="VAR=value",
        help="exported into the job script (repeatable), e.g. I_MPI_PIN_DOMAIN=auto",
    )
    ap.add_argument(
        "--n-anchor",
        type=int,
        default=None,
        help="direct points per line before walking; >= line length = pure direct arm",
    )
    ap.add_argument(
        "--ncv", type=int, default=None, help="SLEPc subspace size for the gain (nev=1)"
    )
    ap.add_argument(
        "--driver-out",
        type=Path,
        default=REPO / "examples" / "savarino",
        help="where the driver writes savarino_mapwalk.csv/.log (one dir per arm)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=HERE / "configs" / "mack_f2_b45.yaml",
        help="point config for the line; its solver block decides CPU (MUMPS) or "
        "GPU (STRUMPACK or cuDSS under FGMRES) -- see savarino_point_gpu.yaml "
        "and savarino_point_cudss.yaml",
    )
    ap.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="Slurm cores per rank (OMP threads for a GPU job's CPU part)",
    )
    ap.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="devices on the node; >0 gives each job one GPU by lock file",
    )
    args = ap.parse_args()

    driver_args = ""
    if args.n_anchor is not None:
        driver_args += f" --n-anchor {int(args.n_anchor)}"
    if args.ncv is not None:
        driver_args += f" --ncv {int(args.ncv)}"
    template = TEMPLATE.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    args.driver_out.mkdir(parents=True, exist_ok=True)
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
                env_lines=env_lines(args.env),
                driver_out=args.driver_out,
                driver_args=driver_args,
                config=Path(args.config).resolve(),
                cpus_per_task=args.cpus_per_task,
                gpu_block=gpu_block(args.gpus, args.driver_out),
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
