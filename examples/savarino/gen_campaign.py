#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the 200-point campaign configs and batch scripts.

Reproduce the exact point grid used for the published Savarino resolvent
gain map (see ``savarino_map3_200pt.csv``).
Those are his own sample points, recovered from the Zenodo mode-file
listing, not a grid of our choosing.
Requires the base config ``configs/map_base.yaml`` and the batch-job
template ``configs/point.sbatch.template``.

Conventions (each with a measurement behind it; see the paper supplement).
  sponge          OFF   -- his treatment.
  mask_bc_forcing ON.
  forcing norm    L2    -- his pair is L2 forcing / Chu response.
  epsilon         0     -- rectilinear mesh, no metric cross-coupling.
  P (input_region)      -- measured from his 13 published forcing modes.
                           vars u,v,w; y in [0, 0.29627645]; x in [0.20291667, x_max].
                           x_max = 1.9470833 for f = 0 and f >= 2 (full domain).
                           x_max = 1.7954167 otherwise.
  ASSUMPTION: 1 < f < 2 has no measurement (he publishes f=1 -> 1.795, f=2 -> 1.947).
  Those four frequencies (1.2, 1.4, 1.6, 1.8) take the 1.795 value.

Usage.
    python gen_campaign.py --out results/campaign                      # 200-pt map (Slurm).
    python gen_campaign.py --out results/campaign --grid dense317      # 317-pt map.
    python gen_campaign.py --out results/campaign --dry-run              # generate only.
    python gen_campaign.py --out results/campaign --local              # print mpiexec commands.
"""

from __future__ import annotations

import argparse
import copy
import csv
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from linstabpy.utils.yaml_io import load_config  # noqa: E402

CONFIGS = HERE / "configs"
BASE_CONFIG = CONFIGS / "map_base.yaml"
TEMPLATE = CONFIGS / "point.sbatch.template"
MAP317_CSV = CONFIGS / "savarino_map317_points.csv"
CASE_DIR = HERE

FULL = [
    0,
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    1,
    1.2,
    1.4,
    1.6,
    1.8,
    2,
    2.1,
    2.15,
    2.2,
    2.3,
    2.4,
    2.5,
    2.8,
    3,
    3.5,
    4,
    4.5,
    5,
    6,
    7,
    8,
    9,
    10,
]
BACKBONE = [0, 1e-4, 0.1, 0.5, 5, 10]
F15 = [0, 1e-4, 0.1, 0.5, 1, 1.4, 1.6, 1.8, 2, 2.1, 2.15, 2.2, 2.5, 5, 10]
MACK9 = [1, 1.4, 1.6, 1.8, 2, 2.1, 2.15, 2.2, 2.5]

GRID = {
    0.0: BACKBONE,
    7.5: [1],
    10.0: FULL,
    15.0: [1e-4],
    30.0: F15,
    40.0: MACK9,
    45.0: FULL,
    50.0: F15,
    70.0: F15,
    100.0: BACKBONE,
    140.0: BACKBONE,
    163.0: FULL,
    180.0: BACKBONE,
    200.0: BACKBONE,
}

X_MIN, Y_MAX = 0.20291667, 0.29627645
TWO_PI = 6.283185307179586


def x_max(f: float) -> float:
    return 1.9470833 if (f == 0 or f >= 2) else 1.7954167


def tag(v):
    return str(v).replace(".", "p").replace("-", "m").replace("+", "")


def env_lines(pairs):
    """Render ``VAR=value`` pairs as ``export`` lines for a job script.

    One line per pair, so the job carries its own rank placement
    (``I_MPI_PIN_DOMAIN=auto``, ``I_MPI_PIN=off``) and interpreter
    (``LINSTABPY_PYTHON=...``).
    It does not inherit them from whoever submitted.
    A value with whitespace (``PETSC_OPTIONS=-use_gpu_aware_mpi 0``) gets
    quotes so the shell keeps it as one word.
    """
    out = []
    for pair in pairs or ():
        if "=" not in pair:
            raise SystemExit(f"--env expects VAR=value, got {pair!r}")
        key, _, value = pair.partition("=")
        if any(ch.isspace() for ch in value):
            value = shlex.quote(value)
        out.append(f"export {key}={value}\n")
    return "".join(out)


def gpu_block(n_gpus, log_dir):
    """Job-script lines that give the job one GPU for its lifetime, or nothing.

    The Slurm on the GPU box has no GRES.
    The job takes the first free device with a lock that lives as long as
    the job's shell (fd 9).
    Two jobs running at once therefore never share a device.
    A ``nvidia-smi`` sampler writes the device memory every 5 s next to
    the job log.
    The run record can then carry the GPU peak alongside the rank-summed
    RSS.
    The sampler closes fd 9 first.
    A child that inherited the lock kept it up to one sleep after the job
    ended.
    The next job then saw both devices busy (c9 jobs 2380/2382, exit 3).
    """
    n = int(n_gpus or 0)
    if n <= 0:
        return ""
    return (
        "# One GPU per job for the job's lifetime (no GRES on this Slurm): first free\n"
        "# device by lock; the lock is fd 9 of this shell and dies with the job.\n"
        f"for _g in $(seq 0 {n - 1}); do\n"
        '  exec 9>"/tmp/linstabpy_gpu${_g}.lock"\n'
        '  if flock -n 9; then export CUDA_VISIBLE_DEVICES="$_g"; break; fi\n'
        "done\n"
        f'[ -n "${{CUDA_VISIBLE_DEVICES:-}}" ] || {{ echo "no free GPU among {n}"; exit 3; }}\n'
        'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"\n'
        "( exec 9>&-; while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu "
        '--format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES"; sleep 5; done '
        f'> "{log_dir}/nvsmi.${{SLURM_JOB_ID:-local}}.log" 2>&1 ) &\n'
        "_NVSMI_PID=$!\n"
        "trap 'kill $_NVSMI_PID 2>/dev/null || true' EXIT\n"
    )


def grid_author200() -> list[tuple[float, float]]:
    return [(b, f) for b in GRID for f in GRID[b]]


def grid_dense317() -> list[tuple[float, float]]:
    if not MAP317_CSV.is_file():
        raise SystemExit(f"missing {MAP317_CSV}")
    pts: list[tuple[float, float]] = []
    with MAP317_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            pts.append((float(row["beta_Xsh"]), float(row["f_Xsh_Uinf"])))
    return pts


GRIDS = {
    "author200": ("Savarino published 200-point map", grid_author200),
    "dense317": ("author 200 + minimal densification (317 points)", grid_dense317),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", required=True, type=Path, help="Directory for generated configs and outputs"
    )
    ap.add_argument(
        "--grid",
        choices=sorted(GRIDS),
        default="author200",
        help="Point grid: author200 (default) or dense317",
    )
    ap.add_argument("--partition", default="LocalQ", help="Slurm partition (default: LocalQ)")
    ap.add_argument("--nranks", type=int, default=4, help="MPI ranks per point (default: 4)")
    ap.add_argument(
        "--dry-run", action="store_true", help="Generate configs/batch scripts but do not submit"
    )
    ap.add_argument(
        "--local",
        action="store_true",
        help="Print the mpiexec command for each point instead of using Slurm",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="VAR=value",
        help="exported into every job script (repeatable), e.g. I_MPI_PIN_DOMAIN=auto",
    )
    ap.add_argument(
        "--ncv",
        type=int,
        default=None,
        help="SLEPc subspace size for the H^H H gain (nev=1); default leaves the solver default",
    )
    ap.add_argument(
        "--beta",
        type=float,
        nargs="+",
        default=None,
        help="only these beta lines of the chosen grid (default: all lines)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=BASE_CONFIG,
        help="base point config; the solver block decides CPU (MUMPS) or GPU "
        "(STRUMPACK or cuDSS under FGMRES) -- see savarino_point_gpu.yaml "
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

    submit = not (args.dry_run or args.local)
    base_config = Path(args.config).resolve()
    cfg0 = load_config(base_config)
    # The job runs from the repository root, but a point config's case_dir is
    # written relative to examples/savarino/ (configs/ and the root YAMLs alike,
    # as savarino_line.py / savarino_mapwalk.py read it). Resolve it here so a
    # point does not depend on the CWD the job starts from.
    case_dir = Path(str(cfg0["paths"]["case_dir"]))
    if not case_dir.is_absolute():
        cfg0["paths"]["case_dir"] = str((HERE / case_dir).resolve())
    template = TEMPLATE.read_text()
    args.out.mkdir(parents=True, exist_ok=True)
    gen_dir = args.out / "configs"
    gen_dir.mkdir(exist_ok=True)

    label, grid_fn = GRIDS[args.grid]
    pts = grid_fn()
    if args.beta is not None:
        wanted = set(args.beta)
        on_grid = {b for b, _ in pts}
        if not wanted <= on_grid:
            raise SystemExit(
                f"--beta {args.beta}: not all on the {args.grid} grid {sorted(on_grid)}"
            )
        pts = [(b, f) for b, f in pts if b in wanted]
    n_lines = len({b for b, _ in pts})
    print(f"{args.grid}: {label}")
    print(f"{len(pts)} points over {n_lines} beta lines; submit={submit} local={args.local}")

    for b, f in pts:
        cfg = copy.deepcopy(cfg0)
        cfg["numerics"]["beta"] = float(b)
        cfg["analysis"]["omega_target"] = {"real": TWO_PI * f, "imag": 0.0}
        cfg["physics"]["sponge"] = {"enabled": False}
        cfg["solver"]["input_region"] = {
            "space": "conservative_momentum",
            "vars": ["u", "v", "w"],
            "x": [X_MIN, x_max(f)],
            "y": [0.0, Y_MAX],
        }
        cfg["solver"]["mask_bc_forcing"] = True
        eigen_cfg = cfg["solver"].setdefault("resolvent", {}).setdefault("eigen", {})
        eigen_cfg["forcing_norm"] = "l2"
        if args.ncv is not None:
            eigen_cfg["ncv"] = int(args.ncv)

        stem = f"sav_map_b{tag(b)}_f{tag(f)}"
        config_path = gen_dir / f"{stem}.yaml"
        config_path.write_text(
            f"# GENERATED by gen_campaign.py -- Savarino point (conservative-momentum "
            f"forcing space) beta={b} f={f}\n" + yaml.safe_dump(cfg, sort_keys=False)
        )
        d = args.out / stem
        d.mkdir(exist_ok=True)
        output_npz = d / f"{stem}.npz"

        if args.local:
            print(
                f"# {stem}\n"
                f"LINSTABPY_CASE_CONFIG={config_path} LINSTABPY_CASE_OUTPUT_NPZ={output_npz} "
                f"mpiexec -n {args.nranks} python -u -c "
                f'"from linstabpy.driver.run import run; run()"'
            )
            continue

        sb = args.out / f"{stem}.sbatch"
        sb.write_text(
            template.format(
                jobname=stem,
                config=config_path,
                output_npz=output_npz,
                workdir=d,
                nranks=args.nranks,
                case_dir=CASE_DIR,
                partition=args.partition,
                repo_root=HERE.parents[1],
                beta=b,
                freq=f,
                env_lines=env_lines(args.env),
                cpus_per_task=args.cpus_per_task,
                gpu_block=gpu_block(args.gpus, d),
            )
        )
        if submit:
            r = subprocess.run(["sbatch", str(sb)], capture_output=True, text=True)
            print(f"{stem}: {r.stdout.strip()}{r.stderr.strip()}", flush=True)
        else:
            print(f"{stem}: (dry run, wrote {sb})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
