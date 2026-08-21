#!/usr/bin/env python3
"""Run the two resolvent points used by ``compare_modes.py``.

Write local outputs under ``output/compare/`` (gitignored).

  output/compare/mack/sav_map_b45p0_f2.npz.
  output/compare/liftup/sav_map_b163p0_f0p0001.npz.

Usage (from this directory, with linstabpy installed and MPI available):

    python run_compare_points.py
    python run_compare_points.py --nranks 4
    python run_compare_points.py --mack-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reference_paths import BASEFLOW_NPY_DIR, LIFTUP_OUTPUT_NPZ, MACK_OUTPUT_NPZ

CONFIGS = HERE / "configs"
POINTS = (
    ("mack", CONFIGS / "mack_f2_b45.yaml", MACK_OUTPUT_NPZ),
    ("liftup", CONFIGS / "liftup_f1em4_b163.yaml", LIFTUP_OUTPUT_NPZ),
)


def _require_baseflow() -> None:
    marker = BASEFLOW_NPY_DIR / "rho.npy"
    if not marker.is_file():
        raise SystemExit(
            f"missing {marker}\nFetch and convert the base flow first:\n"
            "  python fetch_zenodo.py baseflow && python savarino_to_npy.py"
        )


def _run_point(config: Path, out_npz: Path, *, nranks: int) -> None:
    if not config.is_file():
        raise SystemExit(f"missing config: {config}")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LINSTABPY_CASE1_CONFIG"] = str(config.resolve())
    env["LINSTABPY_CASE1_OUTPUT_NPZ"] = str(out_npz.resolve())
    code = "from linstabpy.driver.run import run; run()"
    if nranks > 1:
        cmd = ["mpiexec", "-n", str(nranks), sys.executable, "-u", "-c", code]
    else:
        cmd = [sys.executable, "-u", "-c", code]
    print(f"running {config.name} -> {out_npz}", flush=True)
    subprocess.run(cmd, env=env, cwd=HERE, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--nranks", type=int, default=4, help="MPI ranks per point (default: 4)")
    ap.add_argument("--mack-only", action="store_true", help="run only the (f,beta)=(2,45) point")
    ap.add_argument(
        "--liftup-only", action="store_true", help="run only the (f,beta)=(1e-4,163) point"
    )
    args = ap.parse_args()

    _require_baseflow()

    selected = POINTS
    if args.mack_only:
        selected = (POINTS[0],)
    elif args.liftup_only:
        selected = (POINTS[1],)

    for name, config, out_npz in selected:
        _run_point(config, out_npz, nranks=args.nranks)
        print(f"done {name}: {out_npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
