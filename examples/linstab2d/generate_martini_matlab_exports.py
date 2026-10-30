#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run Martini MATLAB parity exports from examples/linstab2d/linstab2D_scripts/."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.linstab2d.extract_martini_to_npz import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_INPUT_DIR,
    expected_export_stems,
)

DEFAULT_SCRIPTS_DIR = ROOT / "examples" / "linstab2d" / "linstab2D_scripts"
DEFAULT_RUNNER = DEFAULT_SCRIPTS_DIR / "run_martini_exports.sh"

EXAMPLE_ONLY_MAP = {
    "example0": "example_0",
    "example1": "example_1",
    "example2": "example_2",
    "example3_temporal": "example_3",
    "example3_spatial": "example_3",
    "example4": "example_4",
    "example5": "example_5",
}


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", default="matlab", help="MATLAB executable (via MATLAB_BIN).")
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=DEFAULT_SCRIPTS_DIR,
        help="Directory containing linstab2D_scripts drivers.",
    )
    parser.add_argument(
        "--linstab2d-root",
        type=Path,
        default=None,
        help="LinStab2D checkout (aux_matlab + BaseFlows read scripts).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory for raw MATLAB .mat exports.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--only",
        choices=tuple(EXAMPLE_ONLY_MAP),
        action="append",
        help="Run one example block (maps to run_martini_exports.sh --only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when output .mat files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the shell command without invoking MATLAB.",
    )
    args = parser.parse_args()

    scripts_dir = args.scripts_dir.resolve()
    runner = scripts_dir / "run_martini_exports.sh"
    if not runner.is_file():
        raise FileNotFoundError(f"Missing runner: {_display_path(runner)}")

    linstab2d_root = args.linstab2d_root
    if linstab2d_root is None:
        env_root = os.environ.get("LINSTAB2D_ROOT", "")
        linstab2d_root = Path(env_root) if env_root else None
    if linstab2d_root is None or not linstab2d_root.is_dir():
        raise FileNotFoundError("Set --linstab2d-root or LINSTAB2D_ROOT to a LinStab2D checkout.")

    output_dir = args.output_dir.resolve()
    command = ["bash", str(runner)]
    if args.force:
        command.append("--force")
    if args.only:
        seen: set[str] = set()
        for case_id in args.only:
            key = EXAMPLE_ONLY_MAP[case_id]
            if key in seen:
                continue
            command.extend(["--only", key])
            seen.add(key)

    env = os.environ.copy()
    env["LINSTAB2D_ROOT"] = str(linstab2d_root.resolve())
    env["MARTINI2024_OUTPUT_ROOT"] = str(output_dir)
    env["MATLAB_BIN"] = args.matlab

    expected = [f"{stem}.mat" for stem in expected_export_stems(args.config.resolve())]
    summary = {
        "runner": _display_path(runner),
        "scripts_dir": _display_path(scripts_dir),
        "linstab2d_root": str(linstab2d_root.resolve()),
        "output_dir": _display_path(output_dir),
        "selected_only": args.only or "all",
        "expected_exports": expected,
        "command": " ".join(shlex.quote(part) for part in command),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.dry_run:
        return 0

    completed = subprocess.run(command, check=False, env=env, cwd=str(scripts_dir))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
