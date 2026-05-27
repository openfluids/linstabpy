#!/usr/bin/env python3
"""Run or list validation gates for the architecture refactor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT
    / "artifacts"
    / "architecture_refactor"
    / "architecture_refactor_phase0"
    / "matrix_run_latest.json"
)


@dataclass(frozen=True)
class Check:
    id: str
    group: str
    description: str
    command: tuple[str, ...]
    artifact: Path | None = None
    long_running: bool = False


def _python() -> str:
    override = os.environ.get("LINSTABPY_PYTHON") or os.environ.get("LINSTABPY_PYTHON_BIN")
    if override:
        return override
    return str(ROOT / ".venv" / "bin" / "python")


def _checks() -> list[Check]:
    py = _python()
    phase0_dir = ROOT / "artifacts/architecture_refactor/architecture_refactor_phase0"
    return [
        Check(
            id="pytest-full",
            group="phase0-local",
            description="Full pytest suite.",
            command=(py, "-m", "pytest", "tests/", "-q"),
            long_running=True,
        ),
        Check(
            id="pytest-focused",
            group="phase0-local",
            description="Focused tests for BCs, operators, solvers, and acceptance logic.",
            command=(
                py,
                "-m",
                "pytest",
                "tests/test_boundary_conditions.py",
                "tests/test_operators.py",
                "tests/test_solvers.py",
                "tests/test_stability_acceptance.py",
                "-q",
            ),
        ),
        Check(
            id="example-blasius-smoke",
            group="phase0-local",
            description="Fresh reduced Blasius example run without MATLAB parity.",
            command=("bash", "run_all_examples.sh", "--example", "1s", "--skip-parity", "--fresh"),
            artifact=ROOT / "examples/example_1_blasius_smoke.npz",
        ),
        Check(
            id="parity-matrix-status",
            group="phase0-local",
            description="Current Martini operator/branch parity matrix status.",
            command=(
                py,
                "scripts/validation/martini2024/tools/check_parity_matrices.py",
                "--allow-incomplete",
                "--artifacts-dir",
                str(phase0_dir.relative_to(ROOT)),
            ),
            artifact=phase0_dir / "parity_gate_summary.json",
        ),
        Check(
            id="operator-perf-memory",
            group="phase0-local",
            description="Small PETSc/SLEPc operator performance and memory baseline.",
            command=(
                py,
                "scripts/validation/martini2024/tools/perf_memory_operator_parity.py",
                "--meshes",
                "24x16,32x24",
                "--output",
                str((phase0_dir / "operator_perf_memory.json").relative_to(ROOT)),
            ),
            artifact=phase0_dir / "operator_perf_memory.json",
        ),
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def _select(checks: Iterable[Check], *, group: str | None, ids: set[str]) -> list[Check]:
    selected = []
    for check in checks:
        if ids and check.id not in ids:
            continue
        if group is not None and check.group != group:
            continue
        selected.append(check)
    return selected


def _run_check(check: Check, *, dry_run: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": check.id,
        "group": check.group,
        "description": check.description,
        "command": list(check.command),
        "artifact": str(check.artifact.relative_to(ROOT)) if check.artifact else None,
        "started_at_utc": _now(),
        "dry_run": dry_run,
    }
    if dry_run:
        payload.update({"returncode": None, "status": "dry-run"})
        return payload
    result = subprocess.run(
        check.command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    payload.update(
        {
            "finished_at_utc": _now(),
            "returncode": result.returncode,
            "status": "pass" if result.returncode == 0 else "fail",
            "output_tail": result.stdout.splitlines()[-80:],
            "artifact_exists": bool(check.artifact and check.artifact.exists()),
        }
    )
    return payload


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List checks and exit.")
    parser.add_argument("--group", help="Run only checks in this group.")
    parser.add_argument(
        "--check", action="append", default=[], help="Run one check ID. Repeatable."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Write the report without running commands."
    )
    parser.add_argument("--skip-long", action="store_true", help="Skip long-running checks.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    checks = _checks()
    if args.list:
        for check in checks:
            marker = " long" if check.long_running else ""
            print(f"{check.id:24s} {check.group:14s}{marker:6s} {' '.join(check.command)}")
        return 0

    selected = _select(checks, group=args.group, ids=set(args.check))
    if not selected:
        raise SystemExit("No checks selected.")
    if args.skip_long:
        selected = [check for check in selected if not check.long_running]
    if not selected:
        raise SystemExit("Only long-running checks were selected; omit --skip-long to run them.")

    results = [_run_check(check, dry_run=args.dry_run) for check in selected]
    report = {
        "generated_at_utc": _now(),
        "git_sha": _git_sha(),
        "root": str(ROOT),
        "selected_group": args.group,
        "selected_ids": args.check,
        "skip_long": args.skip_long,
        "status": "pass"
        if all(item["status"] in {"pass", "dry-run"} for item in results)
        else "fail",
        "checks": results,
    }
    _write_report(args.report, report)
    print(f"Architecture refactor matrix: status={report['status']} report={args.report}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
