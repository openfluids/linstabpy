# SPDX-License-Identifier: GPL-3.0-only
"""Smoke tests for the assembly metrics diff CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "martini2024" / "tools" / "assembly_metrics_diff.py"


def _write_metrics(
    path: Path, *, wall: float, coeff_bytes: int, stage_seconds: float, stage_rss: int
) -> None:
    payload = {
        "nnz": 100,
        "ndof": 50,
        "wall_seconds": wall,
        "coeff_tensor_bytes_total": coeff_bytes,
        "stages": {
            "prepare_coefficients": {
                "seconds": stage_seconds,
                "rss_peak_delta_bytes": stage_rss,
                "nnz": 0,
                "skipped": False,
            },
            "apply_bc": {
                "seconds": 0.0,
                "rss_peak_delta_bytes": 0,
                "nnz": 0,
                "skipped": True,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_assembly_metrics_diff_cli_emits_json_report(tmp_path: Path):
    """The diff CLI emits a structured JSON report with regression findings."""
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_metrics(baseline, wall=1.0, coeff_bytes=1000, stage_seconds=0.4, stage_rss=200)
    _write_metrics(candidate, wall=1.3, coeff_bytes=1250, stage_seconds=0.5, stage_rss=260)

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(baseline),
            str(candidate),
            "--json",
            "--time-threshold",
            "0.2",
            "--memory-threshold",
            "0.2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(run.stdout)
    assert abs(report["summary"]["wall_seconds"]["delta_ratio"] - 0.3) < 1e-12
    assert abs(report["summary"]["coeff_tensor_bytes_total"]["delta_ratio"] - 0.25) < 1e-12
    assert {item["metric"] for item in report["regressions"]} >= {
        "wall_seconds",
        "coeff_tensor_bytes_total",
        "seconds",
        "rss_peak_delta_bytes",
    }


def test_assembly_metrics_diff_cli_fail_if_regression(tmp_path: Path):
    """The diff CLI returns non-zero when regressions exceed the configured thresholds."""
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_metrics(baseline, wall=1.0, coeff_bytes=1000, stage_seconds=0.4, stage_rss=200)
    _write_metrics(candidate, wall=1.05, coeff_bytes=1010, stage_seconds=0.41, stage_rss=205)

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(baseline),
            str(candidate),
            "--fail-if-regression",
            "--time-threshold",
            "0.01",
            "--memory-threshold",
            "0.01",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 1
    assert "regressions:" in run.stdout
