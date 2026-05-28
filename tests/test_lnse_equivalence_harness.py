# SPDX-License-Identifier: GPL-3.0-only
"""Smoke tests for the LNSE equivalence harness."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - tests execute the checked-in harness script.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "lnse_equivalence_harness.py"
HARNESS_TIMEOUT_SECONDS = 180


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require_true(value: object, message: str) -> None:
    if not isinstance(value, bool) or not value:
        raise AssertionError(message)


def _load_report(path: Path) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Invalid JSON report at {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise AssertionError(f"Expected JSON object report at {path}, got {type(report).__name__}")
    return report


def test_lnse_equivalence_harness_current_path_contract(tmp_path: Path) -> None:
    """The quick baseline-vs-candidate contract is zero-drift on current assembly."""
    output = tmp_path / "lnse_equivalence.json"

    run = subprocess.run(  # nosec B603 - fixed local script path and bounded timeout.
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            "2D",
            "--case",
            "axy",
            "--skip-eigs",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=HARNESS_TIMEOUT_SECONDS,
    )

    _require("pass=True" in run.stdout, run.stdout)
    report = _load_report(output)
    _require(report["mode"] == "current-vs-current", str(report["mode"]))
    _require_true(report["pass"], str(report["pass"]))
    _require([case["case"] for case in report["cases"]] == ["2D", "axy"], str(report["cases"]))

    for case in report["cases"]:
        _require_true(case["coefficients"]["pass"], str(case["coefficients"]))
        _require_true(case["raw_operator"]["pass"], str(case["raw_operator"]))
        _require_true(case["raw_operator"]["structure_match"], str(case["raw_operator"]))
        _require_true(case["pre_bc_operator"]["pass"], str(case["pre_bc_operator"]))
        _require_true(case["pre_bc_operator"]["structure_match"], str(case["pre_bc_operator"]))
        _require_true(case["post_bc_operator"]["pass"], str(case["post_bc_operator"]))
        _require_true(case["post_bc_operator"]["structure_match"], str(case["post_bc_operator"]))
        _require_true(case["bc_indices"]["pass"], str(case["bc_indices"]))
        _require(
            case["bc_indices"]["first_mismatches"] == {"before": [], "after": []},
            str(case["bc_indices"]),
        )
        _require(case["eigen_sanity"]["status"] == "skipped", str(case["eigen_sanity"]))


def test_lnse_equivalence_harness_runs_eigen_sanity(tmp_path: Path) -> None:
    """The full harness includes a small downstream eigenvalue comparison."""
    output = tmp_path / "lnse_equivalence_eigs.json"

    subprocess.run(  # nosec B603 - fixed local script path and bounded timeout.
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            "2D",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=HARNESS_TIMEOUT_SECONDS,
    )

    report = _load_report(output)
    _require_true(report["pass"], str(report))
    _require(report["cases"][0]["eigen_sanity"]["status"] == "ok", str(report["cases"][0]))
    _require_true(report["cases"][0]["eigen_sanity"]["pass"], str(report["cases"][0]))


def test_lnse_equivalence_harness_can_be_enabled_by_env(tmp_path: Path) -> None:
    """Migration runs can enable the dual-path gate without code changes."""
    output = tmp_path / "lnse_dual_path_env.json"

    run = subprocess.run(  # nosec B603 - fixed local script path and bounded timeout.
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            "2D",
            "--skip-eigs",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "LINSTABPY_DUAL_PATH": "1",
            "LINSTABPY_DUAL_PATH_REPORT": str(output),
        },
        timeout=HARNESS_TIMEOUT_SECONDS,
    )

    _require("dual_path_enabled=True" in run.stdout, run.stdout)
    report = _load_report(output)
    _require(report["mode"] == "dual-path-current-vs-current", str(report["mode"]))
    _require_true(report["dual_path_enabled"], str(report["dual_path_enabled"]))
    _require(report["enabled_by"] == "LINSTABPY_DUAL_PATH", str(report["enabled_by"]))
    _require_true(report["pass"], str(report))
