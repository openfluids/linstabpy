# SPDX-License-Identifier: GPL-3.0-only
"""Smoke tests for the LNSE equivalence harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "lnse_equivalence_harness.py"


def test_lnse_equivalence_harness_current_path_contract(tmp_path: Path) -> None:
    """The quick baseline-vs-candidate contract is zero-drift on current assembly."""
    output = tmp_path / "lnse_equivalence.json"

    run = subprocess.run(
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
    )

    assert "pass=True" in run.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "current-vs-current"
    assert report["pass"] is True
    assert [case["case"] for case in report["cases"]] == ["2D", "axy"]

    for case in report["cases"]:
        assert case["coefficients"]["pass"] is True
        assert case["raw_operator"]["pass"] is True
        assert case["raw_operator"]["structure_match"] is True
        assert case["pre_bc_operator"]["pass"] is True
        assert case["pre_bc_operator"]["structure_match"] is True
        assert case["post_bc_operator"]["pass"] is True
        assert case["post_bc_operator"]["structure_match"] is True
        assert case["bc_indices"]["pass"] is True
        assert case["bc_indices"]["first_mismatches"] == {"before": [], "after": []}
        assert case["eigen_sanity"]["status"] == "skipped"


def test_lnse_equivalence_harness_runs_eigen_sanity(tmp_path: Path) -> None:
    """The full harness includes a small downstream eigenvalue comparison."""
    output = tmp_path / "lnse_equivalence_eigs.json"

    subprocess.run(
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
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["cases"][0]["eigen_sanity"]["status"] == "ok"
    assert report["cases"][0]["eigen_sanity"]["pass"] is True
