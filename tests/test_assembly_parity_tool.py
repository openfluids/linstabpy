# SPDX-License-Identifier: GPL-3.0-only
"""Tests for the assembly parity CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "martini" / "assembly_parity.py"


@pytest.mark.parametrize("model", ["2D", "axy"])
def test_assembly_parity_cli_pair_and_artifacts(model: str, tmp_path: Path):
    """The parity CLI succeeds for live backend pairs and saved artifacts."""
    before = tmp_path / f"before_{model}.npz"
    after = tmp_path / f"after_{model}.npz"
    pair_output = tmp_path / f"pair_{model}.json"
    artifact_output = tmp_path / f"artifacts_{model}.json"

    pair_run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "pair",
            "--model",
            model,
            "--backend-before",
            "scipy",
            "--backend-after",
            "petsc",
            "--skip-eigs",
            "--artifact-before",
            str(before),
            "--artifact-after",
            str(after),
            "--output",
            str(pair_output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "pass=True" in pair_run.stdout
    assert before.exists()
    assert after.exists()
    pair_report = json.loads(pair_output.read_text(encoding="utf-8"))
    assert pair_report["mode"] == "pair"
    assert pair_report["pass"] is True
    assert pair_report["sparse"]["pass"] is True
    assert pair_report["sparse"]["structure_match"] is True
    assert pair_report["block_nnz"]["available"] is True
    assert pair_report["block_nnz"]["match"] is True
    assert pair_report["eigen_sanity"]["status"] == "skipped"

    artifact_run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "artifacts",
            str(before),
            str(after),
            "--skip-eigs",
            "--output",
            str(artifact_output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "pass=True" in artifact_run.stdout
    artifact_report = json.loads(artifact_output.read_text(encoding="utf-8"))
    assert artifact_report["mode"] == "artifacts"
    assert artifact_report["pass"] is True
    assert artifact_report["sparse"]["pass"] is True
    assert artifact_report["sparse"]["structure_match"] is True
    assert artifact_report["block_nnz"]["available"] is True
    assert artifact_report["block_nnz"]["match"] is True
    assert artifact_report["eigen_sanity"]["status"] == "skipped"
    assert artifact_report["before_metadata"]["artifact_path"] == str(before.resolve())
    assert artifact_report["after_metadata"]["artifact_path"] == str(after.resolve())


def test_assembly_parity_cli_runs_default_eigen_sanity(tmp_path: Path):
    """The default pair mode executes the eigenvalue sanity check on the small harness case."""
    output = tmp_path / "pair_default_eigs.json"

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "pair",
            "--model",
            "2D",
            "--backend-before",
            "scipy",
            "--backend-after",
            "petsc",
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
    assert report["mode"] == "pair"
    assert report["pass"] is True
    assert report["eigen_sanity"]["status"] == "ok"
    assert report["eigen_sanity"]["pass"] is True
