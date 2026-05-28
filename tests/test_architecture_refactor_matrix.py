# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from scripts.validation import architecture_refactor_matrix


def test_phase0_golden_local_selects_local_and_documentation_checks() -> None:
    checks = architecture_refactor_matrix._select(
        architecture_refactor_matrix._checks(),
        group="phase0-golden-local",
        ids=set(),
    )

    groups = {check.group for check in checks}
    ids = {check.id for check in checks}

    assert groups == {"phase0-local", "phase0-expanded-local", "documentation-gates"}
    assert "martini-fixture-preflight" not in ids
    assert "martini-full-examples" not in ids
    assert "docstring-inventory" in ids


def test_phase0_golden_all_keeps_external_asset_checks_explicit() -> None:
    checks = architecture_refactor_matrix._select(
        architecture_refactor_matrix._checks(),
        group="phase0-golden-all",
        ids=set(),
    )

    ids = {check.id for check in checks}

    assert "martini-fixture-preflight" in ids
    assert "martini-full-examples" in ids
    assert "coefficient-tensor-snapshot" in ids


def test_explicit_check_filter_still_applies_inside_virtual_group() -> None:
    checks = architecture_refactor_matrix._select(
        architecture_refactor_matrix._checks(),
        group="phase0-golden-all",
        ids={"docstring-inventory", "martini-fixture-preflight"},
    )

    assert [check.id for check in checks] == [
        "martini-fixture-preflight",
        "docstring-inventory",
    ]


def test_run_check_reports_timeout() -> None:
    check = architecture_refactor_matrix.Check(
        id="timeout-demo",
        group="test",
        description="Slow command for timeout handling.",
        command=(sys.executable, "-c", "import time; time.sleep(2)"),
        timeout_seconds=0.1,
    )

    result = architecture_refactor_matrix._run_check(check, dry_run=False)

    assert result["status"] == "timeout"
    assert result["returncode"] is None
    assert result["timeout_seconds"] == 0.1
    assert "Timed out after 0.1 seconds." in result["output_tail"]


def test_artifact_inventory_hashes_file(tmp_path: Path) -> None:
    artifact = tmp_path / "baseline.json"
    artifact.write_text('{"status": "pass"}\n', encoding="utf-8")

    inventory = architecture_refactor_matrix._artifact_inventory(artifact)

    assert inventory == {
        "kind": "file",
        "path": str(artifact),
        "exists": True,
        "size_bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def test_artifact_inventory_hashes_directory_deterministically(tmp_path: Path) -> None:
    artifact = tmp_path / "fixtures"
    artifact.mkdir()
    (artifact / "b.txt").write_text("second\n", encoding="utf-8")
    (artifact / "a.txt").write_text("first\n", encoding="utf-8")

    first = architecture_refactor_matrix._artifact_inventory(artifact)
    second = architecture_refactor_matrix._artifact_inventory(artifact)

    assert first == second
    assert first is not None
    assert first["kind"] == "directory"
    assert first["file_count"] == 2
    assert first["total_size_bytes"] == 13
    assert isinstance(first["sha256"], str)
    assert len(first["sha256"]) == 64


def test_run_check_includes_artifact_inventory(tmp_path: Path) -> None:
    artifact = tmp_path / "result.txt"
    check = architecture_refactor_matrix.Check(
        id="artifact-demo",
        group="test",
        description="Command that writes an artifact.",
        command=(
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(artifact)!r}).write_text('ok\\n')",
        ),
        artifact=artifact,
    )

    result = architecture_refactor_matrix._run_check(check, dry_run=False)

    assert result["status"] == "pass"
    assert result["artifact_exists"]
    assert result["artifact_inventory"] == {
        "kind": "file",
        "path": str(artifact),
        "exists": True,
        "size_bytes": 3,
        "sha256": hashlib.sha256(b"ok\n").hexdigest(),
    }


def test_run_check_keeps_report_when_artifact_hashing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "result.txt"
    artifact.write_text("ok\n", encoding="utf-8")

    def fail_hash(path: Path, deadline: float | None = None) -> tuple[str, int]:
        raise OSError(f"cannot read {path.name}")

    monkeypatch.setattr(architecture_refactor_matrix, "_hash_file", fail_hash)
    check = architecture_refactor_matrix.Check(
        id="artifact-error-demo",
        group="test",
        description="Command with an unreadable artifact.",
        command=(sys.executable, "-c", "pass"),
        artifact=artifact,
    )

    result = architecture_refactor_matrix._run_check(check, dry_run=False)

    assert result["status"] == "pass"
    assert result["artifact_exists"]
    assert result["artifact_inventory"] == {
        "kind": "error",
        "path": str(artifact),
        "exists": True,
        "error_type": "OSError",
        "error": "cannot read result.txt",
    }


def test_artifact_inventory_timeout_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "result.txt"
    artifact.write_text("ok\n", encoding="utf-8")

    def fail_deadline(deadline: float | None) -> None:
        raise TimeoutError("artifact inventory timed out")

    monkeypatch.setattr(architecture_refactor_matrix, "_check_inventory_deadline", fail_deadline)

    inventory = architecture_refactor_matrix._artifact_inventory(artifact)

    assert inventory == {
        "kind": "error",
        "path": str(artifact),
        "exists": True,
        "error_type": "TimeoutError",
        "error": "artifact inventory timed out",
    }


def test_long_checks_get_longer_default_timeout() -> None:
    check = architecture_refactor_matrix.Check(
        id="long-demo",
        group="test",
        description="Long-running check.",
        command=(sys.executable, "-c", "pass"),
        long_running=True,
    )

    assert (
        architecture_refactor_matrix._timeout_seconds(check)
        == architecture_refactor_matrix.LONG_CHECK_TIMEOUT_SECONDS
    )
