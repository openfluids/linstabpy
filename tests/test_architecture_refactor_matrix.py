# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import sys

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
