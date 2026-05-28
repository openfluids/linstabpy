# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

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
