# SPDX-License-Identifier: GPL-3.0-only
"""Canonical repository paths for tests and validation fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_ROOT = REPO_ROOT / "studies"
VALIDATION_ROOT = REPO_ROOT / "validation"
OPERATOR_FIXTURE_ROOT = VALIDATION_ROOT / "operator_fixtures"
MARTINI_FIXTURE_ROOT = VALIDATION_ROOT / "martini2024" / "fixtures"


def require_existing(path: Path, description: str) -> Path:
    if not path.exists():
        pytest.skip(f"Missing {description}: {path}", allow_module_level=True)
    return path


def case_fixture(name: str) -> Path:
    """Return the canonical root for a named study fixture."""
    if name not in {"case1", "case2"}:
        raise ValueError(f"Unknown study fixture: {name}")
    return STUDIES_ROOT / name


def martini_fixture(filename: str) -> Path:
    """Return the canonical Martini parity fixture path."""
    return MARTINI_FIXTURE_ROOT / filename


def martini_export(filename: str) -> Path:
    """Return the optional raw Martini MATLAB export path."""
    return REPO_ROOT / "martini_examples" / filename
