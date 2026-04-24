"""Canonical repository paths for tests and validation fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_ROOT = REPO_ROOT / "studies"
VALIDATION_ROOT = REPO_ROOT / "validation"
OPERATOR_FIXTURE_ROOT = VALIDATION_ROOT / "operator_fixtures"
MARTINI_FIXTURE_ROOT = VALIDATION_ROOT / "martini2024" / "fixtures"


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def require_existing(path: Path, description: str) -> Path:
    if not path.exists():
        pytest.skip(f"Missing {description}: {path}", allow_module_level=True)
    return path


def case_fixture(name: str) -> Path:
    """Return the canonical or legacy root for a named study fixture."""
    if name not in {"case1", "case2"}:
        raise ValueError(f"Unknown study fixture: {name}")
    return first_existing(
        STUDIES_ROOT / name,
        REPO_ROOT / "results" / name,
        REPO_ROOT / name,
    )


def martini_fixture(filename: str) -> Path:
    """Return the canonical or legacy Martini parity fixture path."""
    return first_existing(
        MARTINI_FIXTURE_ROOT / filename,
        REPO_ROOT / "tools" / "parity" / "npz" / filename,
    )


def martini_export(filename: str) -> Path:
    """Return the optional raw Martini MATLAB export path."""
    return REPO_ROOT / "martini_examples" / filename
