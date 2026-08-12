# SPDX-License-Identifier: GPL-3.0-only
"""Version source-of-truth checks."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import linstabpy

_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def _pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = _VERSION_RE.search(pyproject.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError("pyproject.toml [project] version line not found")
    return match.group(1)


def test_package_version_surfaces_match_pyproject() -> None:
    expected = _pyproject_version()
    assert linstabpy.__version__ == expected
    imported = Path(linstabpy.__file__).resolve()
    if "site-packages" in imported.parts:
        assert metadata.version("linstabpy") == expected
