# SPDX-License-Identifier: GPL-3.0-only
"""Version source-of-truth checks."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import pytest

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


def test_package_dir_lists_lazy_exports() -> None:
    names = dir(linstabpy)
    assert "wavemaker" in names
    assert "BaseFlow" in names
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(linstabpy, "not_a_public_name")


def test_package_version_falls_back_when_pyproject_is_absent(monkeypatch) -> None:
    import linstabpy as pkg

    monkeypatch.setattr(pkg, "_PYPROJECT", Path("/no/such/linstabpy-pyproject.toml"))
    assert pkg._package_version() == metadata.version("linstabpy")

    def _missing(_name: str) -> str:
        raise metadata.PackageNotFoundError("linstabpy")

    monkeypatch.setattr(pkg.metadata, "version", _missing)
    assert pkg._package_version() == "0.0.0+dev"


def test_stability_and_bc_packages_reject_unknown_names() -> None:
    import linstabpy.boundary_conditions as bc_pkg
    import linstabpy.stability as stab_pkg

    assert "eigenvalues" in stab_pkg.__dir__()
    assert "apply_dirichlet" in bc_pkg.__dir__()
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(stab_pkg, "not_exported")
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(bc_pkg, "not_exported")
