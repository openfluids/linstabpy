# SPDX-License-Identifier: GPL-3.0-only
"""Check environment.yml follows pyproject and keeps PETSc off pip."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "environment.yml"
PYPROJECT_PATH = ROOT / "pyproject.toml"

PIP_FORBIDDEN = frozenset(
    {
        "petsc",
        "petsc4py",
        "slepc",
        "slepc4py",
        "mpi4py",
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
        "PyYAML",
    }
)


def _env() -> dict:
    return yaml.safe_load(ENV_PATH.read_text(encoding="utf-8"))


def _pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _conda_name(spec: str) -> str:
    return spec.replace("<", "=").replace(">", "=").replace("!", "=").split("=")[0].strip()


def _conda_names(deps: list) -> list[str]:
    names: list[str] = []
    for item in deps:
        if isinstance(item, str):
            names.append(_conda_name(item))
        elif isinstance(item, dict) and "pip" in item:
            continue
        else:
            raise AssertionError(f"unexpected environment.yml entry: {item!r}")
    return names


def _pip_items(deps: list) -> list[str]:
    for item in deps:
        if isinstance(item, dict) and "pip" in item:
            return [str(entry) for entry in item["pip"]]
    raise AssertionError("environment.yml has no pip section")


def test_environment_yml_names_the_package() -> None:
    data = _env()
    assert data["name"] == "linstabpy"
    assert "conda-forge" in data["channels"]


def test_environment_yml_python_lower_bound_matches_pyproject() -> None:
    requires = str(_pyproject()["project"]["requires-python"])
    assert requires.startswith(">=")
    lower = requires.removeprefix(">=").strip()
    python_specs = [
        item
        for item in _env()["dependencies"]
        if isinstance(item, str) and item.startswith("python")
    ]
    assert python_specs, "environment.yml must pin Python"
    assert lower in python_specs[0]


def test_environment_yml_conda_covers_pyproject_runtime() -> None:
    names = _conda_names(_env()["dependencies"])
    for required in (
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
        "mpi4py",
        "petsc",
        "slepc",
        "petsc4py",
        "slepc4py",
    ):
        assert required in names, required


def test_environment_yml_petsc_build_is_complex() -> None:
    petsc_specs = [
        item
        for item in _env()["dependencies"]
        if isinstance(item, str) and item.split("=")[0] == "petsc"
    ]
    assert petsc_specs, "environment.yml must list petsc"
    assert "complex" in petsc_specs[0]


def test_environment_yml_pip_installs_this_package_without_deps() -> None:
    pip_items = _pip_items(_env()["dependencies"])
    assert "--no-deps" in pip_items
    assert any(item.startswith("linstabpy>=") for item in pip_items), pip_items
    for item in pip_items:
        if item.startswith("-"):
            continue
        name = re.split(r"[\[=<>!~ ]", item, maxsplit=1)[0]
        assert name not in PIP_FORBIDDEN, item
        assert not name.endswith("4py"), item
