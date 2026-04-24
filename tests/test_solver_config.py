"""Tests for solver configuration helpers that do not require PETSc."""

from __future__ import annotations

from linstab2d.utils._solver_cfg import _mumps_petsc_options


def test_mumps_petsc_options_enable_hardened_defaults() -> None:
    assert _mumps_petsc_options(None) == {
        "mat_mumps_icntl_14": "100",
        "mat_mumps_icntl_24": "1",
    }


def test_mumps_petsc_options_preserve_existing_icntl7_gate() -> None:
    assert _mumps_petsc_options({"mumps_icntl_7": True}) == {
        "mat_mumps_icntl_7": "5",
        "mat_mumps_icntl_14": "100",
        "mat_mumps_icntl_24": "1",
    }


def test_mumps_petsc_options_allow_jsonc_override_and_opt_out() -> None:
    assert _mumps_petsc_options({"mumps_icntl_14": 250, "mumps_icntl_24": None}) == {
        "mat_mumps_icntl_14": "250",
    }


def test_mumps_petsc_options_allow_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("LINSTAB2D_PETSC_MUMPS_ICNTL_14", "300")
    monkeypatch.setenv("LINSTAB2D_PETSC_MUMPS_ICNTL_24", "none")

    assert _mumps_petsc_options(None) == {
        "mat_mumps_icntl_14": "300",
    }
