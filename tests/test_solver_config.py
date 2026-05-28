# SPDX-License-Identifier: GPL-3.0-only
"""Tests for solver configuration helpers that do not require PETSc."""

from __future__ import annotations

import subprocess
import sys

import pytest

from linstabpy.utils import petsc_core
from linstabpy.utils._solver_cfg import SolverConfig, _mumps_petsc_options, _scfg, _scfg_bool


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
    monkeypatch.setenv("LINSTABPY_PETSC_MUMPS_ICNTL_14", "300")
    monkeypatch.setenv("LINSTABPY_PETSC_MUMPS_ICNTL_24", "none")

    assert _mumps_petsc_options(None) == {
        "mat_mumps_icntl_14": "300",
    }


def test_solver_config_dict_overrides_environment(monkeypatch) -> None:
    """JSONC solver values have precedence over environment fallbacks."""
    monkeypatch.setenv("LINSTABPY_EXAMPLE_KEY", "env-value")

    assert _scfg({"mode": "LOCAL"}, "mode", "LINSTABPY_EXAMPLE_KEY", "auto") == "local"
    assert _scfg(None, "mode", "LINSTABPY_EXAMPLE_KEY", "auto") == "env-value"
    assert _scfg(None, "mode", "LINSTABPY_MISSING_KEY", "AUTO") == "auto"


def test_solver_config_bool_resolution(monkeypatch) -> None:
    """Boolean config helpers preserve the current dict -> env -> default order."""
    monkeypatch.setenv("LINSTABPY_BOOL_KEY", "yes")

    assert _scfg_bool({"root_scatter": False}, "root_scatter", "LINSTABPY_BOOL_KEY", True) is False
    assert _scfg_bool(None, "root_scatter", "LINSTABPY_BOOL_KEY", False) is True
    assert _scfg_bool(None, "root_scatter", "LINSTABPY_MISSING_BOOL", False) is False


def test_solver_config_wrapper_preserves_mapping_behavior() -> None:
    raw = {"petsc_matrix_mode": "local", "eigen": {"eps_tol": 1e-9}}
    cfg = SolverConfig.from_config(raw)

    assert isinstance(cfg, SolverConfig)
    assert SolverConfig.from_config(cfg) is cfg
    assert cfg["petsc_matrix_mode"] == "local"
    assert cfg.get("eigen") == {"eps_tol": 1e-9}
    assert dict(cfg) == raw
    assert cfg.setting("petsc_matrix_mode", "LINSTABPY_PETSC_MATRIX_MODE", "auto") == "local"


def test_solver_config_wrapper_resolves_existing_subsection_policy() -> None:
    cfg = SolverConfig(
        {
            "linsolve": {"type": "mumps"},
            "resolvent": {"type": "iterative"},
            "eigen": {"eps_tol": 1e-9},
        }
    )

    assert cfg.subsection("LINSTABPY_EIGEN") == {"eps_tol": 1e-9}
    assert cfg.subsection("LINSTABPY_RESOLVENT") == {"type": "iterative"}
    assert cfg.subsection("LINSTABPY_LINSOLVE") == {"type": "mumps"}
    assert SolverConfig({"linsolve": {"type": "mumps"}}).subsection("LINSTABPY_RESOLVENT") == {
        "type": "mumps"
    }


def test_petsc_matrix_mode_accepts_current_aliases(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_PETSC_MATRIX_MODE", "parallel")

    assert petsc_core._petsc_matrix_mode(None) == "parallel"
    assert petsc_core._petsc_matrix_mode({"petsc_matrix_mode": "SERIAL"}) == "serial"
    assert petsc_core._petsc_matrix_mode({"petsc_matrix_mode": "distributed"}) == "distributed"


def test_petsc_matrix_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="petsc_matrix_mode"):
        petsc_core._petsc_matrix_mode({"petsc_matrix_mode": "cluster"})


def test_explicit_petsc_matrix_modes_do_not_query_petsc(monkeypatch) -> None:
    """Only auto mode needs MPI size; explicit local/distributed modes are pure config."""

    def fail_get_petsc():
        raise AssertionError("_get_petsc should not run for explicit matrix modes")

    monkeypatch.setattr(petsc_core, "_get_petsc", fail_get_petsc)

    assert petsc_core._use_distributed_matrix({"petsc_matrix_mode": "local"}) is False
    assert petsc_core._use_distributed_matrix({"petsc_matrix_mode": "distributed"}) is True


def test_auto_petsc_matrix_mode_uses_comm_world_size(monkeypatch) -> None:
    class FakeComm:
        def getSize(self) -> int:
            return 2

    class FakePETSc:
        COMM_WORLD = FakeComm()

    monkeypatch.setattr(petsc_core, "_get_petsc", lambda: FakePETSc)

    assert petsc_core._use_distributed_matrix({"petsc_matrix_mode": "auto"}) is True


def test_root_scatter_default_is_off_without_env(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_PETSC_ROOT_SCATTER", raising=False)

    assert _scfg_bool(None, "root_scatter", "LINSTABPY_PETSC_ROOT_SCATTER", False) is False
    assert (
        _scfg_bool({"root_scatter": True}, "root_scatter", "LINSTABPY_PETSC_ROOT_SCATTER", False)
        is True
    )


def test_import_linstabpy_is_petsc_free() -> None:
    code = (
        "import sys; "
        "before = {'petsc4py' in sys.modules, 'slepc4py' in sys.modules}; "
        "import linstabpy; "
        "after = {'petsc4py' in sys.modules, 'slepc4py' in sys.modules}; "
        "print(int(bool(after - before)))"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    assert proc.stdout.strip() == "0"


def test_import_petsc_core_is_petsc_free() -> None:
    code = (
        "import sys; "
        "before = set(sys.modules); "
        "import linstabpy.utils.petsc_core as pc; "
        "after = set(sys.modules); "
        "print(int('petsc4py' in (after - before)), int('PETSc' in pc.__dict__))"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    assert proc.stdout.strip() == "0 0"
