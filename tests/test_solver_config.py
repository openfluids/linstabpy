# SPDX-License-Identifier: GPL-3.0-only
"""Tests for solver configuration helpers that do not require PETSc."""

from __future__ import annotations

import subprocess
import sys

import pytest

from linstabpy.config import Config
from linstabpy.utils import petsc_core
from linstabpy.utils._solver_cfg import (
    DIRECT_SOLVER_TYPES,
    INEXACT_EIGEN_SOLVER_TYPES,
    ITERATIVE_EIGEN_SOLVER_TYPES,
    ITERATIVE_LINEAR_SOLVER_TYPES,
    SolverConfig,
    _parse_optional_int,
    _scfg,
    _scfg_bool,
    _scfg_float,
    _scfg_int,
    _scfg_optional_int,
)


def test_solver_config_dict_overrides_environment(monkeypatch) -> None:
    """YAML solver values have precedence over environment fallbacks."""
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


def test_solver_config_float_and_int_resolution(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_FLOAT_KEY", "0.25")
    monkeypatch.setenv("LINSTABPY_INT_KEY", "8")

    assert _scfg_float({"eps_tol": 1e-6}, "eps_tol", "LINSTABPY_FLOAT_KEY", 1e-8) == 1e-6
    assert _scfg_float(None, "eps_tol", "LINSTABPY_FLOAT_KEY", 1e-8) == 0.25
    assert _scfg_float(None, "eps_tol", "LINSTABPY_MISSING_FLOAT", 1e-8) == 1e-8
    assert _scfg_int({"max_it": 40}, "max_it", "LINSTABPY_INT_KEY", 20) == 40
    assert _scfg_int(None, "max_it", "LINSTABPY_INT_KEY", 20) == 8
    assert _scfg_int(None, "max_it", "LINSTABPY_MISSING_INT", 20) == 20


def test_parse_optional_int_sentinels_and_invalid() -> None:
    assert _parse_optional_int(None, "ncv") is None
    assert _parse_optional_int("none", "ncv") is None
    assert _parse_optional_int("petsc", "ncv") is None
    assert _parse_optional_int("7", "ncv") == 7
    with pytest.raises(ValueError, match="ncv"):
        _parse_optional_int("nope", "ncv")


def test_scfg_optional_int_yaml_then_env_then_default(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_NCV", "12")
    assert _scfg_optional_int({"ncv": "none"}, "ncv", "LINSTABPY_NCV", 4) is None
    assert _scfg_optional_int(None, "ncv", "LINSTABPY_NCV", 4) == 12
    monkeypatch.delenv("LINSTABPY_NCV")
    assert _scfg_optional_int(None, "ncv", "LINSTABPY_NCV", 4) == 4


def test_solver_config_len_matches_mapping() -> None:
    cfg = SolverConfig({"a": 1, "b": 2})
    assert len(cfg) == 2
    assert SolverConfig.from_config(None) is None
    assert _scfg_bool({"root_scatter": "on"}, "root_scatter", "LINSTABPY_BOOL_KEY", False) is True
    assert _scfg_bool({"root_scatter": "nope"}, "root_scatter", "LINSTABPY_BOOL_KEY", True) is False


def test_solver_config_wrapper_preserves_mapping_behavior() -> None:
    raw = {"petsc_matrix_mode": "local", "eigen": {"eps_tol": 1e-9}}
    cfg = SolverConfig.from_config(raw)

    assert isinstance(cfg, SolverConfig)
    assert SolverConfig.from_config(cfg) is cfg
    assert cfg["petsc_matrix_mode"] == "local"
    assert cfg.get("eigen") == {"eps_tol": 1e-9}
    assert dict(cfg) == raw


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


def test_solver_config_exports_solver_family_sets() -> None:
    assert {"mumps", "direct", "lu"} == set(DIRECT_SOLVER_TYPES)
    assert {"iterative", "krylov", "petsc"} == set(ITERATIVE_LINEAR_SOLVER_TYPES)
    assert {"iterative", "krylov", "slepc"} == set(ITERATIVE_EIGEN_SOLVER_TYPES)
    assert {"inexact"} == set(INEXACT_EIGEN_SOLVER_TYPES)


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


def _mumps_opts(raw: dict) -> dict[str, str]:
    return petsc_core.mumps_icntl_petsc_options(Config.load(raw).mpi)


def test_mumps_icntl_petsc_options_hardened_defaults() -> None:
    # ICNTL(7) opt-in gate off by default -> no key; 14/24 hardened by default.
    assert _mumps_opts({}) == {"mat_mumps_icntl_14": "100", "mat_mumps_icntl_24": "1"}


def test_mumps_icntl_7_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="mumps_ordering"):
        Config.load({"solver": {"mumps_icntl_7": True}}, env={})


def test_mumps_icntl_petsc_options_overrides() -> None:
    assert _mumps_opts({"mumps_icntl_14": 200})["mat_mumps_icntl_14"] == "200"
    assert _mumps_opts({"mumps_icntl_24": 7})["mat_mumps_icntl_24"] == "7"


def test_mumps_icntl_petsc_options_none_opts_out() -> None:
    # "none"/"petsc" sentinels -> leave the setting to PETSc/MUMPS (no key).
    opts = _mumps_opts({"mumps_icntl_14": "none", "mumps_icntl_24": "petsc"})
    assert "mat_mumps_icntl_14" not in opts
    assert "mat_mumps_icntl_24" not in opts


def test_mumps_icntl_petsc_options_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINSTABPY_PETSC_MUMPS_ICNTL_14", "321")
    assert _mumps_opts({})["mat_mumps_icntl_14"] == "321"


def _uses_mumps(raw: dict) -> bool:
    return petsc_core.uses_mumps_factorization(Config.load(raw, env={}))


def test_uses_mumps_factorization_is_true_by_default() -> None:
    # Every stage defaults to MUMPS, so the ICNTL entries must go in.
    assert _uses_mumps({}) is True


def test_uses_mumps_factorization_is_false_on_a_pure_gpu_run() -> None:
    # No stage can reach MUMPS. PETSc would report the ICNTL entries as unused.
    raw = {
        "solver": {
            "linsolve": {"factor_backend": "cudss"},
            "resolvent": {"factor_backend": "cudss"},
            "eigen": {"factor_backend": "cudss"},
        }
    }
    assert _uses_mumps(raw) is False


def test_uses_mumps_factorization_is_true_when_one_stage_keeps_mumps() -> None:
    # The eigenvalue stage stays on the CPU. The entries are still needed.
    raw = {
        "solver": {
            "linsolve": {"factor_backend": "cudss"},
            "resolvent": {"factor_backend": "cudss"},
            "eigen": {"factor_backend": "mumps"},
        }
    }
    assert _uses_mumps(raw) is True


def test_uses_mumps_factorization_is_true_for_auto() -> None:
    # "auto" resolves at solve time and can still select MUMPS.
    raw = {
        "solver": {
            "linsolve": {"factor_backend": "auto"},
            "resolvent": {"factor_backend": "auto"},
            "eigen": {"factor_backend": "auto"},
        }
    }
    assert _uses_mumps(raw) is True


def test_typed_config_load_defaults_and_yaml_path(tmp_path) -> None:
    empty = Config.load(None, env={})
    assert empty.linsolve.type == "mumps"
    assert empty.eigen.eps_max_it(10) == 500
    assert empty.eigen.eps_max_it(80) == 800
    path = tmp_path / "solver.yaml"
    path.write_text(
        "solver:\n"
        "  linsolve:\n"
        "    type: iterative\n"
        "    ksp_monitor: yes\n"
        "    factor_backend: none\n"
        "  eigen:\n"
        "    st_solver: lu\n"
        "    eps_max_it: 12\n"
        "    pc_type: lu\n"
        "  assembly_metrics_output: '  '\n"
        "  assembly_metrics_dir:\n",
        encoding="utf-8",
    )
    loaded = Config.load(path, env={})
    assert loaded.linsolve.type == "iterative"
    assert loaded.linsolve.ksp_monitor is True
    assert loaded.eigen.eps_max_it(100) == 12
    assert loaded.mesh.assembly_metrics_output is None
    assert loaded.mesh.assembly_metrics_dir is None


def test_typed_config_env_and_st_solver_routes() -> None:
    env = {
        "LINSTABPY_FACTOR_BACKEND": "superlu_dist",
        "LINSTABPY_FACTOR_PRECISION": "single",
        "LINSTABPY_MUMPS_BLR": "1",
        "LINSTABPY_MUMPS_BLR_TOL": "1e-6",
        "LINSTABPY_MUMPS_ORDERING": "metis",
        "LINSTABPY_AUTO_ROUTE_DOF_THRESHOLD": "1234",
        "LINSTABPY_ASSEMBLY_METRICS_OUTPUT": "",
    }
    cfg = Config.load({"solver": {"linsolve": {"factor_precision": "double"}}}, env=env)
    assert cfg.linsolve.factor_backend == "superlu_dist"
    assert cfg.linsolve.factor_precision == "single"
    assert cfg.linsolve.mumps_blr is True
    assert cfg.eigen.auto_route_dof_threshold == 1234
    assert cfg.mesh.assembly_metrics_output is None

    iterative = Config.load({"solver": {"eigen": {"st_solver": "slepc", "pc_type": "asm"}}}, env={})
    assert iterative.eigen.st_factor_backend is None
    lu_pc = Config.load({"solver": {"eigen": {"st_solver": "slepc", "pc_type": "lu"}}}, env={})
    assert lu_pc.eigen.st_factor_backend == "mumps"
    strumpack = Config.load({"solver": {"eigen": {"st_solver": "strumpack_fgmres"}}}, env={})
    assert strumpack.eigen.st_factor_backend == "strumpack"
    fgmres = Config.load({"solver": {"eigen": {"st_solver": "fgmres_lu"}}}, env={})
    assert fgmres.eigen.st_factor_backend == "mumps"
    inexact = Config.load({"solver": {"eigen": {"st_solver": "inexact"}}}, env={})
    assert inexact.eigen.st_factor_backend is None
    yaml_prec = Config.load({"solver": {"linsolve": {"factor_precision": "single"}}}, env={})
    assert yaml_prec.linsolve.factor_precision == "single"
    yaml_blr = Config.load(
        {
            "solver": {
                "linsolve": {"mumps_blr": True, "mumps_blr_tol": 1e-5, "mumps_ordering": "pord"}
            }
        },
        env={},
    )
    assert yaml_blr.linsolve.mumps_blr is True
    yaml_thr = Config.load({"solver": {"eigen": {"auto_route_dof_threshold": 77}}}, env={})
    assert yaml_thr.eigen.auto_route_dof_threshold == 77


def test_typed_config_rejects_removed_mumps_icntl7() -> None:
    with pytest.raises(ValueError, match="mumps_ordering"):
        Config.load({"solver": {"mumps_icntl_7": 5}}, env={})
    with pytest.raises(ValueError, match="LINSTABPY_MUMPS_ORDERING"):
        Config.load(None, env={"LINSTABPY_PETSC_SET_MUMPS_ICNTL_7": "1"})


def test_typed_helpers_cover_none_mapping_and_string_bools() -> None:
    from linstabpy.config import typed as typed_mod

    assert typed_mod._solver_mapping(None) == {}
    assert typed_mod._scfg_bool({"ksp_monitor": "yes"}, "ksp_monitor", {}, "X", False) is True
    assert typed_mod._parse_optional_int(None) is None
    assert typed_mod._scfg_float({"ksp_rtol": 1.0e-8}, "ksp_rtol", {}, "X", 1.0e-10) == 1.0e-8
    assert typed_mod._scfg_int({"ksp_max_it": 12}, "ksp_max_it", {}, "X", 20000) == 12
    cfg = Config.load(
        {"solver": {"linsolve": {"ksp_rtol": 3.0e-9, "ksp_max_it": 33}}},
        env={"LINSTABPY_LINSOLVE_PC_FACTOR_SOLVER_TYPE": "strumpack"},
    )
    assert cfg.linsolve.factor_backend == "strumpack"
    assert cfg.linsolve.ksp_rtol == 3.0e-9
    assert cfg.linsolve.ksp_max_it == 33
