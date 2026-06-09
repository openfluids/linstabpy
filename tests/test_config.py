# SPDX-License-Identifier: GPL-3.0-only
"""Typed config seam tests against the live legacy config readers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest

from linstabpy.config import Config
from linstabpy.solvers.backends import _as_bool, resolve_factor_backend
from linstabpy.solvers.eigenvalue import _probe_residual_rel_max, _resolve_st_factor_backend
from linstabpy.utils._solver_cfg import (
    SolverConfig,
    _scfg,
    _scfg_bool,
    _scfg_float,
    _scfg_int,
    _scfg_optional_int,
)


@dataclass(frozen=True)
class FieldSpec:
    site: str
    field: str
    section: str
    key: str
    env_var: str
    default: Any
    reader: Callable[[Mapping[str, Any] | None, str, str, Any], Any]
    json_value: Any
    env_value: str


FIELD_SPECS = [
    # linsolve.py, LINSTABPY_LINSOLVE family
    FieldSpec("linsolve.py:type", "linsolve.type", "LINSTABPY_LINSOLVE", "type", "LINSTABPY_LINSOLVE_TYPE", "mumps", _scfg, "ITERATIVE", "krylov"),
    FieldSpec("linsolve.py:ksp_type", "linsolve.ksp_type", "LINSTABPY_LINSOLVE", "ksp_type", "LINSTABPY_LINSOLVE_KSP_TYPE", "fgmres", _scfg, "GMRES", "bcgs"),
    FieldSpec("linsolve.py:pc_type", "linsolve.pc_type", "LINSTABPY_LINSOLVE", "pc_type", "LINSTABPY_LINSOLVE_PC_TYPE", "asm", _scfg, "BJACOBI", "lu"),
    FieldSpec("linsolve.py:ksp_pc_side", "linsolve.ksp_pc_side", "LINSTABPY_LINSOLVE", "ksp_pc_side", "LINSTABPY_LINSOLVE_KSP_PC_SIDE", "right", _scfg, "LEFT", "symmetric"),
    FieldSpec("linsolve.py:ksp_rtol", "linsolve.ksp_rtol", "LINSTABPY_LINSOLVE", "ksp_rtol", "LINSTABPY_LINSOLVE_KSP_RTOL", 1e-10, _scfg_float, 1e-7, "2e-7"),
    FieldSpec("linsolve.py:ksp_atol", "linsolve.ksp_atol", "LINSTABPY_LINSOLVE", "ksp_atol", "LINSTABPY_LINSOLVE_KSP_ATOL", 1e-14, _scfg_float, 1e-11, "2e-11"),
    FieldSpec("linsolve.py:ksp_max_it", "linsolve.ksp_max_it", "LINSTABPY_LINSOLVE", "ksp_max_it", "LINSTABPY_LINSOLVE_KSP_MAX_IT", 20000, _scfg_int, 111, "222"),
    FieldSpec("linsolve.py:asm_overlap", "linsolve.asm_overlap", "LINSTABPY_LINSOLVE", "asm_overlap", "LINSTABPY_LINSOLVE_ASM_OVERLAP", 2, _scfg_int, 3, "4"),
    FieldSpec("linsolve.py:sub_ksp_type", "linsolve.sub_ksp_type", "LINSTABPY_LINSOLVE", "sub_ksp_type", "LINSTABPY_LINSOLVE_SUB_KSP_TYPE", "preonly", _scfg, "GMRES", "richardson"),
    FieldSpec("linsolve.py:sub_pc_type", "linsolve.sub_pc_type", "LINSTABPY_LINSOLVE", "sub_pc_type", "LINSTABPY_LINSOLVE_SUB_PC_TYPE", "lu", _scfg, "ILU", "jacobi"),
    FieldSpec("linsolve.py:pc_factor_levels", "linsolve.pc_factor_levels", "LINSTABPY_LINSOLVE", "pc_factor_levels", "LINSTABPY_LINSOLVE_PC_FACTOR_LEVELS", 0, _scfg_int, 2, "3"),
    FieldSpec("linsolve.py:pc_factor_shift_type", "linsolve.pc_factor_shift_type", "LINSTABPY_LINSOLVE", "pc_factor_shift_type", "LINSTABPY_LINSOLVE_PC_FACTOR_SHIFT_TYPE", "nonzero", _scfg, "POSITIVE_DEFINITE", "none"),
    FieldSpec("linsolve.py:pc_factor_shift_amount", "linsolve.pc_factor_shift_amount", "LINSTABPY_LINSOLVE", "pc_factor_shift_amount", "LINSTABPY_LINSOLVE_PC_FACTOR_SHIFT_AMOUNT", 1e-12, _scfg_float, 3e-12, "4e-12"),
    FieldSpec("linsolve.py:ksp_monitor", "linsolve.ksp_monitor", "LINSTABPY_LINSOLVE", "ksp_monitor", "LINSTABPY_LINSOLVE_KSP_MONITOR", False, _scfg_bool, True, "yes"),
    # linsolve.py, LINSTABPY_RESOLVENT family
    FieldSpec("linsolve.py:resolvent type", "resolvent.linear.type", "LINSTABPY_RESOLVENT", "type", "LINSTABPY_RESOLVENT_TYPE", "mumps", _scfg, "ITERATIVE", "krylov"),
    FieldSpec("linsolve.py:resolvent ksp_type", "resolvent.linear.ksp_type", "LINSTABPY_RESOLVENT", "ksp_type", "LINSTABPY_RESOLVENT_KSP_TYPE", "fgmres", _scfg, "GMRES", "bcgs"),
    FieldSpec("linsolve.py:resolvent pc_type", "resolvent.linear.pc_type", "LINSTABPY_RESOLVENT", "pc_type", "LINSTABPY_RESOLVENT_PC_TYPE", "asm", _scfg, "BJACOBI", "lu"),
    FieldSpec("linsolve.py:resolvent ksp_pc_side", "resolvent.linear.ksp_pc_side", "LINSTABPY_RESOLVENT", "ksp_pc_side", "LINSTABPY_RESOLVENT_KSP_PC_SIDE", "right", _scfg, "LEFT", "symmetric"),
    FieldSpec("linsolve.py:resolvent ksp_rtol", "resolvent.linear.ksp_rtol", "LINSTABPY_RESOLVENT", "ksp_rtol", "LINSTABPY_RESOLVENT_KSP_RTOL", 1e-10, _scfg_float, 1e-7, "2e-7"),
    FieldSpec("linsolve.py:resolvent ksp_atol", "resolvent.linear.ksp_atol", "LINSTABPY_RESOLVENT", "ksp_atol", "LINSTABPY_RESOLVENT_KSP_ATOL", 1e-14, _scfg_float, 1e-11, "2e-11"),
    FieldSpec("linsolve.py:resolvent ksp_max_it", "resolvent.linear.ksp_max_it", "LINSTABPY_RESOLVENT", "ksp_max_it", "LINSTABPY_RESOLVENT_KSP_MAX_IT", 20000, _scfg_int, 111, "222"),
    FieldSpec("linsolve.py:resolvent asm_overlap", "resolvent.linear.asm_overlap", "LINSTABPY_RESOLVENT", "asm_overlap", "LINSTABPY_RESOLVENT_ASM_OVERLAP", 2, _scfg_int, 3, "4"),
    FieldSpec("linsolve.py:resolvent sub_ksp_type", "resolvent.linear.sub_ksp_type", "LINSTABPY_RESOLVENT", "sub_ksp_type", "LINSTABPY_RESOLVENT_SUB_KSP_TYPE", "preonly", _scfg, "GMRES", "richardson"),
    FieldSpec("linsolve.py:resolvent sub_pc_type", "resolvent.linear.sub_pc_type", "LINSTABPY_RESOLVENT", "sub_pc_type", "LINSTABPY_RESOLVENT_SUB_PC_TYPE", "lu", _scfg, "ILU", "jacobi"),
    FieldSpec("linsolve.py:resolvent pc_factor_levels", "resolvent.linear.pc_factor_levels", "LINSTABPY_RESOLVENT", "pc_factor_levels", "LINSTABPY_RESOLVENT_PC_FACTOR_LEVELS", 0, _scfg_int, 2, "3"),
    FieldSpec("linsolve.py:resolvent pc_factor_shift_type", "resolvent.linear.pc_factor_shift_type", "LINSTABPY_RESOLVENT", "pc_factor_shift_type", "LINSTABPY_RESOLVENT_PC_FACTOR_SHIFT_TYPE", "nonzero", _scfg, "POSITIVE_DEFINITE", "none"),
    FieldSpec("linsolve.py:resolvent pc_factor_shift_amount", "resolvent.linear.pc_factor_shift_amount", "LINSTABPY_RESOLVENT", "pc_factor_shift_amount", "LINSTABPY_RESOLVENT_PC_FACTOR_SHIFT_AMOUNT", 1e-12, _scfg_float, 3e-12, "4e-12"),
    FieldSpec("linsolve.py:resolvent ksp_monitor", "resolvent.linear.ksp_monitor", "LINSTABPY_RESOLVENT", "ksp_monitor", "LINSTABPY_RESOLVENT_KSP_MONITOR", False, _scfg_bool, True, "yes"),
    # resolvent.py EPS subsection
    FieldSpec("resolvent.py:eps_tol", "resolvent.eps.eps_tol", "RESOLVENT_EPS", "eps_tol", "LINSTABPY_RESOLVENT_EPS_TOL", 1e-8, _scfg_float, 2e-8, "3e-8"),
    FieldSpec("resolvent.py:eps_max_it", "resolvent.eps.eps_max_it", "RESOLVENT_EPS", "eps_max_it", "LINSTABPY_RESOLVENT_EPS_MAX_IT", 1000, _scfg_int, 444, "555"),
    FieldSpec("resolvent.py:conv_test", "resolvent.eps.conv_test", "RESOLVENT_EPS", "conv_test", "LINSTABPY_RESOLVENT_EPS_CONV_TEST", "", _scfg, "REL", "norm"),
    FieldSpec("resolvent.py:true_residual", "resolvent.eps.true_residual", "RESOLVENT_EPS", "true_residual", "LINSTABPY_RESOLVENT_EPS_TRUE_RESIDUAL", False, _scfg_bool, True, "on"),
    FieldSpec("resolvent.py:monitor", "resolvent.eps.monitor", "RESOLVENT_EPS", "monitor", "LINSTABPY_RESOLVENT_EPS_MONITOR", False, _scfg_bool, True, "on"),
    FieldSpec("resolvent.py:monitor_every", "resolvent.eps.monitor_every", "RESOLVENT_EPS", "monitor_every", "LINSTABPY_RESOLVENT_EPS_MONITOR_EVERY", 10, _scfg_int, 7, "8"),
    # eigenvalue.py
    FieldSpec("eigenvalue.py:st_solver", "eigen.st_solver", "LINSTABPY_EIGEN", "st_solver", "LINSTABPY_EIGEN_ST_SOLVER", "mumps", _scfg, "ITERATIVE", "slepc"),
    FieldSpec("eigenvalue.py:pc_factor_solver_type", "eigen.pc_factor_solver_type", "LINSTABPY_EIGEN", "pc_factor_solver_type", "LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE", "", _scfg, "SUPERLU_DIST", "mumps"),
    FieldSpec("eigenvalue.py:cond_warn", "eigen.cond_warn", "LINSTABPY_EIGEN", "cond_warn", "LINSTABPY_EIGEN_COND_WARN", True, _scfg_bool, False, "off"),
    FieldSpec("eigenvalue.py:cond_warn_max_n", "eigen.cond_warn_max_n", "LINSTABPY_EIGEN", "cond_warn_max_n", "LINSTABPY_EIGEN_COND_WARN_MAX_N", 1500, _scfg_int, 120, "130"),
    FieldSpec("eigenvalue.py:cond_warn_threshold", "eigen.cond_warn_threshold", "LINSTABPY_EIGEN", "cond_warn_threshold", "LINSTABPY_EIGEN_COND_WARN_THRESHOLD", 1e12, _scfg_float, 2e9, "3e9"),
    FieldSpec("eigenvalue.py:ksp_type", "eigen.ksp_type", "LINSTABPY_EIGEN", "ksp_type", "LINSTABPY_EIGEN_KSP_TYPE", "gmres", _scfg, "FGMRES", "bcgs"),
    FieldSpec("eigenvalue.py:pc_type", "eigen.pc_type", "LINSTABPY_EIGEN", "pc_type", "LINSTABPY_EIGEN_PC_TYPE", "asm", _scfg, "LU", "bjacobi"),
    FieldSpec("eigenvalue.py:ksp_pc_side", "eigen.ksp_pc_side", "LINSTABPY_EIGEN", "ksp_pc_side", "LINSTABPY_EIGEN_KSP_PC_SIDE", "left", _scfg, "RIGHT", "symmetric"),
    FieldSpec("eigenvalue.py:ksp_rtol", "eigen.ksp_rtol", "LINSTABPY_EIGEN", "ksp_rtol", "LINSTABPY_EIGEN_KSP_RTOL", 1e-8, _scfg_float, 2e-6, "3e-6"),
    FieldSpec("eigenvalue.py:ksp_max_it", "eigen.ksp_max_it", "LINSTABPY_EIGEN", "ksp_max_it", "LINSTABPY_EIGEN_KSP_MAX_IT", 20000, _scfg_int, 333, "444"),
    FieldSpec("eigenvalue.py:asm_overlap", "eigen.asm_overlap", "LINSTABPY_EIGEN", "asm_overlap", "LINSTABPY_EIGEN_ASM_OVERLAP", 2, _scfg_int, 5, "6"),
    FieldSpec("eigenvalue.py:sub_ksp_type", "eigen.sub_ksp_type", "LINSTABPY_EIGEN", "sub_ksp_type", "LINSTABPY_EIGEN_SUB_KSP_TYPE", "preonly", _scfg, "GMRES", "richardson"),
    FieldSpec("eigenvalue.py:sub_pc_type", "eigen.sub_pc_type", "LINSTABPY_EIGEN", "sub_pc_type", "LINSTABPY_EIGEN_SUB_PC_TYPE", "lu", _scfg, "ILU", "jacobi"),
    FieldSpec("eigenvalue.py:conv_test", "eigen.conv_test", "LINSTABPY_EIGEN", "conv_test", "LINSTABPY_EIGEN_CONV_TEST", "", _scfg, "REL", "norm"),
    FieldSpec("eigenvalue.py:eps_tol", "eigen.eps_tol", "LINSTABPY_EIGEN", "eps_tol", "LINSTABPY_EIGEN_EPS_TOL", 1e-10, _scfg_float, 4e-9, "5e-9"),
    FieldSpec("eigenvalue.py:residual_error_type", "eigen.residual_error_type", "LINSTABPY_EIGEN", "residual_error_type", "LINSTABPY_EIGEN_RESIDUAL_ERROR_TYPE", "relative", _scfg, "ABSOLUTE", "relative"),
    FieldSpec("eigenvalue.py:monitor", "eigen.monitor", "LINSTABPY_EIGEN", "monitor", "LINSTABPY_EIGEN_MONITOR", False, _scfg_bool, True, "on"),
    FieldSpec("eigenvalue.py:monitor_every", "eigen.monitor_every", "LINSTABPY_EIGEN", "monitor_every", "LINSTABPY_EIGEN_MONITOR_EVERY", 10, _scfg_int, 11, "12"),
    FieldSpec("eigenvalue.py:true_residual", "eigen.true_residual", "LINSTABPY_EIGEN", "true_residual", "LINSTABPY_EIGEN_TRUE_RESIDUAL", False, _scfg_bool, True, "on"),
    FieldSpec("eigenvalue.py:extract_all_converged", "eigen.extract_all_converged", "LINSTABPY_EIGEN", "extract_all_converged", "LINSTABPY_EIGEN_EXTRACT_ALL_CONVERGED", False, _scfg_bool, True, "on"),
    FieldSpec("eigenvalue.py:weighted_norm_tol", "eigen.weighted_norm_tol", "LINSTABPY_EIGEN", "weighted_norm_tol", "LINSTABPY_EIGEN_WEIGHTED_NORM_TOL", 1e-12, _scfg_float, 2e-12, "3e-12"),
    FieldSpec("eigenvalue.py:eps_max_it", "eigen.eps_max_it_explicit", "LINSTABPY_EIGEN", "eps_max_it", "LINSTABPY_EIGEN_EPS_MAX_IT", None, _scfg_optional_int, 600, "700"),
    # spatial_eig.py
    FieldSpec("spatial_eig.py:st_solver", "spatial.st_solver", "SPATIAL", "st_solver", "LINSTABPY_SPATIAL_ST_SOLVER", "mumps", _scfg, "LU", "direct"),
    # assembly.py / petsc_core.py / _solver_cfg.py top-level
    FieldSpec("assembly.py:assembly_backend", "mesh.assembly_backend", "ROOT", "assembly_backend", "LINSTABPY_ASSEMBLY_BACKEND", "scipy", _scfg, "PETSC", "scipy"),
    FieldSpec("assembly.py:assembly_metrics", "mesh.assembly_metrics", "ROOT", "assembly_metrics", "LINSTABPY_ASSEMBLY_METRICS", False, _scfg_bool, True, "yes"),
    FieldSpec("petsc_core.py:petsc_matrix_mode", "mpi.petsc_matrix_mode", "ROOT", "petsc_matrix_mode", "LINSTABPY_PETSC_MATRIX_MODE", "auto", _scfg, "MPIAIJ", "seqaij"),
    FieldSpec("eigenvalue.py:root_scatter_enabled", "mpi.root_scatter", "ROOT", "root_scatter", "LINSTABPY_PETSC_ROOT_SCATTER", False, _scfg_bool, True, "on"),
    FieldSpec("_solver_cfg.py:mumps_icntl_7", "mpi.mumps_icntl_7", "ROOT", "mumps_icntl_7", "LINSTABPY_PETSC_SET_MUMPS_ICNTL_7", False, _scfg_bool, True, "yes"),
    FieldSpec("_solver_cfg.py:mumps_icntl_14", "mpi.mumps_icntl_14", "ROOT", "mumps_icntl_14", "LINSTABPY_PETSC_MUMPS_ICNTL_14", 100, _scfg_optional_int, 140, "141"),
    FieldSpec("_solver_cfg.py:mumps_icntl_24", "mpi.mumps_icntl_24", "ROOT", "mumps_icntl_24", "LINSTABPY_PETSC_MUMPS_ICNTL_24", 1, _scfg_optional_int, 24, "25"),
]

RAW_VALUE_SPECS = [
    ("assembly.py:assembly_metrics_output", "mesh.assembly_metrics_output", "assembly_metrics_output", "LINSTABPY_ASSEMBLY_METRICS_OUTPUT", "metrics.json", "env_metrics.json"),
    ("assembly.py:assembly_metrics_dir", "mesh.assembly_metrics_dir", "assembly_metrics_dir", "LINSTABPY_ASSEMBLY_METRICS_DIR", "metrics", "env_metrics"),
    ("assembly.py:assembly_metrics_tag", "mesh.assembly_metrics_tag", "assembly_metrics_tag", "LINSTABPY_ASSEMBLY_METRICS_TAG", "case-a", "case-b"),
]


def _clear_linstabpy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("LINSTABPY_"):
            monkeypatch.delenv(name, raising=False)


def _field(obj: object, dotted: str) -> Any:
    value: Any = obj
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def _subsection(raw: Mapping[str, Any], section: str) -> Mapping[str, Any] | None:
    sc = SolverConfig(raw)
    if section == "ROOT":
        return sc
    if section == "SPATIAL":
        return sc.get("spatial", sc.get("eigen", {}))
    if section == "RESOLVENT_EPS":
        res = sc.subsection("LINSTABPY_RESOLVENT")
        return res.get("eigen", res) if isinstance(res, Mapping) else {}
    return sc.subsection(section)


def _raw_with(spec: FieldSpec, value: Any) -> dict[str, Any]:
    if spec.section == "ROOT":
        return {spec.key: value}
    if spec.section == "SPATIAL":
        return {"spatial": {spec.key: value}}
    if spec.section == "RESOLVENT_EPS":
        return {"resolvent": {"eigen": {spec.key: value}}}
    name = {
        "LINSTABPY_LINSOLVE": "linsolve",
        "LINSTABPY_RESOLVENT": "resolvent",
        "LINSTABPY_EIGEN": "eigen",
    }[spec.section]
    return {name: {spec.key: value}}


def _expected(spec: FieldSpec, raw: Mapping[str, Any]) -> Any:
    return spec.reader(_subsection(raw, spec.section), spec.key, spec.env_var, spec.default)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_linstabpy_env(monkeypatch)


def test_import_resolves_to_typed_package_module() -> None:
    import linstabpy.config as c

    assert c.Config.__module__ == "linstabpy.config.typed"
    assert c.Config.load({}).linsolve.type == "mumps"


def test_frozen_and_dynamic_eigen_eps_max_it() -> None:
    cfg = Config.load({})

    with pytest.raises(FrozenInstanceError):
        cfg.eigen.eps_tol = 1e-6  # type: ignore[misc]
    assert cfg.eigen.eps_max_it_explicit is None
    assert cfg.eigen.eps_max_it(20) == 500
    assert cfg.eigen.eps_max_it(80) == 800
    assert Config.load({"eigen": {"eps_max_it": 123}}).eigen.eps_max_it(80) == 123


@pytest.mark.parametrize("spec", FIELD_SPECS, ids=lambda s: s.field)
def test_differential_oracle_defaults(spec: FieldSpec) -> None:
    typed = Config.load({})
    assert _field(typed, spec.field) == _expected(spec, {})


@pytest.mark.parametrize("spec", FIELD_SPECS, ids=lambda s: s.field)
def test_differential_oracle_jsonc_overrides(spec: FieldSpec) -> None:
    raw = _raw_with(spec, spec.json_value)
    typed = Config.load(raw)
    assert _field(typed, spec.field) == _expected(spec, raw)


@pytest.mark.parametrize("spec", FIELD_SPECS, ids=lambda s: s.field)
def test_differential_oracle_env_overrides(monkeypatch: pytest.MonkeyPatch, spec: FieldSpec) -> None:
    monkeypatch.setenv(spec.env_var, spec.env_value)
    typed = Config.load({})
    assert _field(typed, spec.field) == _expected(spec, {})


@pytest.mark.parametrize("spec", FIELD_SPECS, ids=lambda s: s.field)
def test_jsonc_beats_env(monkeypatch: pytest.MonkeyPatch, spec: FieldSpec) -> None:
    monkeypatch.setenv(spec.env_var, spec.env_value)
    raw = _raw_with(spec, spec.json_value)
    typed = Config.load(raw)
    assert _field(typed, spec.field) == _expected(spec, raw)


@pytest.mark.parametrize("site,field,key,env_var,json_value,env_value", RAW_VALUE_SPECS, ids=lambda row: row[1])
def test_raw_string_fields_jsonc_and_env(monkeypatch: pytest.MonkeyPatch, site: str, field: str, key: str, env_var: str, json_value: str, env_value: str) -> None:
    assert _field(Config.load({}), field) is None
    monkeypatch.setenv(env_var, env_value)
    assert _field(Config.load({}), field) == env_value
    raw = {key: json_value}
    assert _field(Config.load(raw), field) == raw[key]


@pytest.mark.parametrize("section,field,prefix", [("LINSTABPY_LINSOLVE", "linsolve.factor_backend", "LINSTABPY_LINSOLVE"), ("LINSTABPY_RESOLVENT", "resolvent.linear.factor_backend", "LINSTABPY_RESOLVENT")])
def test_linsolve_family_factor_backend(monkeypatch: pytest.MonkeyPatch, section: str, field: str, prefix: str) -> None:
    raw = _raw_with(FieldSpec("", field, section, "factor_backend", "", "", _scfg, "superlu_dist", ""), "superlu_dist")
    assert _field(Config.load(raw), field) == resolve_factor_backend(_subsection(raw, section), env_var=f"{prefix}_PC_FACTOR_SOLVER_TYPE", default="mumps")
    monkeypatch.setenv(f"{prefix}_PC_FACTOR_SOLVER_TYPE", "mumps")
    assert _field(Config.load({}), field) == resolve_factor_backend({}, env_var=f"{prefix}_PC_FACTOR_SOLVER_TYPE", default="mumps")


def test_spatial_factor_backend_matches_context_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {"spatial": {"st_solver": "lu", "factor_backend": "superlu_dist"}}
    assert Config.load(raw).spatial.factor_backend == resolve_factor_backend(raw["spatial"], default=None)
    monkeypatch.setenv("LINSTABPY_SPATIAL_ST_SOLVER", "mumps")
    monkeypatch.setenv("LINSTABPY_FACTOR_BACKEND", "superlu_dist")
    assert Config.load({}).spatial.factor_backend == resolve_factor_backend({}, default="mumps")


@pytest.mark.parametrize("st_solver,extra", [("mumps", {}), ("lu", {}), ("direct", {}), ("iterative", {"pc_type": "lu"}), ("iterative", {"pc_type": "asm"}), ("slepc", {"pc_type": "lu"})])
def test_eigen_st_factor_backend_matches_live_reader(monkeypatch: pytest.MonkeyPatch, st_solver: str, extra: dict[str, Any]) -> None:
    raw = {"eigen": {"st_solver": st_solver, **extra}}
    assert Config.load(raw).eigen.st_factor_backend == _resolve_st_factor_backend(st_solver, raw["eigen"])

    monkeypatch.setenv("LINSTABPY_EIGEN_ST_SOLVER", st_solver)
    for key, value in extra.items():
        monkeypatch.setenv(f"LINSTABPY_EIGEN_{key.upper()}", str(value))
    assert Config.load({}).eigen.st_factor_backend == _resolve_st_factor_backend(st_solver, {})


def test_eigen_st_factor_backend_jsonc_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE", "mumps")
    raw = {"eigen": {"st_solver": "iterative", "pc_type": "lu", "pc_factor_solver_type": "superlu_dist"}}
    assert Config.load(raw).eigen.st_factor_backend == _resolve_st_factor_backend("iterative", raw["eigen"])


def test_resolvent_eps_subsection_falls_back_to_linsolve() -> None:
    raw = {"linsolve": {"eigen": {"eps_tol": 9e-7, "monitor": True}}}
    typed = Config.load(raw)
    eps_sub = SolverConfig(raw).subsection("LINSTABPY_RESOLVENT")["eigen"]
    assert typed.resolvent.eps.eps_tol == _scfg_float(eps_sub, "eps_tol", "LINSTABPY_RESOLVENT_EPS_TOL", 1e-8)
    assert typed.resolvent.eps.monitor == _scfg_bool(eps_sub, "monitor", "LINSTABPY_RESOLVENT_EPS_MONITOR", False)


def test_eigen_inexact_matches_plain_cast_jsonc_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINSTABPY_EIGEN_INEXACT_KSP_TYPE", "lowercase-env-ignored")
    raw = {
        "eigen": {
            "inexact": {
                "ksp_type": "FGMRES_Mixed",
                "ksp_rtol": "2e-7",
                "ksp_atol": "3e-11",
                "ksp_max_it": "321",
                "pc_type": "ASM_Mixed",
                "asm_overlap": "4",
                "sub_pc_type": "ILU_Mixed",
                "pc_factor_levels": "5",
            }
        }
    }
    d = raw["eigen"]["inexact"]
    cfg = Config.load(raw).eigen.inexact

    assert cfg.ksp_type == str(d.get("ksp_type", "fgmres"))
    assert cfg.ksp_type == "FGMRES_Mixed"
    assert cfg.ksp_rtol == float(d.get("ksp_rtol", 1e-8))
    assert cfg.ksp_atol == float(d.get("ksp_atol", 1e-10))
    assert cfg.ksp_max_it == int(d.get("ksp_max_it", 200))
    assert cfg.pc_type == str(d.get("pc_type", "asm"))
    assert cfg.asm_overlap == int(d.get("asm_overlap", 1))
    assert cfg.sub_pc_type == str(d.get("sub_pc_type", "ilu"))
    assert cfg.pc_factor_levels == int(d.get("pc_factor_levels", 2))
    assert Config.load({}).eigen.inexact.ksp_type == "fgmres"


def test_eigen_backend_probe_matches_live_reader() -> None:
    assert Config.load({}).eigen.backend_probe.residual_rel_max == _probe_residual_rel_max({})
    assert Config.load({}).eigen.backend_probe.enabled is None

    for value in [True, False, "yes", "off", 1, 0]:
        raw = {"eigen": {"backend_probe": {"residual_rel_max": "4e-7", "enabled": value}}}
        assert Config.load(raw).eigen.backend_probe.residual_rel_max == _probe_residual_rel_max(raw["eigen"])
        assert Config.load(raw).eigen.backend_probe.enabled == _as_bool(value)


def test_ncv_jsonc_only_and_env_has_no_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINSTABPY_RESOLVENT_EPS_NCV", "41")
    monkeypatch.setenv("LINSTABPY_EIGEN_NCV", "52")
    assert Config.load({}).resolvent.eps.ncv is None
    assert Config.load({}).eigen.ncv is None

    raw = {"resolvent": {"eigen": {"ncv": "31"}}, "eigen": {"ncv": "51"}}
    assert Config.load(raw).resolvent.eps.ncv == int(raw["resolvent"]["eigen"]["ncv"])
    assert Config.load(raw).eigen.ncv == int(raw["eigen"]["ncv"])


def test_assembly_metrics_log_explicit_dynamic_default_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # None only when unset by both cfg and env (call site then applies =enabled).
    assert Config.load({}).mesh.assembly_metrics_log_explicit is None
    monkeypatch.setenv("LINSTABPY_ASSEMBLY_METRICS_LOG", "yes")
    assert Config.load({}).mesh.assembly_metrics_log_explicit is True
    monkeypatch.setenv("LINSTABPY_ASSEMBLY_METRICS_LOG", "off")
    assert Config.load({}).mesh.assembly_metrics_log_explicit is False
    # When set, must match the LIVE reader (_scfg_bool), NOT backends._as_bool.
    for value in (True, False, "yes", "no", "1", "0", "on", "off", "y", "n", "maybe", "true"):
        raw = {"assembly_metrics_log": value}
        expected = _scfg_bool(raw, "assembly_metrics_log", "LINSTABPY_ASSEMBLY_METRICS_LOG", False)
        assert Config.load(raw).mesh.assembly_metrics_log_explicit == expected, value
    # _scfg_bool semantics: "y" is False (not in {1,true,yes,on}) and garbage never raises.
    assert Config.load({"assembly_metrics_log": "y"}).mesh.assembly_metrics_log_explicit is False
    assert Config.load({"assembly_metrics_log": "maybe"}).mesh.assembly_metrics_log_explicit is False


def test_site_to_field_table_covers_known_reader_surface() -> None:
    assert len({(s.field, s.env_var, s.default, s.section) for s in FIELD_SPECS}) == len(FIELD_SPECS)
    assert len(FIELD_SPECS) == 64
    assert len(RAW_VALUE_SPECS) == 3
    # factor backends are context readers, not simple _scfg tuples.
    assert Config.load({}).eigen.st_factor_backend == _resolve_st_factor_backend("mumps", {})
