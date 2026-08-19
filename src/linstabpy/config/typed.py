# SPDX-License-Identifier: GPL-3.0-only
"""Typed configuration seam for solver policy.

Mirrors the ``utils._solver_cfg`` YAML/env/default resolution without
migrating call sites yet.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linstabpy.linalg.backends import (
    DEFAULT_AUTO_ROUTE_DOF_THRESHOLD,
    DEFAULT_FACTOR_BACKEND,
    DEFAULT_FACTOR_PRECISION,
    DEFAULT_MUMPS_BLR,
    DEFAULT_MUMPS_BLR_TOL,
    _as_bool,
    _normalize_factor_precision,
    _normalize_mumps_blr_tol,
    _normalize_mumps_ordering,
)
from linstabpy.utils._solver_cfg import DIRECT_SOLVER_TYPES, ITERATIVE_EIGEN_SOLVER_TYPES
from linstabpy.utils.yaml_io import load_config

_TRUE = {"1", "true", "yes", "on"}
_OPTIONAL_INT_SENTINELS = {"", "none", "null", "default", "petsc"}


def _solver_mapping(raw: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if raw is None:
        return {}
    solver = raw.get("solver") if isinstance(raw, Mapping) else None
    return solver if isinstance(solver, Mapping) else raw


def _subsection(solver: Mapping[str, Any], prefix: str) -> Mapping[str, Any]:
    if prefix == "LINSTABPY_RESOLVENT":
        return solver.get("resolvent", solver.get("linsolve", {})) or {}
    if prefix == "LINSTABPY_EIGEN":
        return solver.get("eigen", {}) or {}
    return solver.get("linsolve", {}) or {}


def _spatial_subsection(solver: Mapping[str, Any]) -> Mapping[str, Any]:
    return solver.get("spatial", solver.get("eigen", {})) or {}


def _resolvent_eps_subsection(solver: Mapping[str, Any]) -> Mapping[str, Any]:
    res = _subsection(solver, "LINSTABPY_RESOLVENT")
    return res.get("eigen", res) or {}


def _env_get(env: Mapping[str, str], key: str) -> str | None:
    return env[key] if key in env else None


def _scfg(
    cfg: Mapping[str, Any] | None, key: str, env: Mapping[str, str], env_var: str, default: str
) -> str:
    if cfg is not None and key in cfg:
        return str(cfg[key]).strip().lower()
    raw = _env_get(env, env_var)
    return (raw if raw is not None else default).strip().lower()


def _scfg_float(
    cfg: Mapping[str, Any] | None, key: str, env: Mapping[str, str], env_var: str, default: float
) -> float:
    if cfg is not None and key in cfg:
        return float(cfg[key])
    raw = _env_get(env, env_var)
    return float(raw if raw is not None else str(default))


def _scfg_int(
    cfg: Mapping[str, Any] | None, key: str, env: Mapping[str, str], env_var: str, default: int
) -> int:
    if cfg is not None and key in cfg:
        return int(cfg[key])
    raw = _env_get(env, env_var)
    return int(raw if raw is not None else str(default))


def _scfg_bool(
    cfg: Mapping[str, Any] | None, key: str, env: Mapping[str, str], env_var: str, default: bool
) -> bool:
    if cfg is not None and key in cfg:
        value = cfg[key]
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in _TRUE
    raw = _env_get(env, env_var)
    return (raw if raw is not None else ("1" if default else "0")).strip().lower() in _TRUE


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _OPTIONAL_INT_SENTINELS:
        return None
    return int(value)


def _scfg_optional_int(
    cfg: Mapping[str, Any] | None,
    key: str,
    env: Mapping[str, str],
    env_var: str,
    default: int | None,
) -> int | None:
    if cfg is not None and key in cfg:
        return _parse_optional_int(cfg[key])
    raw = _env_get(env, env_var)
    if raw is not None:
        return _parse_optional_int(raw)
    return default


def _raw_cfg_value(
    cfg: Mapping[str, Any] | None, key: str, env: Mapping[str, str], env_var: str
) -> str | None:
    if cfg is not None and key in cfg:
        value = cfg[key]
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    raw = _env_get(env, env_var)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _normalize_backend(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _OPTIONAL_INT_SENTINELS:
        return None
    return text


def _factor_backend(
    cfg: Mapping[str, Any] | None,
    env: Mapping[str, str],
    *,
    env_var: str | None = None,
    default: str | None = "mumps",
) -> str | None:
    if cfg is not None and "factor_backend" in cfg:
        return _normalize_backend(cfg["factor_backend"])
    if env_var is not None and env_var in env:
        return _normalize_backend(env[env_var])
    if "LINSTABPY_FACTOR_BACKEND" in env:
        return _normalize_backend(env["LINSTABPY_FACTOR_BACKEND"])
    return _normalize_backend(default)


def _factor_precision(cfg: Mapping[str, Any] | None, env: Mapping[str, str]) -> str:
    if "LINSTABPY_FACTOR_PRECISION" in env:
        return _normalize_factor_precision(env["LINSTABPY_FACTOR_PRECISION"])
    if cfg is not None and "factor_precision" in cfg:
        return _normalize_factor_precision(cfg["factor_precision"])
    return DEFAULT_FACTOR_PRECISION


def _mumps_blr(cfg: Mapping[str, Any] | None, env: Mapping[str, str]) -> bool:
    if "LINSTABPY_MUMPS_BLR" in env:
        return _as_bool(env["LINSTABPY_MUMPS_BLR"])
    if cfg is not None and "mumps_blr" in cfg:
        return _as_bool(cfg["mumps_blr"])
    return DEFAULT_MUMPS_BLR


def _mumps_blr_tol(cfg: Mapping[str, Any] | None, env: Mapping[str, str]) -> float:
    if "LINSTABPY_MUMPS_BLR_TOL" in env:
        return _normalize_mumps_blr_tol(env["LINSTABPY_MUMPS_BLR_TOL"])
    if cfg is not None and "mumps_blr_tol" in cfg:
        return _normalize_mumps_blr_tol(cfg["mumps_blr_tol"])
    return DEFAULT_MUMPS_BLR_TOL


def _mumps_ordering(cfg: Mapping[str, Any] | None, env: Mapping[str, str]) -> str | None:
    if "LINSTABPY_MUMPS_ORDERING" in env:
        return _normalize_mumps_ordering(env["LINSTABPY_MUMPS_ORDERING"])
    if cfg is not None and "mumps_ordering" in cfg:
        return _normalize_mumps_ordering(cfg["mumps_ordering"])
    return None


def _auto_route_dof_threshold(cfg: Mapping[str, Any] | None, env: Mapping[str, str]) -> int:
    """Resolve auto-route DOF threshold with precedence env > YAML > default."""
    if "LINSTABPY_AUTO_ROUTE_DOF_THRESHOLD" in env:
        return int(env["LINSTABPY_AUTO_ROUTE_DOF_THRESHOLD"])
    if cfg is not None and "auto_route_dof_threshold" in cfg:
        return int(cfg["auto_route_dof_threshold"])
    return DEFAULT_AUTO_ROUTE_DOF_THRESHOLD


def _st_factor_backend(
    st_solver: str,
    cfg: Mapping[str, Any] | None,
    env: Mapping[str, str],
    solver_default: str | None = None,
) -> str | None:
    if st_solver in DIRECT_SOLVER_TYPES:
        if st_solver == DEFAULT_FACTOR_BACKEND:
            # Generic direct route: a solver-level factor_backend is a real preference and
            # should apply here. This is the path that used to strand the band-marking EVP
            # on CPU MUMPS while the resolvent ran elsewhere.
            return _factor_backend(cfg, env, default=solver_default or DEFAULT_FACTOR_BACKEND)
        # st_solver already names a backend; do not override it from solver level.
        return _factor_backend(cfg, env, default=None)
    if st_solver in ITERATIVE_EIGEN_SOLVER_TYPES:
        pc_type = _scfg(cfg, "pc_type", env, "LINSTABPY_EIGEN_PC_TYPE", "asm")
        if pc_type != "lu":
            return None
        pc_factor_solver_type = _scfg(
            cfg, "pc_factor_solver_type", env, "LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE", ""
        )
        return _factor_backend(
            cfg,
            env,
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
            default=pc_factor_solver_type if pc_factor_solver_type else DEFAULT_FACTOR_BACKEND,
        )
    if st_solver == "strumpack_fgmres":
        return _factor_backend(
            cfg,
            env,
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
            default="strumpack",
        )
    if st_solver == "fgmres_lu":
        return _factor_backend(
            cfg,
            env,
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
            default=DEFAULT_FACTOR_BACKEND,
        )
    return None


@dataclass(frozen=True)
class LinSolveCfg:
    type: str
    factor_backend: str | None
    factor_precision: str
    mumps_blr: bool
    mumps_blr_tol: float
    mumps_ordering: str | None
    ksp_type: str
    pc_type: str
    ksp_pc_side: str
    ksp_rtol: float
    ksp_atol: float
    ksp_max_it: int
    asm_overlap: int
    sub_ksp_type: str
    sub_pc_type: str
    pc_factor_levels: int
    pc_factor_shift_type: str
    pc_factor_shift_amount: float
    ksp_monitor: bool

    @classmethod
    def from_mapping(
        cls,
        cfg: Mapping[str, Any] | None,
        env: Mapping[str, str],
        prefix: str,
        solver_default: str | None = None,
    ) -> "LinSolveCfg":
        return cls(
            type=_scfg(cfg, "type", env, f"{prefix}_TYPE", "mumps"),
            factor_backend=_factor_backend(
                cfg,
                env,
                env_var=f"{prefix}_PC_FACTOR_SOLVER_TYPE",
                default=solver_default or "mumps",
            ),
            factor_precision=_factor_precision(cfg, env),
            mumps_blr=_mumps_blr(cfg, env),
            mumps_blr_tol=_mumps_blr_tol(cfg, env),
            mumps_ordering=_mumps_ordering(cfg, env),
            ksp_type=_scfg(cfg, "ksp_type", env, f"{prefix}_KSP_TYPE", "fgmres"),
            pc_type=_scfg(cfg, "pc_type", env, f"{prefix}_PC_TYPE", "asm"),
            ksp_pc_side=_scfg(cfg, "ksp_pc_side", env, f"{prefix}_KSP_PC_SIDE", "right"),
            ksp_rtol=_scfg_float(cfg, "ksp_rtol", env, f"{prefix}_KSP_RTOL", 1e-10),
            ksp_atol=_scfg_float(cfg, "ksp_atol", env, f"{prefix}_KSP_ATOL", 1e-14),
            ksp_max_it=_scfg_int(cfg, "ksp_max_it", env, f"{prefix}_KSP_MAX_IT", 20000),
            asm_overlap=_scfg_int(cfg, "asm_overlap", env, f"{prefix}_ASM_OVERLAP", 2),
            sub_ksp_type=_scfg(cfg, "sub_ksp_type", env, f"{prefix}_SUB_KSP_TYPE", "preonly"),
            sub_pc_type=_scfg(cfg, "sub_pc_type", env, f"{prefix}_SUB_PC_TYPE", "lu"),
            pc_factor_levels=_scfg_int(
                cfg, "pc_factor_levels", env, f"{prefix}_PC_FACTOR_LEVELS", 0
            ),
            pc_factor_shift_type=_scfg(
                cfg, "pc_factor_shift_type", env, f"{prefix}_PC_FACTOR_SHIFT_TYPE", "nonzero"
            ),
            pc_factor_shift_amount=_scfg_float(
                cfg, "pc_factor_shift_amount", env, f"{prefix}_PC_FACTOR_SHIFT_AMOUNT", 1e-12
            ),
            ksp_monitor=_scfg_bool(cfg, "ksp_monitor", env, f"{prefix}_KSP_MONITOR", False),
        )


@dataclass(frozen=True)
class ResolventEpsCfg:
    eps_tol: float
    eps_max_it: int
    conv_test: str
    true_residual: bool
    monitor: bool
    monitor_every: int
    ncv: int | None
    forcing_norm: str

    @classmethod
    def from_mapping(
        cls, cfg: Mapping[str, Any] | None, env: Mapping[str, str]
    ) -> "ResolventEpsCfg":
        return cls(
            eps_tol=_scfg_float(cfg, "eps_tol", env, "LINSTABPY_RESOLVENT_EPS_TOL", 1e-8),
            eps_max_it=_scfg_int(cfg, "eps_max_it", env, "LINSTABPY_RESOLVENT_EPS_MAX_IT", 1000),
            conv_test=_scfg(cfg, "conv_test", env, "LINSTABPY_RESOLVENT_EPS_CONV_TEST", ""),
            true_residual=_scfg_bool(
                cfg, "true_residual", env, "LINSTABPY_RESOLVENT_EPS_TRUE_RESIDUAL", False
            ),
            monitor=_scfg_bool(cfg, "monitor", env, "LINSTABPY_RESOLVENT_EPS_MONITOR", False),
            monitor_every=_scfg_int(
                cfg, "monitor_every", env, "LINSTABPY_RESOLVENT_EPS_MONITOR_EVERY", 10
            ),
            ncv=int(cfg["ncv"]) if cfg is not None and "ncv" in cfg else None,
            forcing_norm=_scfg(cfg, "forcing_norm", env, "LINSTABPY_RESOLVENT_FORCING_NORM", "chu"),
        )


@dataclass(frozen=True)
class ResolventCfg:
    linear: LinSolveCfg
    eps: ResolventEpsCfg


@dataclass(frozen=True)
class InexactCfg:
    ksp_type: str
    ksp_rtol: float
    ksp_atol: float
    ksp_max_it: int
    pc_type: str
    asm_overlap: int
    sub_pc_type: str
    pc_factor_levels: int

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "InexactCfg":
        d = cfg or {}
        return cls(
            ksp_type=str(d.get("ksp_type", "fgmres")),
            ksp_rtol=float(d.get("ksp_rtol", 1e-8)),
            ksp_atol=float(d.get("ksp_atol", 1e-10)),
            ksp_max_it=int(d.get("ksp_max_it", 200)),
            pc_type=str(d.get("pc_type", "asm")),
            asm_overlap=int(d.get("asm_overlap", 1)),
            sub_pc_type=str(d.get("sub_pc_type", "ilu")),
            pc_factor_levels=int(d.get("pc_factor_levels", 2)),
        )


@dataclass(frozen=True)
class BackendProbeCfg:
    residual_rel_max: float
    enabled: bool | None

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "BackendProbeCfg":
        d = cfg or {}
        return cls(
            residual_rel_max=float(d.get("residual_rel_max", 1.0e-8)),
            enabled=_as_bool(d["enabled"]) if "enabled" in d else None,
        )


@dataclass(frozen=True)
class EigenCfg:
    inexact: InexactCfg
    backend_probe: BackendProbeCfg
    st_solver: str
    st_factor_backend: str | None
    factor_precision: str
    mumps_blr: bool
    mumps_blr_tol: float
    mumps_ordering: str | None
    pc_factor_solver_type: str
    cond_warn: bool
    cond_warn_max_n: int
    cond_warn_threshold: float
    ksp_type: str
    pc_type: str
    ksp_pc_side: str
    ksp_rtol: float
    ksp_max_it: int
    asm_overlap: int
    sub_ksp_type: str
    sub_pc_type: str
    conv_test: str
    eps_tol: float
    residual_error_type: str
    monitor: bool
    monitor_every: int
    true_residual: bool
    extract_all_converged: bool
    weighted_norm_tol: float
    ncv: int | None
    eps_max_it_explicit: int | None
    # Auto-route DOF threshold: carried from config so solve time can read it.
    # "auto" backend intent is resolved to a concrete backend in _eigenvalues.py
    # once ndof is known.  See resolve_auto_factor_backend() and
    # DEFAULT_AUTO_ROUTE_DOF_THRESHOLD for provenance.
    auto_route_dof_threshold: int

    @classmethod
    def from_mapping(
        cls,
        cfg: Mapping[str, Any] | None,
        env: Mapping[str, str],
        solver_default: str | None = None,
    ) -> "EigenCfg":
        st_solver = _scfg(cfg, "st_solver", env, "LINSTABPY_EIGEN_ST_SOLVER", "mumps")
        pc_factor_solver_type = _scfg(
            cfg, "pc_factor_solver_type", env, "LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE", ""
        )
        return cls(
            inexact=InexactCfg.from_mapping(cfg.get("inexact", {}) if cfg else {}),
            backend_probe=BackendProbeCfg.from_mapping(cfg.get("backend_probe", {}) if cfg else {}),
            st_solver=st_solver,
            st_factor_backend=_st_factor_backend(st_solver, cfg, env, solver_default),
            factor_precision=_factor_precision(cfg, env),
            mumps_blr=_mumps_blr(cfg, env),
            mumps_blr_tol=_mumps_blr_tol(cfg, env),
            mumps_ordering=_mumps_ordering(cfg, env),
            pc_factor_solver_type=pc_factor_solver_type,
            cond_warn=_scfg_bool(cfg, "cond_warn", env, "LINSTABPY_EIGEN_COND_WARN", True),
            cond_warn_max_n=_scfg_int(
                cfg, "cond_warn_max_n", env, "LINSTABPY_EIGEN_COND_WARN_MAX_N", 1500
            ),
            cond_warn_threshold=_scfg_float(
                cfg, "cond_warn_threshold", env, "LINSTABPY_EIGEN_COND_WARN_THRESHOLD", 1e12
            ),
            ksp_type=_scfg(cfg, "ksp_type", env, "LINSTABPY_EIGEN_KSP_TYPE", "gmres"),
            pc_type=_scfg(cfg, "pc_type", env, "LINSTABPY_EIGEN_PC_TYPE", "asm"),
            ksp_pc_side=_scfg(cfg, "ksp_pc_side", env, "LINSTABPY_EIGEN_KSP_PC_SIDE", "left"),
            ksp_rtol=_scfg_float(cfg, "ksp_rtol", env, "LINSTABPY_EIGEN_KSP_RTOL", 1e-8),
            ksp_max_it=_scfg_int(cfg, "ksp_max_it", env, "LINSTABPY_EIGEN_KSP_MAX_IT", 20000),
            asm_overlap=_scfg_int(cfg, "asm_overlap", env, "LINSTABPY_EIGEN_ASM_OVERLAP", 2),
            sub_ksp_type=_scfg(cfg, "sub_ksp_type", env, "LINSTABPY_EIGEN_SUB_KSP_TYPE", "preonly"),
            sub_pc_type=_scfg(cfg, "sub_pc_type", env, "LINSTABPY_EIGEN_SUB_PC_TYPE", "lu"),
            conv_test=_scfg(cfg, "conv_test", env, "LINSTABPY_EIGEN_CONV_TEST", ""),
            eps_tol=_scfg_float(cfg, "eps_tol", env, "LINSTABPY_EIGEN_EPS_TOL", 1e-10),
            residual_error_type=_scfg(
                cfg, "residual_error_type", env, "LINSTABPY_EIGEN_RESIDUAL_ERROR_TYPE", "relative"
            ),
            monitor=_scfg_bool(cfg, "monitor", env, "LINSTABPY_EIGEN_MONITOR", False),
            monitor_every=_scfg_int(cfg, "monitor_every", env, "LINSTABPY_EIGEN_MONITOR_EVERY", 10),
            true_residual=_scfg_bool(
                cfg,
                "true_residual",
                env,
                "LINSTABPY_EIGEN_TRUE_RESIDUAL",
                st_solver in {"inexact", "fgmres_lu", "strumpack_fgmres"},
            ),
            extract_all_converged=_scfg_bool(
                cfg, "extract_all_converged", env, "LINSTABPY_EIGEN_EXTRACT_ALL_CONVERGED", False
            ),
            weighted_norm_tol=_scfg_float(
                cfg, "weighted_norm_tol", env, "LINSTABPY_EIGEN_WEIGHTED_NORM_TOL", 1e-12
            ),
            ncv=int(cfg["ncv"]) if cfg is not None and "ncv" in cfg else None,
            eps_max_it_explicit=_scfg_optional_int(
                cfg, "eps_max_it", env, "LINSTABPY_EIGEN_EPS_MAX_IT", None
            ),
            # auto_route_dof_threshold uses env > YAML > default precedence,
            # mirroring _mumps_blr / _mumps_ordering (env first, then YAML).
            auto_route_dof_threshold=_auto_route_dof_threshold(cfg, env),
        )

    def eps_max_it(self, ndof: int) -> int:
        return (
            self.eps_max_it_explicit
            if self.eps_max_it_explicit is not None
            else max(500, 10 * int(ndof))
        )


@dataclass(frozen=True)
class SpatialCfg:
    st_solver: str
    factor_backend: str | None
    factor_precision: str
    mumps_blr: bool
    mumps_blr_tol: float
    mumps_ordering: str | None

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None, env: Mapping[str, str]) -> "SpatialCfg":
        st_solver = _scfg(cfg, "st_solver", env, "LINSTABPY_SPATIAL_ST_SOLVER", "mumps")
        default = "mumps" if st_solver == "mumps" else None
        return cls(
            st_solver=st_solver,
            factor_backend=_factor_backend(cfg, env, default=default),
            factor_precision=_factor_precision(cfg, env),
            mumps_blr=_mumps_blr(cfg, env),
            mumps_blr_tol=_mumps_blr_tol(cfg, env),
            mumps_ordering=_mumps_ordering(cfg, env),
        )


@dataclass(frozen=True)
class MeshCfg:
    assembly_backend: str
    assembly_metrics: bool
    assembly_metrics_output: str | None
    assembly_metrics_dir: str | None
    assembly_metrics_tag: str | None
    assembly_metrics_log_explicit: bool | None

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None, env: Mapping[str, str]) -> "MeshCfg":
        return cls(
            assembly_backend=_scfg(
                cfg, "assembly_backend", env, "LINSTABPY_ASSEMBLY_BACKEND", "scipy"
            ),
            assembly_metrics=_scfg_bool(
                cfg, "assembly_metrics", env, "LINSTABPY_ASSEMBLY_METRICS", False
            ),
            assembly_metrics_output=_raw_cfg_value(
                cfg, "assembly_metrics_output", env, "LINSTABPY_ASSEMBLY_METRICS_OUTPUT"
            ),
            assembly_metrics_dir=_raw_cfg_value(
                cfg, "assembly_metrics_dir", env, "LINSTABPY_ASSEMBLY_METRICS_DIR"
            ),
            assembly_metrics_tag=_raw_cfg_value(
                cfg, "assembly_metrics_tag", env, "LINSTABPY_ASSEMBLY_METRICS_TAG"
            ),
            # Faithful to the real reader: assembly.py uses _scfg_bool (truthy
            # {1,true,yes,on}, False otherwise, never raises) — NOT backends._as_bool
            # (which accepts y/n and raises on garbage). Return None only when the
            # value is set by neither cfg nor env, so the call site can apply the
            # dynamic default (= metrics enabled).
            assembly_metrics_log_explicit=(
                _scfg_bool(
                    cfg, "assembly_metrics_log", env, "LINSTABPY_ASSEMBLY_METRICS_LOG", False
                )
                if (cfg is not None and "assembly_metrics_log" in cfg)
                or "LINSTABPY_ASSEMBLY_METRICS_LOG" in env
                else None
            ),
        )


@dataclass(frozen=True)
class MpiCfg:
    """MPI/PETSc init-time options."""

    petsc_matrix_mode: str
    root_scatter: bool
    mumps_icntl_14: int | None
    mumps_icntl_24: int | None

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None, env: Mapping[str, str]) -> "MpiCfg":
        if cfg is not None and "mumps_icntl_7" in cfg:
            raise ValueError("mumps_icntl_7 is gone. Set mumps_ordering: metis.")
        if "LINSTABPY_PETSC_SET_MUMPS_ICNTL_7" in env:
            raise ValueError(
                "LINSTABPY_PETSC_SET_MUMPS_ICNTL_7 is gone. Set LINSTABPY_MUMPS_ORDERING=metis."
            )
        return cls(
            petsc_matrix_mode=_scfg(
                cfg, "petsc_matrix_mode", env, "LINSTABPY_PETSC_MATRIX_MODE", "auto"
            ),
            root_scatter=_scfg_bool(
                cfg, "root_scatter", env, "LINSTABPY_PETSC_ROOT_SCATTER", False
            ),
            mumps_icntl_14=_scfg_optional_int(
                cfg, "mumps_icntl_14", env, "LINSTABPY_PETSC_MUMPS_ICNTL_14", 100
            ),
            mumps_icntl_24=_scfg_optional_int(
                cfg, "mumps_icntl_24", env, "LINSTABPY_PETSC_MUMPS_ICNTL_24", 1
            ),
        )


@dataclass(frozen=True)
class Config:
    linsolve: LinSolveCfg
    resolvent: ResolventCfg
    eigen: EigenCfg
    spatial: SpatialCfg
    mesh: MeshCfg
    mpi: MpiCfg

    @classmethod
    def load(
        cls,
        yaml_path_or_dict: str | Path | Mapping[str, Any] | None,
        env: Mapping[str, str] = os.environ,
    ) -> "Config":
        if isinstance(yaml_path_or_dict, (str, Path)):
            raw = load_config(yaml_path_or_dict)
        elif yaml_path_or_dict is None:
            raw = {}
        else:
            raw = yaml_path_or_dict
        solver = _solver_mapping(raw)
        linsolve_sub = _subsection(solver, "LINSTABPY_LINSOLVE")
        resolvent_sub = _subsection(solver, "LINSTABPY_RESOLVENT")
        eigen_sub = _subsection(solver, "LINSTABPY_EIGEN")
        # A `factor_backend` at solver level used to be read by NOTHING: every resolver is
        # handed a SUB-config, so the key silently did nothing (once costing a run that was
        # meant to be on GPU and executed entirely on CPU). It now acts as the lowest-priority
        # fallback -- below the per-stage key and below the environment -- so no configuration
        # that resolves today resolves differently.
        solver_default = (
            _normalize_backend(solver.get("factor_backend"))
            if isinstance(solver, Mapping) and "factor_backend" in solver
            else None
        )
        return cls(
            linsolve=LinSolveCfg.from_mapping(
                linsolve_sub, env, "LINSTABPY_LINSOLVE", solver_default
            ),
            resolvent=ResolventCfg(
                linear=LinSolveCfg.from_mapping(
                    resolvent_sub, env, "LINSTABPY_RESOLVENT", solver_default
                ),
                eps=ResolventEpsCfg.from_mapping(_resolvent_eps_subsection(solver), env),
            ),
            eigen=EigenCfg.from_mapping(eigen_sub, env, solver_default),
            spatial=SpatialCfg.from_mapping(_spatial_subsection(solver), env),
            mesh=MeshCfg.from_mapping(solver, env),
            mpi=MpiCfg.from_mapping(solver, env),
        )
