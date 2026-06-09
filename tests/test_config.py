# SPDX-License-Identifier: GPL-3.0-only
"""Typed config seam tests against the live legacy config readers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from linstabpy.config import Config
from linstabpy.solvers.backends import resolve_factor_backend
from linstabpy.utils._solver_cfg import (
    SolverConfig,
    _scfg,
    _scfg_bool,
    _scfg_float,
    _scfg_int,
    _scfg_optional_int,
)


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


def test_differential_oracle_defaults(monkeypatch) -> None:
    env_names = [
        "LINSTABPY_LINSOLVE_TYPE",
        "LINSTABPY_LINSOLVE_KSP_RTOL",
        "LINSTABPY_LINSOLVE_KSP_MONITOR",
        "LINSTABPY_RESOLVENT_KSP_TYPE",
        "LINSTABPY_RESOLVENT_EPS_TOL",
        "LINSTABPY_EIGEN_ST_SOLVER",
        "LINSTABPY_EIGEN_COND_WARN",
        "LINSTABPY_EIGEN_NCV",
        "LINSTABPY_SPATIAL_ST_SOLVER",
        "LINSTABPY_ASSEMBLY_BACKEND",
        "LINSTABPY_PETSC_ROOT_SCATTER",
        "LINSTABPY_PETSC_MUMPS_ICNTL_14",
        "LINSTABPY_FACTOR_BACKEND",
    ]
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    typed = Config.load({})
    sc = SolverConfig({})

    assert typed.linsolve.type == _scfg(sc.subsection("LINSTABPY_LINSOLVE"), "type", "LINSTABPY_LINSOLVE_TYPE", "mumps")
    assert typed.linsolve.ksp_rtol == _scfg_float(sc.subsection("LINSTABPY_LINSOLVE"), "ksp_rtol", "LINSTABPY_LINSOLVE_KSP_RTOL", 1e-10)
    assert typed.linsolve.ksp_monitor == _scfg_bool(sc.subsection("LINSTABPY_LINSOLVE"), "ksp_monitor", "LINSTABPY_LINSOLVE_KSP_MONITOR", False)
    assert typed.resolvent.linear.ksp_type == _scfg(sc.subsection("LINSTABPY_RESOLVENT"), "ksp_type", "LINSTABPY_RESOLVENT_KSP_TYPE", "fgmres")
    assert typed.resolvent.eps.eps_tol == _scfg_float({}, "eps_tol", "LINSTABPY_RESOLVENT_EPS_TOL", 1e-8)
    assert typed.eigen.st_solver == sc.eigen_st_solver()
    assert typed.eigen.cond_warn == _scfg_bool(sc.subsection("LINSTABPY_EIGEN"), "cond_warn", "LINSTABPY_EIGEN_COND_WARN", True)
    assert typed.eigen.ncv == _scfg_optional_int(sc.subsection("LINSTABPY_EIGEN"), "ncv", "LINSTABPY_EIGEN_NCV", None)
    spatial = sc.get("spatial", sc.get("eigen", {}))
    assert typed.spatial.st_solver == _scfg(spatial, "st_solver", "LINSTABPY_SPATIAL_ST_SOLVER", "mumps")
    assert typed.mesh.assembly_backend == _scfg(sc, "assembly_backend", "LINSTABPY_ASSEMBLY_BACKEND", "scipy")
    assert typed.mpi.root_scatter == sc.root_scatter_enabled()
    assert typed.mpi.mumps_icntl_14 == _scfg_optional_int(sc, "mumps_icntl_14", "LINSTABPY_PETSC_MUMPS_ICNTL_14", 100)
    assert typed.linsolve.factor_backend == resolve_factor_backend(sc.subsection("LINSTABPY_LINSOLVE"), env_var="LINSTABPY_LINSOLVE_PC_FACTOR_SOLVER_TYPE", default="mumps")


def test_differential_oracle_jsonc_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_LINSOLVE_TYPE", "env-wins-only-without-jsonc")
    raw = {
        "linsolve": {"type": "ITERATIVE", "ksp_rtol": 1e-6, "ksp_monitor": True, "factor_backend": "petsc"},
        "resolvent": {"ksp_type": "gmres", "eigen": {"eps_tol": 2e-7, "ncv": 31}},
        "eigen": {"st_solver": "inexact", "cond_warn": False, "ncv": 41},
        "spatial": {"st_solver": "lu", "factor_backend": "superlu_dist"},
        "assembly_backend": "petsc",
        "root_scatter": True,
        "mumps_icntl_14": "none",
    }
    typed = Config.load(raw)
    sc = SolverConfig(raw)

    assert typed.linsolve.type == _scfg(sc.subsection("LINSTABPY_LINSOLVE"), "type", "LINSTABPY_LINSOLVE_TYPE", "mumps")
    assert typed.linsolve.ksp_rtol == _scfg_float(sc.subsection("LINSTABPY_LINSOLVE"), "ksp_rtol", "LINSTABPY_LINSOLVE_KSP_RTOL", 1e-10)
    assert typed.linsolve.ksp_monitor == _scfg_bool(sc.subsection("LINSTABPY_LINSOLVE"), "ksp_monitor", "LINSTABPY_LINSOLVE_KSP_MONITOR", False)
    assert typed.linsolve.factor_backend == resolve_factor_backend(sc.subsection("LINSTABPY_LINSOLVE"), env_var="LINSTABPY_LINSOLVE_PC_FACTOR_SOLVER_TYPE", default="mumps")
    assert typed.resolvent.linear.ksp_type == _scfg(sc.subsection("LINSTABPY_RESOLVENT"), "ksp_type", "LINSTABPY_RESOLVENT_KSP_TYPE", "fgmres")
    assert typed.resolvent.eps.eps_tol == _scfg_float(raw["resolvent"]["eigen"], "eps_tol", "LINSTABPY_RESOLVENT_EPS_TOL", 1e-8)
    assert typed.resolvent.eps.ncv == 31
    assert typed.eigen.st_solver == sc.eigen_st_solver()
    assert typed.eigen.true_residual == _scfg_bool(sc.subsection("LINSTABPY_EIGEN"), "true_residual", "LINSTABPY_EIGEN_TRUE_RESIDUAL", True)
    assert typed.eigen.ncv == _scfg_optional_int(sc.subsection("LINSTABPY_EIGEN"), "ncv", "LINSTABPY_EIGEN_NCV", None)
    assert typed.spatial.factor_backend == resolve_factor_backend(raw["spatial"], default=None)
    assert typed.mesh.assembly_backend == _scfg(sc, "assembly_backend", "LINSTABPY_ASSEMBLY_BACKEND", "scipy")
    assert typed.mpi.root_scatter == sc.root_scatter_enabled()
    assert typed.mpi.mumps_icntl_14 == _scfg_optional_int(sc, "mumps_icntl_14", "LINSTABPY_PETSC_MUMPS_ICNTL_14", 100)


def test_differential_oracle_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_LINSOLVE_TYPE", "KRYLOV")
    monkeypatch.setenv("LINSTABPY_LINSOLVE_KSP_RTOL", "1e-5")
    monkeypatch.setenv("LINSTABPY_LINSOLVE_KSP_MONITOR", "yes")
    monkeypatch.setenv("LINSTABPY_RESOLVENT_KSP_TYPE", "bcgs")
    monkeypatch.setenv("LINSTABPY_RESOLVENT_EPS_TOL", "3e-7")
    monkeypatch.setenv("LINSTABPY_EIGEN_ST_SOLVER", "inexact")
    monkeypatch.setenv("LINSTABPY_EIGEN_COND_WARN", "off")
    monkeypatch.setenv("LINSTABPY_EIGEN_NCV", "55")
    monkeypatch.setenv("LINSTABPY_SPATIAL_ST_SOLVER", "direct")
    monkeypatch.setenv("LINSTABPY_ASSEMBLY_BACKEND", "petsc")
    monkeypatch.setenv("LINSTABPY_PETSC_ROOT_SCATTER", "on")
    monkeypatch.setenv("LINSTABPY_PETSC_MUMPS_ICNTL_14", "250")
    monkeypatch.setenv("LINSTABPY_FACTOR_BACKEND", "superlu_dist")

    typed = Config.load({})
    assert typed.linsolve.type == _scfg({}, "type", "LINSTABPY_LINSOLVE_TYPE", "mumps")
    assert typed.linsolve.ksp_rtol == _scfg_float({}, "ksp_rtol", "LINSTABPY_LINSOLVE_KSP_RTOL", 1e-10)
    assert typed.linsolve.ksp_monitor == _scfg_bool({}, "ksp_monitor", "LINSTABPY_LINSOLVE_KSP_MONITOR", False)
    assert typed.resolvent.linear.ksp_type == _scfg({}, "ksp_type", "LINSTABPY_RESOLVENT_KSP_TYPE", "fgmres")
    assert typed.resolvent.eps.eps_tol == _scfg_float({}, "eps_tol", "LINSTABPY_RESOLVENT_EPS_TOL", 1e-8)
    assert typed.eigen.st_solver == _scfg({}, "st_solver", "LINSTABPY_EIGEN_ST_SOLVER", "mumps")
    assert typed.eigen.cond_warn == _scfg_bool({}, "cond_warn", "LINSTABPY_EIGEN_COND_WARN", True)
    assert typed.eigen.ncv == _scfg_optional_int({}, "ncv", "LINSTABPY_EIGEN_NCV", None)
    assert typed.spatial.st_solver == _scfg({}, "st_solver", "LINSTABPY_SPATIAL_ST_SOLVER", "mumps")
    assert typed.mesh.assembly_backend == _scfg({}, "assembly_backend", "LINSTABPY_ASSEMBLY_BACKEND", "scipy")
    assert typed.mpi.root_scatter == _scfg_bool({}, "root_scatter", "LINSTABPY_PETSC_ROOT_SCATTER", False)
    assert typed.mpi.mumps_icntl_14 == _scfg_optional_int({}, "mumps_icntl_14", "LINSTABPY_PETSC_MUMPS_ICNTL_14", 100)
    assert typed.linsolve.factor_backend == resolve_factor_backend({}, env_var="LINSTABPY_LINSOLVE_PC_FACTOR_SOLVER_TYPE", default="mumps")


def test_jsonc_beats_env(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_EIGEN_EPS_TOL", "1e-3")
    monkeypatch.setenv("LINSTABPY_RESOLVENT_EPS_NCV", "99")
    monkeypatch.setenv("LINSTABPY_FACTOR_BACKEND", "superlu_dist")
    raw = {"eigen": {"eps_tol": 1e-9}, "resolvent": {"eigen": {"ncv": 25}}, "linsolve": {"factor_backend": "mumps"}}

    typed = Config.load(raw)

    assert typed.eigen.eps_tol == _scfg_float(raw["eigen"], "eps_tol", "LINSTABPY_EIGEN_EPS_TOL", 1e-10)
    assert typed.resolvent.eps.ncv == 25
    assert typed.linsolve.factor_backend == "mumps"
