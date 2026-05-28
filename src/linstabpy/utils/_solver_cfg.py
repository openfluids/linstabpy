# SPDX-License-Identifier: GPL-3.0-only
"""Solver-config lookup: JSONC dict -> env var -> default."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

DIRECT_SOLVER_TYPES = frozenset({"mumps", "direct", "lu"})
ITERATIVE_LINEAR_SOLVER_TYPES = frozenset({"iterative", "krylov", "petsc"})
ITERATIVE_EIGEN_SOLVER_TYPES = frozenset({"iterative", "krylov", "slepc"})
INEXACT_EIGEN_SOLVER_TYPES = frozenset({"inexact"})


@dataclass(frozen=True)
class SolverConfig(Mapping[str, Any]):
    """Typed view over the legacy ``solver_cfg`` mapping.

    The public JSONC surface is still a plain dictionary.  This wrapper gives
    solver-policy code one place to resolve subsections and env fallbacks while
    preserving the existing ``dict`` methods that current callers use.
    """

    _data: Mapping[str, Any]

    @classmethod
    def from_config(cls, cfg: "SolverConfig | Mapping[str, Any] | None") -> "SolverConfig | None":
        if cfg is None:
            return None
        if isinstance(cfg, SolverConfig):
            return cfg
        return cls(cfg)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def subsection(self, env_prefix: str) -> Mapping[str, Any] | None:
        """Return the current solver subsection selected by an env-prefix family."""
        if env_prefix == "LINSTABPY_RESOLVENT":
            return self.get("resolvent", self.get("linsolve", {}))
        if env_prefix == "LINSTABPY_EIGEN":
            return self.get("eigen", {})
        return self.get("linsolve", {})

    def linear_solver_type(self, env_prefix: str) -> str:
        """Resolve the KSP solver family for a linsolve/resolvent prefix."""
        return self.setting_from_subsection(env_prefix, "type", f"{env_prefix}_TYPE", "mumps")

    def eigen_st_solver(self) -> str:
        """Resolve the SLEPc shift-transform backend family."""
        return self.setting_from_subsection(
            "LINSTABPY_EIGEN",
            "st_solver",
            "LINSTABPY_EIGEN_ST_SOLVER",
            "mumps",
        )

    def root_scatter_enabled(self) -> bool:
        """Resolve the top-level PETSc root-scatter policy."""
        return self.setting_bool("root_scatter", "LINSTABPY_PETSC_ROOT_SCATTER", False)

    def setting(self, key: str, env_var: str, default: str) -> str:
        return _scfg(self, key, env_var, default)

    def setting_from_subsection(
        self,
        env_prefix: str,
        key: str,
        env_var: str,
        default: str,
    ) -> str:
        return _scfg(self.subsection(env_prefix), key, env_var, default)

    def setting_bool(self, key: str, env_var: str, default: bool) -> bool:
        return _scfg_bool(self, key, env_var, default)

    def setting_float(self, key: str, env_var: str, default: float) -> float:
        return _scfg_float(self, key, env_var, default)

    def setting_int(self, key: str, env_var: str, default: int) -> int:
        return _scfg_int(self, key, env_var, default)


def _scfg(cfg: Mapping[str, Any] | None, key: str, env_var: str, default: str) -> str:
    """Read solver setting from dict, else env var, else default."""
    if cfg is not None and key in cfg:
        return str(cfg[key]).strip().lower()
    return os.getenv(env_var, default).strip().lower()


def _scfg_float(cfg: Mapping[str, Any] | None, key: str, env_var: str, default: float) -> float:
    if cfg is not None and key in cfg:
        return float(cfg[key])
    return float(os.getenv(env_var, str(default)))


def _scfg_int(cfg: Mapping[str, Any] | None, key: str, env_var: str, default: int) -> int:
    if cfg is not None and key in cfg:
        return int(cfg[key])
    return int(os.getenv(env_var, str(default)))


def _parse_optional_int(value, name: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "default", "petsc"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {name}: {value!r}") from exc


def _scfg_optional_int(
    cfg: Mapping[str, Any] | None,
    key: str,
    env_var: str,
    default: int | None,
) -> int | None:
    if cfg is not None and key in cfg:
        return _parse_optional_int(cfg[key], key)
    if env_var in os.environ:
        return _parse_optional_int(os.environ[env_var], env_var)
    return default


def _scfg_bool(cfg: Mapping[str, Any] | None, key: str, env_var: str, default: bool) -> bool:
    if cfg is not None and key in cfg:
        v = cfg[key]
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "on"}
    raw = os.getenv(env_var, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _mumps_petsc_options(cfg: Mapping[str, Any] | None) -> dict[str, str]:
    """Return PETSc option-database defaults for MUMPS factorization.

    ``None``/``"none"``/``"petsc"`` leaves a setting to PETSc/MUMPS native
    defaults. ICNTL(7) keeps its historical opt-in gate because changing the
    ordering can change factorization behavior; ICNTL(14) and ICNTL(24) are
    hardened by default.
    """
    options: dict[str, str] = {}

    if _scfg_bool(cfg, "mumps_icntl_7", "LINSTABPY_PETSC_SET_MUMPS_ICNTL_7", False):
        options["mat_mumps_icntl_7"] = "5"  # METIS ordering

    mumps_icntl_14 = _scfg_optional_int(
        cfg,
        "mumps_icntl_14",
        "LINSTABPY_PETSC_MUMPS_ICNTL_14",
        100,
    )
    if mumps_icntl_14 is not None:
        options["mat_mumps_icntl_14"] = str(mumps_icntl_14)

    mumps_icntl_24 = _scfg_optional_int(
        cfg,
        "mumps_icntl_24",
        "LINSTABPY_PETSC_MUMPS_ICNTL_24",
        1,
    )
    if mumps_icntl_24 is not None:
        options["mat_mumps_icntl_24"] = str(mumps_icntl_24)

    return options
