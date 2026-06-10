# SPDX-License-Identifier: GPL-3.0-only
"""Solver configuration policy shared by PETSc/SLEPc callers.

The precedence is intentionally simple: JSONC values win, then environment
variables, then the project default.  Keeping that rule here avoids small
differences between eigenvalue, resolvent, and linear-solve paths.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

# Public JSONC still accepts the historical names.  These sets are the policy
# boundary used by solver modules; adding a new family should happen here first.
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
        """Return the solver subsection selected by an environment-prefix family.

        Resolvent solves reuse the linear-solver policy by default, but may
        override it in ``solver.resolvent``.  This preserves old JSONC files
        while giving resolvent sweeps their own tuning surface.
        """
        if env_prefix == "LINSTABPY_RESOLVENT":
            return self.get("resolvent", self.get("linsolve", {}))
        if env_prefix == "LINSTABPY_EIGEN":
            return self.get("eigen", {})
        return self.get("linsolve", {})


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
