# SPDX-License-Identifier: GPL-3.0-only
"""Solver-config lookup: JSONC dict -> env var -> default."""

from __future__ import annotations

import os


def _scfg(cfg: dict | None, key: str, env_var: str, default: str) -> str:
    """Read solver setting from dict, else env var, else default."""
    if cfg is not None and key in cfg:
        return str(cfg[key]).strip().lower()
    return os.getenv(env_var, default).strip().lower()


def _scfg_float(cfg: dict | None, key: str, env_var: str, default: float) -> float:
    if cfg is not None and key in cfg:
        return float(cfg[key])
    return float(os.getenv(env_var, str(default)))


def _scfg_int(cfg: dict | None, key: str, env_var: str, default: int) -> int:
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
    cfg: dict | None,
    key: str,
    env_var: str,
    default: int | None,
) -> int | None:
    if cfg is not None and key in cfg:
        return _parse_optional_int(cfg[key], key)
    if env_var in os.environ:
        return _parse_optional_int(os.environ[env_var], env_var)
    return default


def _scfg_bool(cfg: dict | None, key: str, env_var: str, default: bool) -> bool:
    if cfg is not None and key in cfg:
        v = cfg[key]
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "on"}
    raw = os.getenv(env_var, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _mumps_petsc_options(cfg: dict | None) -> dict[str, str]:
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
