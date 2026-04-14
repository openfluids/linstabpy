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


def _scfg_bool(cfg: dict | None, key: str, env_var: str, default: bool) -> bool:
    if cfg is not None and key in cfg:
        v = cfg[key]
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "on"}
    raw = os.getenv(env_var, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}
