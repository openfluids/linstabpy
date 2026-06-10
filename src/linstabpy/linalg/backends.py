# SPDX-License-Identifier: GPL-3.0-only
"""Typed sparse-direct factor backend registry."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from linstabpy.solvers.backends import (
    ensure_factor_backend_available,
    factor_backend_available,
    resolve_factor_backend,
)


class Backend(StrEnum):
    """Known PETSc sparse-direct factor backends."""

    MUMPS = "mumps"
    SUPERLU = "superlu"
    SUPERLU_DIST = "superlu_dist"
    STRUMPACK = "strumpack"
    CUDSS = "cudss"

    @classmethod
    def _missing_(cls, value: object) -> Backend:
        text = str(value).strip().lower()
        backend = str.__new__(cls, text)
        backend._name_ = text.upper()
        backend._value_ = text
        return backend


def select(
    cfg: Mapping[str, Any] | None = None,
    *,
    env_var: str | None = None,
    default: str | Backend | None = Backend.MUMPS,
) -> Backend | None:
    """Resolve the configured factor backend as a typed registry entry."""
    resolved = resolve_factor_backend(cfg, env_var=env_var, default=default)
    if resolved is None:
        return None
    return Backend(resolved)


def ensure(backend: str | Backend | None) -> None:
    """Raise if ``backend`` is not available in this PETSc build."""
    if backend is not None:
        ensure_factor_backend_available(str(backend))


def available() -> dict[Backend, bool]:
    """Return availability for known registry backends."""
    return {backend: factor_backend_available(str(backend)) for backend in Backend}


__all__ = ["Backend", "available", "ensure", "select"]
