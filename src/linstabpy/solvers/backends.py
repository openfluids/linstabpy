# SPDX-License-Identifier: GPL-3.0-only
"""Deprecated sparse-direct factor backend import path.

Use :mod:`linstabpy.linalg.backends` instead.  This module remains as a
one-minor-version compatibility shim.
"""

from __future__ import annotations

import warnings
from typing import Any

_DEPRECATED_EXPORTS = {
    "Backend": "Backend",
    "DEFAULT_FACTOR_BACKEND": "DEFAULT_FACTOR_BACKEND",
    "FACTOR_PRECISION": "FACTOR_PRECISION",
    "apply_factor_backend": "apply_factor_backend",
    "available": "available",
    "ensure": "ensure",
    "ensure_factor_backend_available": "ensure_factor_backend_available",
    "factor_backend_available": "factor_backend_available",
    "resolve_factor_backend": "resolve_factor_backend",
    "select": "select",
    "shifted_ksp_probe_residual": "shifted_ksp_probe_residual",
    "should_probe_backend": "should_probe_backend",
    "_as_bool": "_as_bool",
    "_normalize_backend": "_normalize_backend",
}

__all__ = sorted(_DEPRECATED_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _DEPRECATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"linstabpy.solvers.backends.{name} is deprecated; "
        f"use linstabpy.linalg.backends.{_DEPRECATED_EXPORTS[name]} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from linstabpy import linalg

    return getattr(linalg.backends, _DEPRECATED_EXPORTS[name])
