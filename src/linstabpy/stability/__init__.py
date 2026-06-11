# SPDX-License-Identifier: GPL-3.0-only
"""Stability analysis utilities."""

from __future__ import annotations

from functools import wraps
from typing import Any

_LAZY_EXPORTS = {
    "eigenvalues": "linstabpy.stability._eigenvalues",
    "adjoint_eigenvalues": "linstabpy.stability._eigenvalues",
    "spatial_eigenvalues": "linstabpy.stability._eigenvalues",
    "_solve_near_target": "linstabpy.stability._eigenvalues",
}

__all__ = sorted(name for name in _LAZY_EXPORTS if not name.startswith("_"))


def __getattr__(name: str):  # noqa: ANN201
    if name in _LAZY_EXPORTS:
        import importlib

        def _resolve() -> Any:
            return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)

        @wraps(_resolve)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            return _resolve()(*args, **kwargs)

        _wrapper.__name__ = name
        _wrapper.__module__ = __name__
        globals()[name] = _wrapper
        return _wrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
