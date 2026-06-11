# SPDX-License-Identifier: GPL-3.0-only
"""Linear algebra seams for reusable factorizations."""

from __future__ import annotations

from .backends import Backend, available, ensure, select

_LAZY_EXPORTS = {
    "Factorization": "linstabpy.linalg.factorize",
    "factorize": "linstabpy.linalg.factorize",
    "shifted_operator": "linstabpy.linalg.shifted",
}

__all__ = [
    "Backend",
    "Factorization",
    "available",
    "ensure",
    "factorize",
    "select",
    "shifted_operator",
]


def __getattr__(name: str):  # noqa: ANN201
    if name in _LAZY_EXPORTS:
        import importlib

        attr = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
