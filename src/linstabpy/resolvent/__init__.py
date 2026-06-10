# SPDX-License-Identifier: GPL-3.0-only
"""Resolvent operator utilities."""

from __future__ import annotations

_LAZY_EXPORTS = {
    "ResolventOperator": "linstabpy.resolvent.operator",
    "WeightedNormalOperator": "linstabpy.resolvent.gain",
    "resolvent_analysis": "linstabpy.resolvent.analysis",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):  # noqa: ANN201
    if name in _LAZY_EXPORTS:
        import importlib

        attr = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
