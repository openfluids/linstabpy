# SPDX-License-Identifier: GPL-3.0-only
"""Deprecated shim for :mod:`linstabpy.stability.acceptance`."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_NEW_MODULE = "linstabpy.stability.acceptance"

_EXPORTS = {
    "_postcheck_target_distance",
    "check_conjugate_pair_completeness",
    "evaluate_stability_acceptance",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        warnings.warn(
            f"linstabpy.utils.stability_acceptance.{name} is deprecated; use {_NEW_MODULE}.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        attr = getattr(importlib.import_module(_NEW_MODULE), name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
