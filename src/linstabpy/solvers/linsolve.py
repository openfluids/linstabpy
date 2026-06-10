# SPDX-License-Identifier: GPL-3.0-only
"""Deprecated linear-solver import path.

Use :mod:`linstabpy.linalg.factorize` instead.  This module remains as a
one-minor-version compatibility shim.
"""

from __future__ import annotations

import warnings
from typing import Any

_DEPRECATED_EXPORTS = {
    "Factorization": "Factorization",
    "LinearSolver": "Factorization",
    "factorize": "factorize",
    "make_linear_solver": "factorize",
}

__all__ = sorted(_DEPRECATED_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _DEPRECATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"linstabpy.solvers.linsolve.{name} is deprecated; "
        f"use linstabpy.linalg.factorize.{_DEPRECATED_EXPORTS[name]} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from linstabpy.linalg.factorize import Factorization, factorize

    exports = {"Factorization": Factorization, "factorize": factorize}
    return exports[_DEPRECATED_EXPORTS[name]]
