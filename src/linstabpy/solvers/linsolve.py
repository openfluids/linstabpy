# SPDX-License-Identifier: GPL-3.0-only
"""Deprecated linear-solver import path.

Use :mod:`linstabpy.linalg.factorize` instead.  This module remains as a
one-minor-version compatibility shim.  The public wrappers read the
``solver.linsolve`` JSONC section.  ``factor_backend`` selects the PETSc LU
factor backend for direct solves.  Resolution order is JSONC
``factor_backend``, then ``LINSTABPY_LINSOLVE_PC_FACTOR_SOLVER_TYPE``, then
``LINSTABPY_FACTOR_BACKEND``, then the call-site default (``mumps`` for the
reference direct path).  Requested backends are checked with a small complex
PETSc LU probe and an unavailable backend fails loudly instead of falling back.
See ``docs/COMPUTE_BACKENDS.md`` for the backend contract and current evidence.

Current backend status is deliberately conservative: ``mumps`` is the default
and reference path; ``strumpack`` is experimental and currently documented only
for the diagnostic FGMRES+LU configuration; ``superlu_dist`` is rejected for the
tested CUDA build; and ``cudss`` is not wired to the ``linstabpy`` PETSc factor
interface.

The experimental factorization knobs are environment-enforced today:
``LINSTABPY_FACTOR_PRECISION`` (``single`` only with ``superlu_dist``),
``LINSTABPY_MUMPS_BLR`` with ``LINSTABPY_MUMPS_BLR_TOL``, and
``LINSTABPY_MUMPS_ORDERING``.  They are off or unset by default, fail loudly
when requested with the wrong backend, and make no performance claim.  Matching
JSONC fields are validated and carried in the typed config but are not yet
consumed by the solver call sites for factorization setup.
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
    """Lazily expose deprecated public linear-solver wrappers.

    Backend selection for the returned wrappers is documented in the module
    docstring above and in ``docs/COMPUTE_BACKENDS.md``.
    """
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
