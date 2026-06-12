# SPDX-License-Identifier: GPL-3.0-only
"""Deprecated resolvent solver import path.

Use :mod:`linstabpy.resolvent.analysis` instead.  This module remains as a
one-minor-version compatibility shim.  The shifted solves used by the public
wrappers read the ``solver.resolvent`` JSONC section, with ``solver.linsolve``
as the compatibility fallback.  ``factor_backend`` selects the PETSc LU factor
backend for direct solves.  Resolution order is JSONC ``factor_backend``, then
``LINSTABPY_RESOLVENT_PC_FACTOR_SOLVER_TYPE``, then
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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only imports keep this PEP-562 shim importable without PETSc;
    # `from __future__ import annotations` defers signature evaluation.
    from collections.abc import Callable

    import numpy as np
    import scipy.sparse as sp
    from petsc4py import PETSc

    from linstabpy.config import Config

_DEPRECATED_EXPORTS = {
    "resolvent_analysis": "resolvent_analysis",
    "compute_resolvent_svd_residuals": "compute_resolvent_svd_residuals",
    "resolvent_analysis_with_residuals": "resolvent_analysis",
}

__all__ = sorted(_DEPRECATED_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily expose deprecated public resolvent wrappers.

    Backend selection for the returned wrappers is documented in the module
    docstring above and in ``docs/COMPUTE_BACKENDS.md``.
    """
    if name not in _DEPRECATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "resolvent_analysis_with_residuals":
        warnings.warn(
            "linstabpy.solvers.resolvent.resolvent_analysis_with_residuals is deprecated; "
            "use linstabpy.resolvent.analysis.resolvent_analysis(..., residuals=True) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _resolvent_analysis_with_residuals

    warnings.warn(
        f"linstabpy.solvers.resolvent.{name} is deprecated; "
        f"use linstabpy.resolvent.analysis.{_DEPRECATED_EXPORTS[name]} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from linstabpy.resolvent.analysis import compute_resolvent_svd_residuals, resolvent_analysis

    exports = {
        "compute_resolvent_svd_residuals": compute_resolvent_svd_residuals,
        "resolvent_analysis": resolvent_analysis,
    }
    return exports[_DEPRECATED_EXPORTS[name]]


def _resolvent_analysis_with_residuals(
    L0: sp.spmatrix | PETSc.Mat,
    omega: float,
    n_modes: int,
    W_diag: np.ndarray,
    B_mat: sp.spmatrix | PETSc.Mat | None = None,
    C_mat: sp.spmatrix | PETSc.Mat | None = None,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
    *,
    gamma: float = 0.0,
    return_info: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]
):
    """Deprecated wrapper for ``resolvent_analysis(..., residuals=True)``.

    Direct shifted solves use ``solver.resolvent.factor_backend`` (or the
    ``solver.linsolve`` fallback), then
    ``LINSTABPY_RESOLVENT_PC_FACTOR_SOLVER_TYPE``, then
    ``LINSTABPY_FACTOR_BACKEND``, then the ``mumps`` default.  Availability is
    probe-checked and unavailable backends fail loudly.  Experimental
    precision, MUMPS BLR, and MUMPS ordering knobs are enforced from
    environment variables only at factorization time today; matching JSONC
    fields are carried but not consumed.  See ``docs/COMPUTE_BACKENDS.md``.
    """
    warnings.warn(
        "linstabpy.solvers.resolvent.resolvent_analysis_with_residuals is deprecated; "
        "use linstabpy.resolvent.analysis.resolvent_analysis(..., residuals=True) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from linstabpy.resolvent.analysis import resolvent_analysis

    return resolvent_analysis(
        L0,
        omega,
        n_modes,
        W_diag,
        B_mat=B_mat,
        C_mat=C_mat,
        filter_fn=filter_fn,
        filter_ct_fn=filter_ct_fn,
        solver_cfg=solver_cfg,
        cfg=cfg,
        residuals=True,
        gamma=gamma,
        return_info=return_info,
    )
