# SPDX-License-Identifier: GPL-3.0-only
"""Sparse-direct factor backend selection and typed registry for PETSc LU paths."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from functools import cache
from typing import Any

import numpy as np
from petsc4py import PETSc

DEFAULT_FACTOR_BACKEND = "mumps"
FACTOR_PRECISION = "double"
ERR_SUP = 92
_PROBE_DENOMINATOR_TINY = 1.0e-300


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


def resolve_factor_backend(
    cfg: Mapping[str, Any] | None = None,
    *,
    env_var: str | None = None,
    default: str | None = DEFAULT_FACTOR_BACKEND,
) -> str | None:
    """Resolve a direct-factor backend from JSONC, env vars, then default."""
    if cfg is not None and "factor_backend" in cfg:
        return _normalize_backend(cfg["factor_backend"])
    if env_var is not None and env_var in os.environ:
        return _normalize_backend(os.environ[env_var])
    if "LINSTABPY_FACTOR_BACKEND" in os.environ:
        return _normalize_backend(os.environ["LINSTABPY_FACTOR_BACKEND"])
    return _normalize_backend(default)


def apply_factor_backend(
    pc: PETSc.PC,
    backend: str | None,
    *,
    options_prefix: str = "",
) -> str | None:
    """Apply ``backend`` to a PCLU after ``pc.setType('lu')``."""
    backend = _normalize_backend(backend)
    if backend is None:
        return None
    ensure_factor_backend_available(backend)
    opts = PETSc.Options()
    key = f"{options_prefix}pc_factor_mat_solver_type"
    opts.setValue(key, backend)
    try:
        pc.setFromOptions()
        pc.setFactorSolverType(backend)
    finally:
        opts.delValue(key)
    return backend


def ensure_factor_backend_available(backend: str) -> None:
    """Raise a clear error if PETSc cannot set up a tiny LU with ``backend``."""
    if not factor_backend_available(backend):
        version = PETSc.Sys.getVersion()
        raise RuntimeError(
            f"PETSc factor backend '{backend}' is not available in this build "
            f"(PETSc {version}). Rebuild PETSc/petsc4py with that sparse-direct "
            "solver enabled, then rerun. See docs/COMPUTE_BACKENDS.md."
        )


def should_probe_backend(
    resolved_backend: str | None,
    probe_cfg: Mapping[str, Any] | None = None,
    *,
    st_solver: str | None = None,
) -> bool:
    """Return whether a resolved factor backend should run the accuracy probe."""
    backend = _normalize_backend(resolved_backend)
    if backend is None:
        return False
    # The fgmres+direct-PC families are inexact outer solves even over the
    # default MUMPS factor backend, so they never take the default-backend
    # probe exemption; an explicit probe_cfg "enabled" still wins either way.
    inexact_outer = str(st_solver or "").strip().lower() in {"fgmres_lu", "strumpack_fgmres"}
    if backend == DEFAULT_FACTOR_BACKEND and not inexact_outer:
        return False
    if probe_cfg is not None and "enabled" in probe_cfg:
        return _as_bool(probe_cfg["enabled"])
    return True


def shifted_ksp_probe_residual(
    ksp: PETSc.KSP,
    *,
    rhs_seed: int = 17,
) -> dict[str, float | int | str | None]:
    """Solve one deterministic shifted-system RHS and return the true residual.

    This is a diagnostic probe for an already-configured shift-invert KSP.  It
    does not know about eigenpairs or acceptance policy; it only measures
    ``||Kx-b|| / ||b||`` for the KSP's current operator using PETSc collective
    vector norms.
    """
    operators = ksp.getOperators()
    A = operators[0]
    b = A.createVecRight()
    x = A.createVecRight()
    r = A.createVecLeft()
    old_nonzero_guess = _safe_call(ksp.getInitialGuessNonzero)
    try:
        _fill_deterministic_rhs(b, rhs_seed=rhs_seed)
        x.set(0)
        ksp.setInitialGuessNonzero(False)
        ksp.solve(b, x)
        A.mult(x, r)
        r.axpy(-1, b)
        residual_abs = float(r.norm())
        rhs_norm = float(b.norm())
        residual_rel = residual_abs / max(rhs_norm, _PROBE_DENOMINATOR_TINY)
        pc = ksp.getPC()
        return {
            "residual_abs": residual_abs,
            "residual_rel": residual_rel,
            "ksp_iterations": int(ksp.getIterationNumber()),
            "ksp_reason": int(ksp.getConvergedReason()),
            "ksp_type": ksp.getType(),
            "pc_type": pc.getType(),
            "pc_factor_solver_type": _safe_call(pc.getFactorSolverType),
        }
    finally:
        if old_nonzero_guess is not None:
            ksp.setInitialGuessNonzero(bool(old_nonzero_guess))
        r.destroy()
        x.destroy()
        b.destroy()


@cache
def factor_backend_available(backend: str) -> bool:
    """Return whether a tiny complex AIJ LU can be set up with ``backend``."""
    backend = _normalize_backend(backend)
    if backend is None:
        return True
    A = PETSc.Mat().createAIJ(size=(1, 1), comm=PETSc.COMM_SELF)
    A.setValue(0, 0, 1.0 + 0.0j)
    A.assemble()
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    try:
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType(backend)
        ksp.setUp()
        return True
    except PETSc.Error as exc:
        if _petsc_error_code(exc) == ERR_SUP:
            return False
        raise
    finally:
        ksp.destroy()
        A.destroy()


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


def _normalize_backend(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "default", "petsc"}:
        return None
    return text


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid backend_probe.enabled={value!r}; expected boolean")


def _fill_deterministic_rhs(vec: PETSc.Vec, *, rhs_seed: int) -> None:
    rstart, rend = vec.getOwnershipRange()
    for row in range(rstart, rend):
        real = np.sin((row + 1) * (rhs_seed + 0.5))
        imag = np.cos((row + 1) * (rhs_seed + 1.5))
        vec.setValue(row, complex(real, imag))
    vec.assemble()


def _safe_call(method):  # noqa: ANN001
    try:
        return method()
    except (AttributeError, PETSc.Error):
        return None


def _petsc_error_code(exc: PETSc.Error) -> int | None:
    return getattr(exc, "ierr", None) or (exc.args[0] if exc.args else None)


__all__ = [
    "Backend",
    "DEFAULT_FACTOR_BACKEND",
    "FACTOR_PRECISION",
    "apply_factor_backend",
    "available",
    "ensure",
    "ensure_factor_backend_available",
    "factor_backend_available",
    "resolve_factor_backend",
    "select",
    "shifted_ksp_probe_residual",
    "should_probe_backend",
]
