# SPDX-License-Identifier: GPL-3.0-only
"""Sparse-direct factor backend selection for PETSc LU paths."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from petsc4py import PETSc

DEFAULT_FACTOR_BACKEND = "mumps"
FACTOR_PRECISION = "double"
ERR_SUP = 92


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


@lru_cache(maxsize=None)
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


def _normalize_backend(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "default", "petsc"}:
        return None
    return text


def _petsc_error_code(exc: PETSc.Error) -> int | None:
    return getattr(exc, "ierr", None) or (exc.args[0] if exc.args else None)
