# SPDX-License-Identifier: GPL-3.0-only
"""Generalized eigensolvers for spatial stability using SLEPc EPS."""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
from petsc4py import PETSc
from slepc4py import SLEPc

from linstabpy.config import Config
from linstabpy.linalg.backends import apply_factor_backend

from .eigenvalue import _resolve_ncv
from .petsc_utils import (
    _is_petsc_matrix,
    _operator_shape,
    petsc_to_scipy,
    require_complex,
    scipy_to_petsc,
    vec_to_numpy,
)

logger = logging.getLogger(__name__)


def _to_dense_generalized_pair(
    A: sp.spmatrix | PETSc.Mat,
    B: sp.spmatrix | PETSc.Mat,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a generalized eigenpair matrix pair to dense arrays."""
    if _is_petsc_matrix(A):
        A_sp = petsc_to_scipy(A, format="csc")
    else:
        A_sp = sp.csc_matrix(A, dtype=complex)
    if _is_petsc_matrix(B):
        B_sp = petsc_to_scipy(B, format="csc")
    else:
        B_sp = sp.csc_matrix(B, dtype=complex)
    return np.asarray(A_sp.toarray()), np.asarray(B_sp.toarray())


def _eigs_generalized(
    Lc: sp.spmatrix | PETSc.Mat,
    R: sp.spmatrix | PETSc.Mat,
    n_modes: int,
    target: float | complex,
    *,
    ncv: int | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``A x = lambda R x`` near ``target`` via SLEPc Krylov-Schur."""
    if n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")

    a_shape = _operator_shape(Lc)
    b_shape = _operator_shape(R)
    if a_shape[0] != a_shape[1] or b_shape != a_shape:
        raise ValueError("Lc and R must have identical square shapes")
    if n_modes > a_shape[0]:
        raise ValueError("n_modes cannot exceed matrix size")

    ndof = a_shape[0]

    if ndof <= 2 or n_modes >= ndof - 1:
        # Use QZ-based generalized eigensolver (not B⁻¹A) because B is
        # often singular in spatial problems — boundary-condition rows and
        # algebraic constraints introduce zero rows in B.  QZ handles this
        # natively, returning inf eigenvalues for the null subspace.
        dense, dense_R = _to_dense_generalized_pair(Lc, R)
        eigvals, eigvecs = scipy.linalg.eig(dense, dense_R)
        order = np.argsort(np.abs(eigvals - target))
        sel = order[:n_modes]
        return eigvals[sel], eigvecs[:, sel]

    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    require_complex(solver_cfg)
    A = (
        Lc.copy()
        if _is_petsc_matrix(Lc)
        else scipy_to_petsc(sp.csc_matrix(Lc, dtype=complex), solver_cfg)
    )
    B = (
        R.copy()
        if _is_petsc_matrix(R)
        else scipy_to_petsc(sp.csc_matrix(R, dtype=complex), solver_cfg)
    )

    eps = SLEPc.EPS().create()
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    default_ncv = min(max(4 * n_modes, 30), ndof - 1)
    ncv = _resolve_ncv(
        ndof=ndof,
        n_modes=n_modes,
        default_ncv=default_ncv,
        requested_ncv=ncv,
    )
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setTolerances(tol=1e-10, max_it=max(500, 10 * ndof))

    # Shift-invert spectral transformation (MUMPS by default, configurable).
    st_solver = cfg.spatial.st_solver
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    # Mirror the temporal solver and set the shift explicitly on ST so the
    # generalized shift-invert center is not left implicit.
    st.setShift(target)
    ksp = st.getKSP()
    if st_solver in {"mumps", "direct", "lu"}:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        if st_solver == "mumps":
            apply_factor_backend(
                pc,
                cfg.spatial.factor_backend,
                options_prefix=ksp.getOptionsPrefix() or "",
            )
        else:
            apply_factor_backend(
                pc,
                cfg.spatial.factor_backend,
                options_prefix=ksp.getOptionsPrefix() or "",
            )

    eps.setUp()

    logger.info(
        "  SLEPc spatial EPS: sigma = %.6g%+.6gj, nev=%d ...",
        np.real(target),
        np.imag(target),
        n_modes,
    )
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError(f"SLEPc EPS did not converge for any eigenpair (target={target})")
    if nconv < n_modes:
        logger.warning(
            "SLEPc spatial EPS converged %d/%d eigenpairs; returning partial result",
            nconv,
            n_modes,
        )

    n_extract = min(nconv, n_modes)
    eigenvalues = np.empty(n_extract, dtype=complex)
    eigenvectors = np.empty((ndof, n_extract), dtype=complex)

    vr = A.createVecRight()
    vi = A.createVecRight()
    try:
        for i in range(n_extract):
            eig = eps.getEigenpair(i, vr, vi)
            eigenvalues[i] = eig
            eigenvectors[:, i] = vec_to_numpy(vr)
    finally:
        vr.destroy()
        vi.destroy()
        eps.destroy()
        A.destroy()
        B.destroy()

    return eigenvalues, eigenvectors


_DEPRECATED_EXPORTS = {
    "spatial_eigenvalues": ("linstabpy.stability._eigenvalues", "spatial_eigenvalues"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _DEPRECATED_EXPORTS:
        import importlib
        import warnings

        module_name, attr_name = _DEPRECATED_EXPORTS[name]
        warnings.warn(
            f"linstabpy.solvers.spatial_eig.{name} is deprecated; "
            f"use linstabpy.stability.{attr_name}",
            DeprecationWarning,
            stacklevel=2,
        )
        attr = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEPRECATED_EXPORTS))
