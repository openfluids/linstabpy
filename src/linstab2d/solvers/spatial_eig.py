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

from .petsc_utils import require_complex, scipy_to_petsc, vec_to_numpy
from ..utils._solver_cfg import _scfg

logger = logging.getLogger(__name__)


def _resolve_ncv(
    *,
    ndof: int,
    n_modes: int,
    default_ncv: int,
    requested_ncv: int | None,
) -> int:
    """Choose a valid Arnoldi subspace size for SLEPc EPS."""
    max_ncv = max(1, ndof - 1)
    ncv = default_ncv if requested_ncv is None else int(requested_ncv)
    return min(max(ncv, n_modes + 1), max_ncv)


def _eigs_generalized(
    Lc: sp.spmatrix,
    R: sp.spmatrix,
    n_modes: int,
    target: float | complex,
    *,
    ncv: int | None = None,
    solver_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``A x = lambda R x`` near ``target`` via SLEPc Krylov-Schur."""
    if n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")

    A_sp = sp.csc_matrix(Lc, dtype=complex)
    B_sp = sp.csc_matrix(R, dtype=complex)
    if A_sp.shape[0] != A_sp.shape[1] or B_sp.shape != A_sp.shape:
        raise ValueError("Lc and R must have identical square shapes")
    if n_modes > A_sp.shape[0]:
        raise ValueError("n_modes cannot exceed matrix size")

    ndof = A_sp.shape[0]

    if ndof <= 2 or n_modes >= ndof - 1:
        # Use QZ-based generalized eigensolver (not B⁻¹A) because B is
        # often singular in spatial problems — boundary-condition rows and
        # algebraic constraints introduce zero rows in B.  QZ handles this
        # natively, returning inf eigenvalues for the null subspace.
        dense = np.asarray(A_sp.toarray())
        dense_R = np.asarray(B_sp.toarray())
        eigvals, eigvecs = scipy.linalg.eig(dense, dense_R)
        order = np.argsort(np.abs(eigvals - target))
        sel = order[:n_modes]
        return eigvals[sel], eigvecs[:, sel]

    require_complex(solver_cfg)
    A = scipy_to_petsc(A_sp, solver_cfg)
    B = scipy_to_petsc(B_sp, solver_cfg)

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
    _scfg_spatial = solver_cfg.get("spatial", solver_cfg.get("eigen", {})) if solver_cfg is not None else None
    st_solver = _scfg(_scfg_spatial, "st_solver", "LINSTAB2D_SPATIAL_ST_SOLVER", "mumps")
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
            pc.setFactorSolverType("mumps")

    eps.setUp()

    logger.info(
        "  SLEPc spatial EPS: sigma = %.6g%+.6gj, nev=%d ...",
        np.real(target), np.imag(target), n_modes,
    )
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError(
            f"SLEPc EPS did not converge for any eigenpair (target={target})"
        )
    if nconv < n_modes:
        logger.warning(
            "SLEPc spatial EPS converged %d/%d eigenpairs; returning partial result",
            nconv, n_modes,
        )

    n_extract = min(nconv, n_modes)
    eigenvalues = np.empty(n_extract, dtype=complex)
    eigenvectors = np.empty((ndof, n_extract), dtype=complex)

    vr = A.createVecRight()
    vi = A.createVecRight()
    for i in range(n_extract):
        eig = eps.getEigenpair(i, vr, vi)
        eigenvalues[i] = eig
        eigenvectors[:, i] = vec_to_numpy(vr)

    return eigenvalues, eigenvectors


def spatial_eigenvalues(
    Lc: sp.spmatrix,
    Lw: sp.spmatrix,
    R: sp.spmatrix,
    omega: float | complex,
    n_modes: int,
    target: float | complex,
    ncv: int | None = None,
    solver_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the spatial EVP `(Lc + omega Lw) @ v = alpha R @ v`.

    Returns:
        alpha : ndarray, shape (n_modes,)
            Spatial eigenvalues.
        modes : ndarray, shape (N, n_modes)
            Right eigenvectors.
    """
    eigenvalues, eigenvectors = _eigs_generalized(
        Lc + omega * Lw,
        R,
        n_modes,
        target,
        ncv=ncv,
        solver_cfg=solver_cfg,
    )
    return eigenvalues, eigenvectors
