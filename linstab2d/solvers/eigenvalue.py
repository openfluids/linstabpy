"""Temporal and adjoint eigenvalue solvers using SLEPc EPS."""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from petsc4py import PETSc
from slepc4py import SLEPc

from .petsc_utils import require_complex, scipy_to_petsc, vec_to_numpy

logger = logging.getLogger(__name__)


def _check_inputs_shape(operator: sp.spmatrix, n_modes: int) -> tuple[sp.csc_matrix, int]:
    """Validate input dimensions for eigensolvers and return CSC copy."""
    if n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("L0 must be a square matrix")
    if n_modes > operator.shape[0]:
        raise ValueError("n_modes cannot exceed matrix size")
    return sp.csc_matrix(operator, dtype=complex), operator.shape[0]


def _normalize_in_weighted_inner(
    vectors: np.ndarray,
    W_diag: np.ndarray | None,
) -> np.ndarray:
    """Normalise columns in the weighted inner product using ``W_diag``."""
    if W_diag is None:
        return vectors

    W = np.asarray(W_diag)
    if W.ndim != 1:
        raise ValueError("W_diag must be one-dimensional")
    if vectors.shape[0] != W.size:
        raise ValueError("W_diag length must match eigenvector length")
    if np.any(W == 0):
        raise ValueError("W_diag contains zero entries; cannot normalise")

    normed = np.array(vectors, dtype=complex, copy=True)
    for i in range(vectors.shape[1]):
        v = vectors[:, i]
        norm_sq = np.real(np.vdot(v, W * v))
        if norm_sq <= 0.0:
            continue
        normed[:, i] = v / np.sqrt(norm_sq)
    return normed


def _sort_nearest(values: np.ndarray, vectors: np.ndarray, target: complex) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.abs(values - target))
    return values[order], vectors[:, order]


def _eigs_near_target(
    operator: sp.csc_matrix,
    n_modes: int,
    target: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a shift-invert EVP near ``target`` using SLEPc Krylov-Schur."""
    require_complex()
    ndof = operator.shape[0]

    if ndof <= 2 or n_modes >= ndof - 1:
        dense = np.asarray(operator.toarray())
        all_vals, all_vecs = np.linalg.eig(dense)
        order = np.argsort(np.abs(all_vals - target))
        sel = order[:n_modes]
        return all_vals[sel], all_vecs[:, sel]

    A = scipy_to_petsc(operator)

    eps = SLEPc.EPS().create()
    eps.setOperators(A)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    ncv = min(max(4 * n_modes, 30), ndof - 1)
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTolerances(tol=1e-10, max_it=max(500, 10 * ndof))

    # Shift-invert spectral transformation with MUMPS
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    eps.setUp()

    logger.info(
        "  SLEPc EPS: sigma = %.6g%+.6gj, nev=%d ...",
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
            "SLEPc EPS converged %d/%d eigenpairs; returning partial result",
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


def temporal_eigenvalues(
    L0: sp.spmatrix,
    n_modes: int,
    target: float | complex,
    W_diag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute temporal eigenvalues ``omega = eig(i*L0)`` near a target.

    The physical PDE is ``dq/dt = -L0 q``, so ``eig(L0) = iω_OS`` and
    ``eig(i*L0) = -ω_OS``.  Unstable modes have ``Im(omega) < 0``.

    Parameters
    ----------
    L0:
        State matrix from ``build_lnse_operator``.
    n_modes:
        Number of eigenpairs requested.
    target:
        Target eigenvalue for the shift-invert search.
    W_diag:
        Optional weight vector for normalising eigenvectors in ``v^H W v``.

    Returns
    -------
    omega:
        Temporal eigenvalues ``omega = eig(i*L0) = -ω_OS``.
    modes:
        Right eigenvectors corresponding to ``omega``.
    """
    operator, _ = _check_inputs_shape(L0, n_modes)

    temporal_operator = 1j * operator
    omega = complex(target)
    eigenvalues, eigenvectors = _eigs_near_target(temporal_operator, n_modes, omega)
    eigenvalues, eigenvectors = _sort_nearest(eigenvalues, eigenvectors, omega)

    if W_diag is not None:
        eigenvectors = _normalize_in_weighted_inner(eigenvectors, np.asarray(W_diag))

    return eigenvalues, eigenvectors


def _build_adjoint_operator(
    L0: sp.csc_matrix,
    W_diag: np.ndarray,
    invW_diag: np.ndarray,
) -> sp.csc_matrix:
    """Build the W-weighted adjoint operator ``W^{-1} L0^* W``."""
    W = np.asarray(W_diag)
    invW = np.asarray(invW_diag)
    if W.shape != invW.shape:
        raise ValueError("W_diag and invW_diag must be the same shape")
    if W.ndim != 1:
        raise ValueError("W_diag and invW_diag must be one-dimensional")
    if W.size != L0.shape[0]:
        raise ValueError("W vector lengths must match L0 shape")
    if np.any(invW == 0):
        raise ValueError("invW_diag contains zero entries")

    return sp.diags(invW, format="csc") @ (L0.conj().T @ sp.diags(W, format="csc"))


def adjoint_eigenvalues(
    L0: sp.spmatrix,
    n_modes: int,
    target: float | complex,
    W_diag: np.ndarray,
    invW_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute adjoint temporal eigenvalues near a target with W weighting.

    The adjoint operator is formed as ``W^{-1} L0^* W`` and eigendecomposed
    with shift-invert at ``target``.
    """
    operator, _ = _check_inputs_shape(L0, n_modes)
    adjoint_op = _build_adjoint_operator(operator, W_diag, invW_diag)

    omega = complex(target)
    eigenvalues, eigenvectors = _eigs_near_target(adjoint_op, n_modes, omega)
    eigenvalues, eigenvectors = _sort_nearest(eigenvalues, eigenvectors, omega)
    eigenvectors = _normalize_in_weighted_inner(eigenvectors, np.asarray(W_diag))

    return eigenvalues, eigenvectors
