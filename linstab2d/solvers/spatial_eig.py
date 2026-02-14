"""Generalized eigensolvers for spatial stability formulations."""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


def _eigs_generalized(
    Lc: sp.spmatrix,
    R: sp.spmatrix,
    n_modes: int,
    omega: float | complex,
    target: float | complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``(Lc + omega*Lw) x = lambda R x`` near ``target``."""
    if n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")

    Lc = sp.csc_matrix(Lc, dtype=complex)
    R = sp.csc_matrix(R, dtype=complex)
    if Lc.shape[0] != Lc.shape[1] or R.shape != Lc.shape:
        raise ValueError("Lc and R must have identical square shapes")
    if n_modes > Lc.shape[0]:
        raise ValueError("n_modes cannot exceed matrix size")

    A = Lc + omega * R
    if A.shape[0] <= 2 or n_modes >= A.shape[0] - 1:
        dense = np.asarray(A.toarray())
        dense_R = np.asarray(R.toarray())
        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(dense_R, dense))
        order = np.argsort(np.abs(eigvals - target))
        sel = order[:n_modes]
        return eigvals[sel], eigvecs[:, sel]

    B = R
    shifted = A - target * B
    lu = spla.splu(shifted)
    op_inv = spla.LinearOperator(
        shape=A.shape,
        matvec=lu.solve,
        dtype=complex,
    )
    logger.info("  Generalized eigs with shift sigma = %.6g%+.6gj", np.real(target), np.imag(target))
    eigvals, eigvecs = spla.eigs(
        A,
        k=n_modes,
        M=B,
        sigma=target,
        OPinv=op_inv,
        which="LM",
        tol=np.finfo(float).eps,
    )
    return eigvals, eigvecs


def spatial_eigenvalues(
    Lc: sp.spmatrix,
    Lw: sp.spmatrix,
    R: sp.spmatrix,
    omega: float | complex,
    n_modes: int,
    target: float | complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the spatial EVP `(Lc + omega Lw) @ v = alpha R @ v`.

    Returns:
        alpha : ndarray, shape (n_modes,)
            Spatial eigenvalues.
        modes : ndarray, shape (N, n_modes)
            Right eigenvectors.
    """
    eigenvalues, eigenvectors = _eigs_generalized(Lc + omega * Lw, R, n_modes, omega, target)
    return eigenvalues, eigenvectors
