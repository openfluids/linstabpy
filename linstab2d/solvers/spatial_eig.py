"""Generalized eigensolvers for spatial stability formulations."""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .linsolve import make_linear_solver

logger = logging.getLogger(__name__)


def _eigs_generalized(
    Lc: sp.spmatrix,
    R: sp.spmatrix,
    n_modes: int,
    target: float | complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``A x = lambda R x`` near ``target`` via shift-invert.

    Converts the generalized EVP to standard form to avoid scipy's
    unreliable eigenvector recovery in generalized mode.  The shift-invert
    operator is ``inv(A - sigma R) @ R``, whose eigenvalues are
    ``1 / (lambda - sigma)``.
    """
    if n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")

    A = sp.csc_matrix(Lc, dtype=complex)
    B = sp.csc_matrix(R, dtype=complex)
    if A.shape[0] != A.shape[1] or B.shape != A.shape:
        raise ValueError("Lc and R must have identical square shapes")
    if n_modes > A.shape[0]:
        raise ValueError("n_modes cannot exceed matrix size")

    if A.shape[0] <= 2 or n_modes >= A.shape[0] - 1:
        dense = np.asarray(A.toarray())
        dense_R = np.asarray(B.toarray())
        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(dense_R, dense))
        order = np.argsort(np.abs(eigvals - target))
        sel = order[:n_modes]
        return eigvals[sel], eigvecs[:, sel]

    # Factor shifted matrix: S = A - sigma * R
    shifted = A - target * B
    solver = make_linear_solver(shifted)

    # Standard-form shift-invert operator: inv(S) @ R
    def _op(x):
        return solver.solve(B @ x)

    op_inv = spla.LinearOperator(
        shape=A.shape,
        matvec=_op,
        dtype=np.complex128,
    )

    ncv = min(max(2 * n_modes + 1, 20), A.shape[0])
    logger.info(
        "  Spatial eigs: sigma = %.6g%+.6gj, k=%d, ncv=%d",
        np.real(target), np.imag(target), n_modes, ncv,
    )

    # Solve standard EVP for mu = 1/(lambda - sigma)
    mu, vecs = spla.eigs(
        op_inv,
        k=n_modes,
        which="LM",
        ncv=ncv,
        tol=1e-10,
    )

    eigvals = target + 1.0 / mu
    return eigvals, vecs


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
    eigenvalues, eigenvectors = _eigs_generalized(Lc + omega * Lw, R, n_modes, target)
    return eigenvalues, eigenvectors
