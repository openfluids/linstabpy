"""Resolvent analysis via SVD of the weighted resolvent operator.

Computes the optimal forcing/response pairs and gains for the linearised
Navier--Stokes operator at a given frequency.  The resolvent operator is

    H = C @ inv(L0 - i*omega*I) @ B

and its singular value decomposition is obtained using SLEPc SVD
(thick-restart Lanczos bidiagonalisation) on the weighted operator

    A = W^{1/2} H W^{-1/2}

whose singular values are the resolvent gains.  The factorisation of the
shifted operator uses PETSc KSP with MUMPS (computed once, reused for
both forward and adjoint solves).

References
----------
McKeon, B. J., & Sharma, A. S. (2010). A critical-layer framework for
    turbulent pipe flow. J. Fluid Mech., 658, 336-382.
Frantz, R. A. S., Loiseau, J.-Ch., & Robinet, J.-Ch. (2023). Krylov
    methods for large-scale dynamical systems: application in fluid dynamics.
    Appl. Mech. Rev., 75(3), 030802.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from petsc4py import PETSc
from slepc4py import SLEPc

from .linsolve import make_linear_solver
from .petsc_utils import require_complex, petsc_vec, vec_to_numpy

logger = logging.getLogger(__name__)


def resolvent_analysis(
    L0: sp.spmatrix,
    omega: float,
    n_modes: int,
    W_diag: np.ndarray,
    B_mat: sp.spmatrix | None = None,
    C_mat: sp.spmatrix | None = None,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute resolvent gains, optimal forcings, and optimal responses.

    Performs a resolvent analysis of the linear operator *L0* at angular
    frequency *omega*.  The resolvent operator is

        H = FILTER( C inv(L0 - i omega I) B FILTER(v) )

    and its singular value decomposition is computed via SLEPc SVD
    (thick-restart Lanczos bidiagonalisation) on the weighted operator

        A = W^{1/2} H W^{-1/2}

    whose singular values are the resolvent gains.  A MUMPS factorisation
    of the shifted operator is computed once and reused for both the forward
    and adjoint (conjugate-transpose) solves.

    Parameters
    ----------
    L0 : sparse matrix, shape (ndof, ndof)
        LNSE operator with boundary conditions already applied.
    omega : float
        Angular frequency (rad/s) for the analysis.
    n_modes : int
        Number of leading resolvent modes to compute.
    W_diag : ndarray, shape (ndof,)
        Diagonal of the energy-norm matrix (e.g. Chu energy norm multiplied
        by quadrature weights).
    B_mat : sparse matrix, shape (ndof, ndof), optional
        Input (forcing) matrix.  Defaults to the identity.
    C_mat : sparse matrix, shape (ndof, ndof), optional
        Output (observation) matrix.  Defaults to the identity.
    filter_fn : Callable, optional
        Spatial filter applied before and after the resolvent operator.
        Typically ``mesh.filters.filter``.  Defaults to identity.
    filter_ct_fn : Callable, optional
        Adjoint-compliant spatial filter for the adjoint resolvent.
        Typically ``mesh.filters.filter_ct``.  Defaults to identity.

    Returns
    -------
    gains : ndarray, shape (n_modes,)
        Resolvent gains (singular values of *H*), sorted in descending order.
    responses : ndarray, shape (ndof, n_modes)
        Optimal response modes (columns), normalised in the *W*-inner product.
    forcings : ndarray, shape (ndof, n_modes)
        Optimal forcing modes (columns), normalised in the *W*-inner product.
    """
    require_complex()
    ndof = L0.shape[0]
    logger.info("Resolvent analysis: omega = %.6g, n_modes = %d, ndof = %d",
                omega, n_modes, ndof)

    # ------------------------------------------------------------------
    # 1. Form the shifted operator  L_sigma = L0 - i*omega*I
    # ------------------------------------------------------------------
    L_sigma = sp.csc_matrix(L0, copy=True)
    L_sigma.setdiag(L_sigma.diagonal() - 1j * omega)

    # ------------------------------------------------------------------
    # 2. MUMPS factorisation (computed once, reused for forward and adjoint)
    # ------------------------------------------------------------------
    logger.info("  MUMPS factorisation of L_sigma ...")
    lu = make_linear_solver(L_sigma)
    logger.info("  MUMPS factorisation done.")

    # ------------------------------------------------------------------
    # 3. Define forward (H) and adjoint (H^H) resolvent operators
    # ------------------------------------------------------------------
    _filt = filter_fn if filter_fn is not None else (lambda v: v)
    _filt_ct = filter_ct_fn if filter_ct_fn is not None else (lambda v: v)

    def _apply_H(v: np.ndarray) -> np.ndarray:
        """Apply resolvent: H v = FILTER(C inv(L_sigma) B FILTER(v))."""
        w = _filt(v)
        w = B_mat @ w if B_mat is not None else w
        w = lu.solve(w)
        if C_mat is not None:
            w = C_mat @ w
        w = _filt(w)
        return w

    def _apply_H_adj(v: np.ndarray) -> np.ndarray:
        """Apply adjoint resolvent: H^H v = FILTER_ct(B^H inv(L_sigma^H) C^H FILTER_ct(v))."""
        w = _filt_ct(v)
        w = C_mat.conj().T @ w if C_mat is not None else w
        w = lu.solve_transpose(w)
        if B_mat is not None:
            w = B_mat.conj().T @ w
        w = _filt_ct(w)
        return w

    # ------------------------------------------------------------------
    # 4. Build PETSc MatShell for A = W^{1/2} H W^{-1/2}
    # ------------------------------------------------------------------
    sqrt_W = np.sqrt(W_diag)
    sqrt_invW = 1.0 / sqrt_W

    class _ResolventShell:
        def mult(self, mat, x, y):
            """y = A x = W^{1/2} H W^{-1/2} x"""
            v = vec_to_numpy(x)
            v = _apply_H(sqrt_invW * v)
            y.setArray(sqrt_W * v)

        def multTranspose(self, mat, x, y):
            """y = A^T x  (plain transpose, not Hermitian transpose).

            PETSc's Python matrix type maps ``multTranspose`` to
            ``MATOP_MULT_TRANSPOSE``, and SLEPc SVD with implicit
            transpose calls this to apply A^T.  We compute A^T via
            the identity ``A^T x = conj(A^H conj(x))``.
            """
            v = vec_to_numpy(x).conj()
            v = _apply_H_adj(sqrt_W * v)
            y.setArray((sqrt_invW * v).conj())

    H_shell = PETSc.Mat().createPython((ndof, ndof), context=_ResolventShell())
    H_shell.setUp()

    # ------------------------------------------------------------------
    # 5. SLEPc SVD (thick-restart Lanczos bidiagonalisation)
    # ------------------------------------------------------------------
    ncv = min(max(4 * n_modes, 20), ndof - 1)
    logger.info("  SLEPc SVD on resolvent operator (nsv=%d, ncv=%d) ...",
                n_modes, ncv)

    svd = SLEPc.SVD().create()
    svd.setOperators(H_shell)
    svd.setType(SLEPc.SVD.Type.TRLANCZOS)
    svd.setImplicitTranspose(True)
    svd.setDimensions(nsv=n_modes, ncv=ncv)
    svd.setTolerances(tol=1e-12, max_it=1000)
    svd.solve()

    nconv = svd.getConverged()
    if nconv == 0:
        raise RuntimeError("SLEPc SVD did not converge for any singular triplet")
    if nconv < n_modes:
        logger.warning(
            "SLEPc SVD converged %d/%d singular triplets; returning partial result",
            nconv, n_modes,
        )

    n_extract = min(nconv, n_modes)
    logger.info("  SLEPc SVD done (%d converged).", nconv)

    # ------------------------------------------------------------------
    # 6. Extract gains, forcings, and responses
    #
    #    SVD: A v_i = sigma_i u_i  where A = W^{1/2} H W^{-1/2}
    #    Forcing:  f_i = W^{-1/2} v_i   (W-normalised: f^H W f = v^H v = 1)
    #    Response: r_i = W^{-1/2} u_i   (W-normalised: r^H W r = u^H u = 1)
    # ------------------------------------------------------------------
    gains = np.empty(n_extract, dtype=float)
    forcings = np.empty((ndof, n_extract), dtype=complex)
    responses = np.empty((ndof, n_extract), dtype=complex)

    u_vec = H_shell.createVecLeft()
    v_vec = H_shell.createVecRight()

    for i in range(n_extract):
        sigma = svd.getSingularTriplet(i, u_vec, v_vec)
        gains[i] = sigma
        forcings[:, i] = sqrt_invW * vec_to_numpy(v_vec)
        responses[:, i] = sqrt_invW * vec_to_numpy(u_vec)

    return gains, responses, forcings
