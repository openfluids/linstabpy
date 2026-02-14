"""Resolvent analysis via eigendecomposition of the weighted resolvent operator.

Computes the optimal forcing/response pairs and gains for the linearised
Navier--Stokes operator at a given frequency.  The resolvent operator is

    H = C @ inv(L0 - i*omega*I) @ B

and its singular value decomposition is obtained through the eigendecomposition
of the Hermitian positive-semi-definite operator

    inv(W) @ H^H @ W @ H

whose eigenvalues are the squared resolvent gains.

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

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


def _log_solver_backend() -> None:
    """Log solver-backend intent and availability.

    Default behavior is to use SciPy's built-in SuperLU path (via ``spla.splu``),
    with UMFPACK retained as an optional drop-in path when ``scikit-umfpack`` is
    available. Keep this as a runtime contract: robust default first, optional
    performance path available for future benchmarking.
    """

    try:
        import sksparse  # type: ignore
    except Exception:
        logger.info("scikit-umfpack not detected; SciPy LU backend will be used")
    else:
        logger.info(
            "scikit-umfpack detected (%s); SciPy may route LU to UMFPACK; "
            "SuperLU remains default unless explicitly toggled by SciPy runtime path.",
            getattr(sksparse, "__version__", "installed"),
        )


def resolvent_analysis(
    L0: sp.spmatrix,
    omega: float,
    n_modes: int,
    W_diag: np.ndarray,
    B_mat: sp.spmatrix | None = None,
    C_mat: sp.spmatrix | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute resolvent gains, optimal forcings, and optimal responses.

    Performs a resolvent analysis of the linear operator *L0* at angular
    frequency *omega*.  The resolvent operator ``H = C inv(L0 - i omega I) B``
    is characterised through the eigendecomposition of

        inv(W) H^H W H

    using :func:`scipy.sparse.linalg.eigs`.  An LU factorisation of the
    shifted operator is computed once and reused for both the forward and
    adjoint (conjugate-transpose) solves.

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

    Returns
    -------
    gains : ndarray, shape (n_modes,)
        Resolvent gains (singular values of *H*), sorted in descending order.
    responses : ndarray, shape (ndof, n_modes)
        Optimal response modes (columns), normalised in the *W*-inner product.
    forcings : ndarray, shape (ndof, n_modes)
        Optimal forcing modes (columns), normalised in the *W*-inner product.
    """
    ndof = L0.shape[0]
    logger.info("Resolvent analysis: omega = %.6g, n_modes = %d, ndof = %d",
                omega, n_modes, ndof)
    _log_solver_backend()

    # ------------------------------------------------------------------
    # 1. Form the shifted operator  L_sigma = L0 - i*omega*I
    # ------------------------------------------------------------------
    L_sigma = sp.csc_matrix(L0 - 1j * omega * sp.eye(ndof, format="csc"))

    # ------------------------------------------------------------------
    # 2. LU factorisation (computed once, reused for forward and adjoint)
    # ------------------------------------------------------------------
    logger.info("  LU factorisation of L_sigma ...")
    lu = spla.splu(L_sigma)
    logger.info("  LU factorisation done.")

    # ------------------------------------------------------------------
    # 3. Define forward (H) and adjoint (H^H) resolvent operators
    # ------------------------------------------------------------------
    def _apply_H(v: np.ndarray) -> np.ndarray:
        """Apply resolvent: H v = C inv(L_sigma) B v."""
        w = B_mat @ v if B_mat is not None else v.copy()
        w = lu.solve(w)
        if C_mat is not None:
            w = C_mat @ w
        return w

    def _apply_H_adj(v: np.ndarray) -> np.ndarray:
        """Apply adjoint resolvent: H^H v = B^H inv(L_sigma^H) C^H v."""
        w = C_mat.conj().T @ v if C_mat is not None else v.copy()
        w = lu.solve(w, trans="H")
        if B_mat is not None:
            w = B_mat.conj().T @ w
        return w

    # ------------------------------------------------------------------
    # 4. Build the matvec for  inv(W) H^H W H
    # ------------------------------------------------------------------
    inv_W_diag = 1.0 / W_diag

    def _matvec_HtrWH(v: np.ndarray) -> np.ndarray:
        """Apply inv(W) @ H^H @ W @ H to vector *v*."""
        u = _apply_H(v)
        u = W_diag * u             # W @ H v  (diagonal multiply)
        u = _apply_H_adj(u)
        u = inv_W_diag * u         # inv(W) @ (...)
        return u

    A_op = spla.LinearOperator(
        shape=(ndof, ndof),
        matvec=_matvec_HtrWH,
        dtype=complex,
    )

    # ------------------------------------------------------------------
    # 5. Eigendecomposition (largest-magnitude eigenvalues)
    # ------------------------------------------------------------------
    logger.info("  Eigendecomposition via eigs (k=%d) ...", n_modes)
    eigenvalues, eigenvectors = spla.eigs(
        A_op,
        k=n_modes,
        which="LM",
        tol=np.finfo(float).eps,
    )
    logger.info("  Eigendecomposition done.")

    # Eigenvalues of H^H W H should be real and positive; small imaginary
    # parts are numerical artefacts.
    sigma_sq = np.real(eigenvalues)

    # Sort in descending order of gain
    order = np.argsort(-sigma_sq)
    sigma_sq = sigma_sq[order]
    V = eigenvectors[:, order]

    gains = np.sqrt(np.maximum(sigma_sq, 0.0))

    # ------------------------------------------------------------------
    # 6. Normalise forcings and recover responses
    # ------------------------------------------------------------------
    forcings = np.empty((ndof, n_modes), dtype=complex)
    responses = np.empty((ndof, n_modes), dtype=complex)

    for i in range(n_modes):
        vi = V[:, i]
        # Normalise in the W-inner product:  v^H W v = 1
        norm_W = np.sqrt(np.maximum(np.real(vi.conj() @ (W_diag * vi)), 0.0))
        forcings[:, i] = vi / norm_W
        # Response mode: u = H v / sigma
        if gains[i] > 0.0:
            responses[:, i] = _apply_H(forcings[:, i]) / gains[i]
        else:
            responses[:, i] = 0.0

    return gains, responses, forcings
