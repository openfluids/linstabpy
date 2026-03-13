"""Resolvent analysis via PETSc/SLEPc on the weighted resolvent operator.

Computes the optimal forcing/response pairs and gains for the linearised
Navier--Stokes operator at a given frequency.  The resolvent operator is

    H = C @ inv(L0 - i*omega*I) @ B

The implementation follows MATLAB's reference route by solving the eigenproblem
for ``H^H W H`` in the weighted inner product:

    HtrH(v) = invW * H^H * W * H(v)

using SLEPc EPS on a MatShell. Singular values are recovered as
``sigma = sqrt(real(lambda))`` and response modes from ``u = H(v)/sigma``.
The shifted operator factorisation uses PETSc KSP with MUMPS (computed once,
reused for both forward and adjoint solves).

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

from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
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
    solver_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute resolvent gains, optimal forcings, and optimal responses.

    Performs a resolvent analysis of the linear operator *L0* at angular
    frequency *omega*.  The resolvent operator is

        H = FILTER( C inv(L0 - i omega I) B FILTER(v) )

    and its gains/modes are computed from the weighted Hermitian operator
    ``HtrH(v) = invW * H^H * W * H(v)`` using SLEPc EPS. This follows the
    MATLAB parity implementation directly.

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
    require_complex(solver_cfg)
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
    lu = make_linear_solver(L_sigma, env_prefix="LINSTAB2D_RESOLVENT", solver_cfg=solver_cfg)
    logger.info("  linear solver backend: %s", lu.backend)
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
    # 4. Build PETSc MatShell for HtrH(v) = invW * H^H * W * H(v)
    # ------------------------------------------------------------------
    W_vec = np.asarray(W_diag, dtype=float)
    if W_vec.ndim != 1 or W_vec.size != ndof:
        raise ValueError("W_diag must be a vector of length ndof")
    if np.any(W_vec == 0.0):
        raise ValueError("W_diag contains zeros; cannot build invW weighting")
    invW_vec = 1.0 / W_vec

    class _HtrHShell:
        def mult(self, mat, x, y):
            v = vec_to_numpy(x)
            v = _apply_H(v)
            v = W_vec * v
            v = _apply_H_adj(v)
            y.setArray(invW_vec * v)

    htrh_shell = PETSc.Mat().createPython((ndof, ndof), context=_HtrHShell())
    htrh_shell.setUp()

    # ------------------------------------------------------------------
    # 5. SLEPc EPS on HtrH (MATLAB-parity route)
    # ------------------------------------------------------------------
    ncv = min(max(2 * n_modes + 3, 20), ndof - 1)
    logger.info("  SLEPc EPS on HtrH operator (nev=%d, ncv=%d) ...", n_modes, ncv)

    eps = SLEPc.EPS().create()
    eps.setOperators(htrh_shell)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setTolerances(tol=1e-12, max_it=1000)
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError("SLEPc EPS did not converge for any HtrH eigenpair")
    if nconv < n_modes:
        logger.warning(
            "SLEPc EPS converged %d/%d HtrH eigenpairs; returning partial result",
            nconv, n_modes,
        )

    n_extract = min(nconv, n_modes)
    logger.info("  SLEPc EPS done (%d converged).", nconv)

    # ------------------------------------------------------------------
    # 6. Extract gains, forcings, and responses
    # ------------------------------------------------------------------
    gains = np.empty(n_extract, dtype=float)
    forcings = np.empty((ndof, n_extract), dtype=complex)
    responses = np.empty((ndof, n_extract), dtype=complex)

    vr = htrh_shell.createVecRight()
    vi = htrh_shell.createVecRight()
    for i in range(n_extract):
        eig = eps.getEigenpair(i, vr, vi)
        sigma = float(np.sqrt(max(float(np.real(eig)), 0.0)))
        gains[i] = sigma

        forcing = vec_to_numpy(vr)
        norm_sq = float(np.real(np.vdot(forcing, W_vec * forcing)))
        if norm_sq > 0.0:
            forcing = forcing / np.sqrt(norm_sq)
        forcings[:, i] = forcing

        response = _apply_H(forcing)
        if sigma > 0.0:
            response = response / sigma
        responses[:, i] = response

    order = np.argsort(gains)[::-1]
    gains = gains[order]
    forcings = forcings[:, order]
    responses = responses[:, order]

    return gains, responses, forcings


def compute_resolvent_svd_residuals(
    L0: sp.spmatrix,
    omega: complex,
    B_mat: sp.spmatrix | None,
    C_mat: sp.spmatrix | None,
    gains: np.ndarray,
    forcings: np.ndarray,
    responses: np.ndarray,
    W_diag: np.ndarray,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    solver_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute self-residuals for resolvent SVD triplets.

    The resolvent analysis solves the weighted eigenvalue problem

        invW H^H W H v_i = sigma_i^2 v_i

    with ``v_i`` normalised in the W-inner product (``v_i^H W v_i = 1``) and
    response ``u_i = H v_i / sigma_i``.  The two residuals reported are:

        rho_fwd[i] = ||H v_i − sigma_i u_i|| / (sigma_i ||v_i||_2)
        rho_adj[i] = ||invW H^H (W u_i) − sigma_i v_i|| / (sigma_i ||v_i||_2)

    *rho_fwd* is near machine precision by construction (u_i is defined as
    H v_i / sigma_i).  *rho_adj* is the meaningful residual: it checks the
    adjoint half of the weighted eigenproblem and converges to ≤ SLEPc's
    tolerance (1e-12) when the discrete operator is solved correctly.

    Derivation: from ``HtrH v = sigma^2 v`` and ``u = H v / sigma``:

        H^H W (sigma u) = sigma^2 W v   →   invW H^H (W u) = sigma v

    so ``rho_adj`` ≈ SLEPc tol ≈ 1e-12 for converged triplets.

    Parameters
    ----------
    L0 : sparse matrix, shape (ndof, ndof)
        LNSE operator with BCs applied.
    omega : complex
        Angular frequency used in the resolvent analysis (may be complex for
        the paraboloid fallback shift).
    B_mat, C_mat : sparse matrix or None
        Input/output matrices passed to ``resolvent_analysis``.
    gains : ndarray, shape (n_modes,)
        Singular values returned by ``resolvent_analysis``.
    forcings : ndarray, shape (ndof, n_modes)
        Optimal forcing modes (right singular vectors, W-normalised).
    responses : ndarray, shape (ndof, n_modes)
        Optimal response modes (left singular vectors).
    W_diag : ndarray, shape (ndof,)
        Diagonal of the energy-norm weight matrix used in ``resolvent_analysis``.
    filter_fn, filter_ct_fn : Callable or None
        Same spatial filters passed to ``resolvent_analysis``.

    Returns
    -------
    rho_fwd : ndarray, shape (n_modes,)
        Forward residuals (near eps by construction).
    rho_adj : ndarray, shape (n_modes,)
        Adjoint residuals (primary quality signal; ≈ SLEPc tol for correct solutions).
    """
    require_complex(solver_cfg)
    n_modes = gains.shape[0]
    logger.info("Computing resolvent SVD residuals for %d modes ...", n_modes)

    W_vec = np.asarray(W_diag, dtype=float)
    invW_vec = 1.0 / W_vec

    # Rebuild shifted operator and factorization (same path as resolvent_analysis).
    L_sigma = sp.csc_matrix(L0, copy=True)
    L_sigma.setdiag(L_sigma.diagonal() - 1j * omega)
    lu = make_linear_solver(L_sigma, env_prefix="LINSTAB2D_RESOLVENT", solver_cfg=solver_cfg)

    _filt = filter_fn if filter_fn is not None else (lambda v: v)
    _filt_ct = filter_ct_fn if filter_ct_fn is not None else (lambda v: v)

    def _apply_H(v: np.ndarray) -> np.ndarray:
        w = _filt(v)
        w = B_mat @ w if B_mat is not None else w
        w = lu.solve(w)
        if C_mat is not None:
            w = C_mat @ w
        w = _filt(w)
        return w

    def _apply_H_adj(v: np.ndarray) -> np.ndarray:
        w = _filt_ct(v)
        w = C_mat.conj().T @ w if C_mat is not None else w
        w = lu.solve_transpose(w)
        if B_mat is not None:
            w = B_mat.conj().T @ w
        w = _filt_ct(w)
        return w

    rho_fwd = np.empty(n_modes, dtype=float)
    rho_adj = np.empty(n_modes, dtype=float)

    print("SVD residuals  rho_fwd = ||H v - s u||/(s||v||)  "
          "rho_adj = ||invW H^H(Wu) - s v||/(s||v||)")
    print(f"{'Mode':>4}  {'sigma':>14}  {'rho_fwd':>12}  {'rho_adj':>12}")
    print("-" * 50)
    for i in range(n_modes):
        sigma = gains[i]
        v_i = forcings[:, i]
        u_i = responses[:, i]
        norm_v = max(float(np.linalg.norm(v_i)), 1e-300)

        # Forward: ||H v_i - sigma u_i|| / (sigma ||v_i||)
        # Near eps by construction since u_i = H(v_i)/sigma in resolvent_analysis.
        Hv = _apply_H(v_i)
        rho_fwd[i] = float(np.linalg.norm(Hv - sigma * u_i)) / (sigma * norm_v)

        # Adjoint: ||invW H^H(W u_i) - sigma v_i|| / (sigma ||v_i||)
        # Tests the adjoint half of the weighted eigenproblem; converges to
        # ≤ SLEPc tolerance when the discrete operator is solved correctly.
        HH_Wu = _apply_H_adj(W_vec * u_i)
        rho_adj[i] = float(np.linalg.norm(invW_vec * HH_Wu - sigma * v_i)) / (sigma * norm_v)

        print(f"{i + 1:>4}  {sigma:>14.6g}  {rho_fwd[i]:>12.4e}  {rho_adj[i]:>12.4e}")

    print(f"{'max':>20}  {np.max(rho_fwd):>12.4e}  {np.max(rho_adj):>12.4e}")
    return rho_fwd, rho_adj
