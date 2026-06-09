# SPDX-License-Identifier: GPL-3.0-only
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

from linstabpy.config import Config

from ..utils._solver_cfg import _scfg, _scfg_bool, _scfg_float, _scfg_int
from ._eps_helpers import (
    _install_eps_monitor,
    _resolve_eps_convergence_test,
    _resolve_solver_cfg_subsection,
)
from .linsolve import make_linear_solver
from .petsc_utils import _is_petsc_matrix, _operator_shape, petsc_vec, require_complex, vec_to_numpy

logger = logging.getLogger(__name__)


def _operator_ndof(operator: sp.spmatrix | PETSc.Mat) -> int:
    """Return the square-system size for SciPy or PETSc operators."""
    nrows, ncols = _operator_shape(operator)
    if nrows != ncols:
        raise ValueError("L0 must be a square matrix")
    return nrows


def _shifted_operator(
    operator: sp.spmatrix | PETSc.Mat,
    omega: complex,
    gamma: float = 0.0,
) -> sp.csc_matrix | PETSc.Mat:
    """Return ``operator - (gamma + 1j*omega)*I`` preserving the backend."""
    if _is_petsc_matrix(operator):
        shifted = operator.copy()
        if gamma == 0.0:
            shifted.shift(-1j * omega)
        else:
            shifted.shift(-(float(gamma) + 1j * omega))
        shifted.assemble()
        return shifted
    shifted = sp.csc_matrix(operator, copy=True)
    if gamma == 0.0:
        shifted.setdiag(shifted.diagonal() - 1j * omega)
    else:
        shifted.setdiag(shifted.diagonal() - (float(gamma) + 1j * omega))
    return shifted


def _apply_optional_matrix(
    matrix: sp.spmatrix | PETSc.Mat | None,
    vector: np.ndarray,
) -> np.ndarray:
    """Apply an optional SciPy/PETSc matrix to a NumPy vector."""
    if matrix is None:
        return vector
    if isinstance(matrix, PETSc.Mat):
        x = petsc_vec(vector)
        y = x.duplicate()
        try:
            matrix.mult(x, y)
            return vec_to_numpy(y)
        finally:
            x.destroy()
            y.destroy()
    return matrix @ vector


def _apply_optional_adjoint_matrix(
    matrix: sp.spmatrix | PETSc.Mat | None,
    vector: np.ndarray,
) -> np.ndarray:
    """Apply an optional conjugate-transpose matrix to a NumPy vector."""
    if matrix is None:
        return vector
    if isinstance(matrix, PETSc.Mat):
        x = petsc_vec(vector)
        y = x.duplicate()
        try:
            matrix.multHermitian(x, y)
            return vec_to_numpy(y)
        finally:
            x.destroy()
            y.destroy()
    return matrix.conj().T @ vector


def _resolve_resolvent_ncv(ndof: int, n_modes: int, requested_ncv: int | None) -> int:
    """Resolve the EPS subspace size for the resolvent H*H eigenproblem."""
    default_ncv = min(max(2 * n_modes + 3, 20), ndof - 1)
    if requested_ncv is None:
        return default_ncv
    ncv = int(requested_ncv)
    if ncv <= n_modes:
        raise ValueError(f"Resolvent ncv must be > nev ({n_modes}), got {ncv}")
    return min(max(ncv, n_modes + 1), ndof - 1)


def resolvent_analysis(
    L0: sp.spmatrix | PETSc.Mat,
    omega: float,
    n_modes: int,
    W_diag: np.ndarray,
    B_mat: sp.spmatrix | PETSc.Mat | None = None,
    C_mat: sp.spmatrix | PETSc.Mat | None = None,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
    *,
    gamma: float = 0.0,
    return_info: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]
):
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
    gamma : float, optional
        Real discount parameter in ``L0 - (gamma + i omega) I``.  The default
        ``gamma=0`` is the unshifted classical resolvent and keeps the frozen
        ``L0 - i omega I`` arithmetic.  Larger ``gamma`` values damp the
        resolvent and keep the shifted solve numerically well-posed at poles.
    return_info : bool, optional
        If True, append an info dictionary containing
        ``{"resolvent_discount_gamma": float(gamma)}``.

    Returns
    -------
    gains : ndarray, shape (n_modes,)
        Resolvent gains (singular values of *H*), sorted in descending order.
    responses : ndarray, shape (ndof, n_modes)
        Optimal response modes (columns), normalised in the *W*-inner product.
    forcings : ndarray, shape (ndof, n_modes)
        Optimal forcing modes (columns), normalised in the *W*-inner product.
    info : dict, optional
        Returned only when ``return_info=True``.
    """
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    gains, responses, forcings, _, _ = _resolvent_analysis_impl(
        L0,
        omega,
        n_modes,
        W_diag,
        B_mat=B_mat,
        C_mat=C_mat,
        filter_fn=filter_fn,
        filter_ct_fn=filter_ct_fn,
        solver_cfg=solver_cfg,
        cfg=cfg,
        gamma=gamma,
        compute_residuals=False,
    )
    if return_info:
        return gains, responses, forcings, {"resolvent_discount_gamma": float(gamma)}
    return gains, responses, forcings


def resolvent_analysis_with_residuals(
    L0: sp.spmatrix | PETSc.Mat,
    omega: float,
    n_modes: int,
    W_diag: np.ndarray,
    B_mat: sp.spmatrix | PETSc.Mat | None = None,
    C_mat: sp.spmatrix | PETSc.Mat | None = None,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
    *,
    gamma: float = 0.0,
    return_info: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]
):
    """Compute resolvent modes and SVD residuals with one shifted solve setup.

    This is equivalent to calling ``resolvent_analysis`` followed by
    ``compute_resolvent_svd_residuals`` at the same ``omega``, but it reuses
    the already-factorized shifted operator.  The reuse boundary is deliberately
    exact-frequency/discount only: a different ``omega`` or ``gamma`` changes
    the numerical values in ``L0 - (gamma + i*omega)*I``, so a MUMPS
    factorization cannot be carried across frequencies or discounts unless
    PETSc/MUMPS symbolic reuse is introduced and tested as a separate policy.

    ``gamma=0`` is the unshifted classical resolvent and keeps the frozen
    ``L0 - i*omega*I`` arithmetic.  Larger ``gamma`` values damp the resolvent
    and keep the shifted solve numerically well-posed at poles.  If
    ``return_info=True``, an info dictionary containing
    ``{"resolvent_discount_gamma": float(gamma)}`` is appended to the existing
    return tuple.
    """
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    result = _resolvent_analysis_impl(
        L0,
        omega,
        n_modes,
        W_diag,
        B_mat=B_mat,
        C_mat=C_mat,
        filter_fn=filter_fn,
        filter_ct_fn=filter_ct_fn,
        solver_cfg=solver_cfg,
        cfg=cfg,
        gamma=gamma,
        compute_residuals=True,
    )
    if return_info:
        return (*result, {"resolvent_discount_gamma": float(gamma)})
    return result


def _resolvent_analysis_impl(
    L0: sp.spmatrix | PETSc.Mat,
    omega: float,
    n_modes: int,
    W_diag: np.ndarray,
    B_mat: sp.spmatrix | PETSc.Mat | None = None,
    C_mat: sp.spmatrix | PETSc.Mat | None = None,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    solver_cfg: dict | None = None,
    *,
    gamma: float = 0.0,
    compute_residuals: bool,
    cfg: Config | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Implementation shared by the public resolvent entry points."""
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    require_complex(solver_cfg)
    ndof = _operator_ndof(L0)
    logger.info("Resolvent analysis: omega = %.6g, n_modes = %d, ndof = %d", omega, n_modes, ndof)

    # ------------------------------------------------------------------
    # 1. Form the shifted operator  L_sigma = L0 - (gamma + i*omega)*I
    # ------------------------------------------------------------------
    L_sigma = _shifted_operator(L0, omega, gamma=gamma)

    # ------------------------------------------------------------------
    # 2. Shifted solve setup.  Direct MUMPS is the default, but the concrete
    #    KSP/PC stack is selected by SolverConfig through make_linear_solver.
    # ------------------------------------------------------------------
    logger.info("  MUMPS factorisation of L_sigma ...")
    lu = make_linear_solver(L_sigma, env_prefix="LINSTABPY_RESOLVENT", solver_cfg=solver_cfg, cfg=cfg)
    # make_linear_solver copies the operator internally; free the shifted copy.
    if _is_petsc_matrix(L_sigma):
        L_sigma.destroy()
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
        w = _apply_optional_matrix(B_mat, w)
        w = lu.solve(w)
        w = _apply_optional_matrix(C_mat, w)
        w = _filt(w)
        return w

    def _apply_H_adj(v: np.ndarray) -> np.ndarray:
        """Apply adjoint resolvent: H^H v = FILTER_ct(B^H inv(L_sigma^H) C^H FILTER_ct(v))."""
        w = _filt_ct(v)
        w = _apply_optional_adjoint_matrix(C_mat, w)
        w = lu.solve_transpose(w)
        w = _apply_optional_adjoint_matrix(B_mat, w)
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
            weighted = invW_vec * v
            if y.getComm().getSize() == 1:
                y.setArray(weighted)
            else:
                r0, r1 = y.getOwnershipRange()
                y.setArray(weighted[r0:r1])

    htrh_shell = PETSc.Mat().createPython((ndof, ndof), context=_HtrHShell())
    htrh_shell.setUp()

    # ------------------------------------------------------------------
    # 5. SLEPc EPS on HtrH.  This keeps the MATLAB-parity formulation: solve
    #    the weighted normal operator, then recover response modes through H.
    # ------------------------------------------------------------------
    _res_cfg = _resolve_solver_cfg_subsection(solver_cfg, "LINSTABPY_RESOLVENT")
    _res_eps_cfg = _res_cfg.get("eigen", _res_cfg) if isinstance(_res_cfg, dict) else {}
    ncv = _resolve_resolvent_ncv(
        ndof,
        n_modes,
        int(_res_eps_cfg["ncv"])
        if isinstance(_res_eps_cfg, dict) and "ncv" in _res_eps_cfg
        else None,
    )
    eps_tol = _scfg_float(_res_eps_cfg, "eps_tol", "LINSTABPY_RESOLVENT_EPS_TOL", 1e-8)
    eps_max_it = _scfg_int(_res_eps_cfg, "eps_max_it", "LINSTABPY_RESOLVENT_EPS_MAX_IT", 1000)
    conv_test_name = _scfg(_res_eps_cfg, "conv_test", "LINSTABPY_RESOLVENT_EPS_CONV_TEST", "")
    true_residual = _scfg_bool(
        _res_eps_cfg,
        "true_residual",
        "LINSTABPY_RESOLVENT_EPS_TRUE_RESIDUAL",
        False,
    )
    logger.info(
        "  SLEPc EPS on HtrH operator (nev=%d, ncv=%d, tol=%g, conv_test=%s, true_residual=%s) ...",
        n_modes,
        ncv,
        eps_tol,
        conv_test_name or "default",
        true_residual,
    )

    eps = SLEPc.EPS().create()
    eps.setOperators(htrh_shell)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    if conv_test_name:
        eps.setConvergenceTest(_resolve_eps_convergence_test(conv_test_name))
    eps.setTolerances(tol=eps_tol, max_it=eps_max_it)
    if hasattr(eps, "setTrueResidual") and true_residual:
        eps.setTrueResidual(True)
    if _scfg_bool(_res_eps_cfg, "monitor", "LINSTABPY_RESOLVENT_EPS_MONITOR", False):
        _install_eps_monitor(
            eps,
            _scfg_int(_res_eps_cfg, "monitor_every", "LINSTABPY_RESOLVENT_EPS_MONITOR_EVERY", 10),
            label="resolvent",
        )
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError("SLEPc EPS did not converge for any HtrH eigenpair")
    if nconv < n_modes:
        logger.warning(
            "SLEPc EPS converged %d/%d HtrH eigenpairs; returning partial result",
            nconv,
            n_modes,
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
    try:
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
    finally:
        vr.destroy()
        vi.destroy()
        eps.destroy()
        htrh_shell.destroy()

    try:
        order = np.argsort(gains)[::-1]
        gains = gains[order]
        forcings = forcings[:, order]
        responses = responses[:, order]

        rho_fwd: np.ndarray | None = None
        rho_adj: np.ndarray | None = None
        if compute_residuals:
            rho_fwd, rho_adj = _compute_resolvent_svd_residuals_from_apply(
                gains,
                forcings,
                responses,
                W_vec,
                invW_vec,
                _apply_H,
                _apply_H_adj,
            )

        return gains, responses, forcings, rho_fwd, rho_adj
    finally:
        lu.destroy()


def _compute_resolvent_svd_residuals_from_apply(
    gains: np.ndarray,
    forcings: np.ndarray,
    responses: np.ndarray,
    W_vec: np.ndarray,
    invW_vec: np.ndarray,
    apply_H: Callable[[np.ndarray], np.ndarray],
    apply_H_adj: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SVD residuals from already-constructed resolvent actions."""
    n_modes = gains.shape[0]
    rho_fwd = np.empty(n_modes, dtype=float)
    rho_adj = np.empty(n_modes, dtype=float)

    print(
        "SVD residuals  rho_fwd = ||H v - s u||_W/(s||v||_W)  "
        "rho_adj = ||invW H^H(Wu) - s v||_W/(s||v||_W)"
    )
    print(f"{'Mode':>4}  {'sigma':>14}  {'rho_fwd':>12}  {'rho_adj':>12}")
    print("-" * 50)

    def _w_norm(residual: np.ndarray) -> float:
        # W-inner-product norm: sqrt(r^H W r). Used for BOTH numerator and
        # denominator so the residual measure is norm-consistent. An earlier
        # version measured the numerator in the unweighted 2-norm while the
        # denominator was the W-norm; because the Chu weight spans orders of
        # magnitude that mismatch amplified a tiny (~1e-12) parallel-FP jitter
        # into a ~1e-7 reading under MPI. See
        # resolvent/docs/mpi_resolvent_residual_investigation.md.
        return float(np.sqrt(max(np.real(residual.conj() @ (W_vec * residual)), 0.0)))

    for i in range(n_modes):
        sigma = gains[i]
        v_i = forcings[:, i]
        u_i = responses[:, i]
        # W-norm: sqrt(v^H W v). For properly W-normalised forcing modes this
        # equals 1, but computing it keeps the residuals meaningful after any
        # filter projection or small eigensolver normalization drift.
        w_norm_v = max(_w_norm(v_i), 1e-300)

        Hv = apply_H(v_i)
        rho_fwd[i] = _w_norm(Hv - sigma * u_i) / (sigma * w_norm_v)

        HH_Wu = apply_H_adj(W_vec * u_i)
        rho_adj[i] = _w_norm(invW_vec * HH_Wu - sigma * v_i) / (sigma * w_norm_v)

        print(f"{i + 1:>4}  {sigma:>14.6g}  {rho_fwd[i]:>12.4e}  {rho_adj[i]:>12.4e}")

    print(f"{'max':>20}  {np.max(rho_fwd):>12.4e}  {np.max(rho_adj):>12.4e}")
    return rho_fwd, rho_adj


def compute_resolvent_svd_residuals(
    L0: sp.spmatrix | PETSc.Mat,
    omega: complex,
    B_mat: sp.spmatrix | PETSc.Mat | None,
    C_mat: sp.spmatrix | PETSc.Mat | None,
    gains: np.ndarray,
    forcings: np.ndarray,
    responses: np.ndarray,
    W_diag: np.ndarray,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
    *,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute self-residuals for resolvent SVD triplets.

    ``gamma`` is REQUIRED (no default): this postcheck rebuilds the shifted
    operator ``L0 - (gamma + i*omega) I`` and MUST use the SAME ``gamma`` the
    analysis used, or the residual is measured against a different operator and a
    valid discounted-resolvent result fails the postcheck gate. Pass ``gamma=0.0``
    explicitly for the standard (undiscounted) resolvent.

    The resolvent analysis solves the weighted eigenvalue problem

        invW H^H W H v_i = sigma_i^2 v_i

    with ``v_i`` normalised in the W-inner product (``v_i^H W v_i = 1``) and
    response ``u_i = H v_i / sigma_i``.  Both residuals use the W-inner-product
    norm ``||x||_W = sqrt(x^H W x)`` consistently in numerator and denominator:

        rho_fwd[i] = ||H v_i − sigma_i u_i||_W / (sigma_i ||v_i||_W)
        rho_adj[i] = ||invW H^H (W u_i) − sigma_i v_i||_W / (sigma_i ||v_i||_W)

    *rho_fwd* is near machine precision by construction (u_i is defined as
    H v_i / sigma_i).  *rho_adj* is the meaningful residual: it checks the
    adjoint half of the weighted eigenproblem and converges to ≤ SLEPc's
    tolerance (1e-12) when the discrete operator is solved correctly.  Measuring
    the numerator in the W-norm (rather than the unweighted 2-norm) keeps the
    measure norm-consistent; with a 2-norm numerator the Chu-weight dynamic range
    amplified parallel-FP jitter into a spurious ~1e-7 reading under MPI.

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
    gamma : float, optional
        Real discount parameter in ``L0 - (gamma + i omega) I``.  The default
        ``gamma=0`` is the unshifted classical resolvent and keeps the frozen
        ``L0 - i omega I`` arithmetic.  Larger ``gamma`` values damp the
        resolvent and keep the shifted solve numerically well-posed at poles.

    Returns
    -------
    rho_fwd : ndarray, shape (n_modes,)
        Forward residuals (near eps by construction).
    rho_adj : ndarray, shape (n_modes,)
        Adjoint residuals (primary quality signal; ≈ SLEPc tol for correct solutions).
    """
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    require_complex(solver_cfg)
    n_modes = gains.shape[0]
    logger.info("Computing resolvent SVD residuals for %d modes ...", n_modes)

    W_vec = np.asarray(W_diag, dtype=float)
    invW_vec = 1.0 / W_vec

    # Legacy postcheck path: rebuild the same shifted solve so callers that need
    # a separate "analysis" and "postcheck" stage keep their timing/provenance
    # boundaries. New code that can combine both stages should call
    # resolvent_analysis_with_residuals().
    L_sigma = _shifted_operator(L0, omega, gamma=gamma)
    lu = make_linear_solver(L_sigma, env_prefix="LINSTABPY_RESOLVENT", solver_cfg=solver_cfg, cfg=cfg)
    try:
        if _is_petsc_matrix(L_sigma):
            L_sigma.destroy()

        _filt = filter_fn if filter_fn is not None else (lambda v: v)
        _filt_ct = filter_ct_fn if filter_ct_fn is not None else (lambda v: v)

        def _apply_H(v: np.ndarray) -> np.ndarray:
            w = _filt(v)
            w = _apply_optional_matrix(B_mat, w)
            w = lu.solve(w)
            w = _apply_optional_matrix(C_mat, w)
            w = _filt(w)
            return w

        def _apply_H_adj(v: np.ndarray) -> np.ndarray:
            w = _filt_ct(v)
            w = _apply_optional_adjoint_matrix(C_mat, w)
            w = lu.solve_transpose(w)
            w = _apply_optional_adjoint_matrix(B_mat, w)
            w = _filt_ct(w)
            return w

        return _compute_resolvent_svd_residuals_from_apply(
            gains,
            forcings,
            responses,
            W_vec,
            invW_vec,
            _apply_H,
            _apply_H_adj,
        )
    finally:
        lu.destroy()
