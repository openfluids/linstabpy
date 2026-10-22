# SPDX-License-Identifier: GPL-3.0-only
"""Resolvent analysis via PETSc/SLEPc on the weighted resolvent operator.

Computes the optimal forcing/response pairs and gains for the linearised
Navier--Stokes operator at a given frequency.  The resolvent operator is

    H = C @ inv(L0 - i*omega*I) @ B.

The implementation follows MATLAB's reference route.
It solves the eigenproblem for ``H^H W H`` in the weighted inner product.

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
import numbers
import time
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from linstabpy.linalg import shifted_operator

from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
# mpi4py must bind MPI before PETSc does: mpi4py requests MPI_THREAD_MULTIPLE,
# PETSc only MPI_THREAD_FUNNELED, and init cannot raise the level later.
# STRUMPACK/SLATE refuses to build a factor below MULTIPLE (PETSc error 57).
# Sorts above petsc4py alphabetically, so `ruff --fix` preserves the order.
# Enforced by tests/test_mpi_thread_level.py.
import mpi4py.MPI  # noqa: F401
from petsc4py import PETSc

from linstabpy.config import Config
from linstabpy.linalg.factorize import factorize
from linstabpy.resolvent.gain import compute_leading_gains
from linstabpy.resolvent.operator import ResolventOperator
from linstabpy.solvers.petsc_utils import _is_petsc_matrix, _operator_shape, require_complex

logger = logging.getLogger(__name__)


def _operator_ndof(operator: sp.spmatrix | PETSc.Mat) -> int:
    """Return the square-system size for SciPy or PETSc operators."""
    if not (_is_petsc_matrix(operator) or sp.issparse(operator)):
        raise TypeError("L0 must be a 2-D sparse matrix or PETSc Mat")
    nrows, ncols = _operator_shape(operator)
    if nrows != ncols:
        raise ValueError("L0 must be a square matrix")
    return nrows


def _validate_finite_number(name: str, value: float | complex) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        raise TypeError(f"{name} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_resolvent_inputs(
    L0: sp.spmatrix | PETSc.Mat,
    omega: float | complex,
    n_modes: int,
    W_diag: np.ndarray,
    gamma: float | complex,
) -> int:
    if isinstance(n_modes, bool) or not isinstance(n_modes, numbers.Integral):
        raise TypeError("n_modes must be an integer >= 1")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    _validate_finite_number("omega", omega)
    _validate_finite_number("gamma", gamma)
    ndof = _operator_ndof(L0)
    W = np.asarray(W_diag)
    if W.ndim != 1:
        raise ValueError("W_diag must be a 1-D vector")
    if W.size != ndof:
        raise ValueError(f"W_diag length must match L0 size {ndof}")
    return ndof


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
    residuals: bool = False,
    gamma: float = 0.0,
    return_info: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]
):
    """Compute resolvent gains, optimal modes, and optionally SVD residuals.

    Perform a resolvent analysis of the linear operator *L0* at angular
    frequency *omega*.  The resolvent operator is

        H = C FILTER( inv(L0 - i omega I) FILTER( B v ) ).

    SLEPc EPS computes its gains and modes from the weighted Hermitian
    operator.

        HtrH(v) = invW * H^H * W * H(v).
    This follows the MATLAB parity implementation directly.

    When ``residuals=False`` (the default), the returned payload is the
    historical three-tuple ``(gains, responses, forcings)``.
    When ``residuals=True``, reuse the shifted operator factorization
    to append the historical residual payload ``(rho_fwd, rho_adj)``.
    That is equivalent to the old ``resolvent_analysis_with_residuals``
    entry point.

    The residuals match ``compute_resolvent_svd_residuals`` at the same
    ``omega`` and ``gamma``, but reuse the already-factorized shifted
    operator.
    The reuse boundary is exact-frequency/discount only.
    A different ``omega`` or ``gamma`` changes the numerical values in
    ``L0 - (gamma + i*omega)*I``.
    A MUMPS factorization cannot travel across frequencies or discounts
    unless a later change introduces PETSc/MUMPS symbolic reuse and tests it.

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
    residuals : bool, optional
        If True, append SVD residual arrays ``rho_fwd`` and ``rho_adj`` to the
        returned payload.  The default False preserves the historical
        ``resolvent_analysis`` return tuple bit-for-bit.
    gamma : float, optional
        Real discount parameter in ``L0 - (gamma + i omega) I``.  The default
        ``gamma=0`` is the unshifted classical resolvent and keeps the frozen
        ``L0 - i omega I`` arithmetic.  Larger ``gamma`` values damp the
        resolvent and keep the shifted solve numerically well-posed at poles.
    return_info : bool, optional
        If True, append an info dictionary.
        It holds the discount and the exclusive stage clocks
        ``factor_s``, ``svd_s``, and ``residual_s``.

    Returns
    -------
    gains : ndarray, shape (n_modes,)
        Resolvent gains (singular values of *H*), sorted in descending order.
    responses : ndarray, shape (ndof, n_modes)
        Optimal response modes (columns), normalised in the *W*-inner product.
    forcings : ndarray, shape (ndof, n_modes)
        Optimal forcing modes (columns).
        ``solver_cfg["resolvent"]["eigen"]["forcing_norm"]`` selects the
        normalisation: *W*-inner product if ``"chu"`` (default), or the
        plain L2 norm if ``"l2"``.
    rho_fwd, rho_adj : ndarray, optional
        Returned only when ``residuals=True``.
    info : dict, optional
        Returned only when ``return_info=True``.
    """
    _validate_resolvent_inputs(L0, omega, n_modes, W_diag, gamma)
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    gains, responses, forcings, rho_fwd, rho_adj, timings = _resolvent_analysis_impl(
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
        compute_residuals=residuals,
    )
    result = (
        (gains, responses, forcings, rho_fwd, rho_adj)
        if residuals
        else (gains, responses, forcings)
    )
    if return_info:
        return (*result, {"resolvent_discount_gamma": float(gamma), **timings})
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
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, float]
]:
    """Implementation shared by the public resolvent entry points."""
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    require_complex(solver_cfg)
    ndof = _operator_ndof(L0)
    logger.info("Resolvent analysis: omega = %.6g, n_modes = %d, ndof = %d", omega, n_modes, ndof)

    # ------------------------------------------------------------------
    # 1. Form the shifted operator  L_sigma = (gamma - i*omega)*I - L0, so that
    #    H = L_sigma^-1 is the physical resolvent for q ~ exp(-i*omega*t).
    #    solver.resolvent_sign_convention = matlab_parity restores the old
    #    LinStab2D matrix L0 - (gamma + i*omega)*I (docs/resolvent_sign_convention.md).
    # ------------------------------------------------------------------
    L_sigma = shifted_operator(L0, omega, gamma=gamma, matlab_parity=cfg.resolvent.matlab_parity)

    # ------------------------------------------------------------------
    # 2. Shifted solve setup.  Direct MUMPS is the default, but the concrete
    #    KSP/PC stack comes from typed config through factorize.
    # ------------------------------------------------------------------
    logger.info("  MUMPS factorisation of L_sigma ...")
    t_factor = time.perf_counter()
    lu = factorize(L_sigma, env_prefix="LINSTABPY_RESOLVENT", solver_cfg=solver_cfg, cfg=cfg)
    factor_s = time.perf_counter() - t_factor
    logger.info("  linear solver backend: %s", lu.backend)
    logger.info("  MUMPS factorisation done.")

    # ------------------------------------------------------------------
    # 3. Define forward (H) and adjoint (H^H) resolvent operators
    # ------------------------------------------------------------------
    try:
        op = ResolventOperator(lu, B_mat, C_mat, filter_fn, filter_ct_fn)

        t_svd = time.perf_counter()
        gains, responses, forcings, W_vec, invW_vec = compute_leading_gains(
            op,
            W_diag,
            n_modes,
            ndof,
            cfg,
        )
        svd_s = time.perf_counter() - t_svd

        rho_fwd: np.ndarray | None = None
        rho_adj: np.ndarray | None = None
        residual_s = 0.0
        if compute_residuals:
            t_res = time.perf_counter()
            rho_fwd, rho_adj = _compute_resolvent_svd_residuals_from_apply(
                gains,
                forcings,
                responses,
                W_vec,
                invW_vec,
                op.apply,
                op.apply_adjoint,
                cfg.resolvent.eps.forcing_norm,
            )
            residual_s = time.perf_counter() - t_res

        timings = {"factor_s": factor_s, "svd_s": svd_s, "residual_s": residual_s}
        return gains, responses, forcings, rho_fwd, rho_adj, timings
    finally:
        lu.destroy()
        # The adjoint heal transposes L_sigma on first use, so keep it alive until here.
        if _is_petsc_matrix(L_sigma):
            L_sigma.destroy()


def _compute_resolvent_svd_residuals_from_apply(
    gains: np.ndarray,
    forcings: np.ndarray,
    responses: np.ndarray,
    W_vec: np.ndarray,
    invW_vec: np.ndarray,
    apply_H: Callable[[np.ndarray], np.ndarray],
    apply_H_adj: Callable[[np.ndarray], np.ndarray],
    forcing_norm: str = "chu",
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

    def _l2_norm(residual: np.ndarray) -> float:
        # Plain L2 norm: sqrt(r^H r). Used for both numerator and denominator
        # to maintain consistency, regardless of the forcing norm choice.
        return float(np.sqrt(max(np.real(residual.conj() @ residual), 0.0)))

    # Select the norm function based on forcing_norm
    if forcing_norm == "chu":
        norm_fn = _w_norm
        denominator_norm_fn = _w_norm
    else:  # forcing_norm == "l2"
        norm_fn = _l2_norm
        denominator_norm_fn = _l2_norm

    for i in range(n_modes):
        sigma = gains[i]
        v_i = forcings[:, i]
        u_i = responses[:, i]
        # Compute norm of v_i using the same norm as the residuals.
        # For "chu": sqrt(v^H W v); for "l2": sqrt(v^H v).
        # For properly normalised forcing modes this equals 1, but computing it
        # keeps the residuals meaningful after any filter projection or small
        # eigensolver normalization drift.
        norm_v = max(denominator_norm_fn(v_i), 1e-300)

        Hv = apply_H(v_i)
        rho_fwd[i] = norm_fn(Hv - sigma * u_i) / (sigma * norm_v)

        # The adjoint relation the singular triplet satisfies depends on which
        # operator was actually solved, and the two differ by exactly invW:
        #
        #   chu:  M = invW H^H W H,  M v = s^2 v  =>  invW H^H (W u) = s v
        #   l2:   M =      H^H W H,  M v = s^2 v  =>       H^H (W u) = s v
        #
        # Both follow from u = H v / s together with H^H W H v = s^2 v. Keeping
        # the invW in the l2 branch measures a relation the l2 triplet does not
        # satisfy -- it evaluates to s*||(invW - I) v||, large whenever W is not
        # the identity, so the adjoint gate would reject every l2 run. Checked
        # numerically on a dense 6x6 test matrix with W spanning 1e-3..1e3: the
        # correct form gives 1e-13 where the other gives 6e+04.
        HH_Wu = apply_H_adj(W_vec * u_i)
        adj_image = invW_vec * HH_Wu if forcing_norm == "chu" else HH_Wu
        rho_adj[i] = norm_fn(adj_image - sigma * v_i) / (sigma * norm_v)

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

    ``gamma`` has no default.
    This postcheck rebuilds the shifted operator
    ``L0 - (gamma + i*omega) I``.
    It MUST use the SAME ``gamma`` the analysis used.
    Otherwise the residual measures a different operator
    and a valid discounted-resolvent result fails the postcheck gate.
    Pass ``gamma=0.0`` explicitly for the standard (undiscounted) resolvent.

    The resolvent analysis solves the weighted eigenvalue problem

        invW H^H W H v_i = sigma_i^2 v_i.

    ``v_i`` uses the W-inner product (``v_i^H W v_i = 1``).
    Response is ``u_i = H v_i / sigma_i``.
    Both residuals use the W-inner-product norm
    ``||x||_W = sqrt(x^H W x)`` in numerator and denominator.

        rho_fwd[i] = ||H v_i − sigma_i u_i||_W / (sigma_i ||v_i||_W).
        rho_adj[i] = ||invW H^H (W u_i) − sigma_i v_i||_W / (sigma_i ||v_i||_W).

    *rho_fwd* is near machine precision by construction (u_i is defined as
    H v_i / sigma_i).
    *rho_adj* is the meaningful residual: it checks the adjoint half of
    the weighted eigenproblem.
    It converges to ≤ SLEPc's tolerance (1e-12) when the discrete
    operator solve is correct.
    Measuring the numerator in the W-norm (rather than the unweighted
    2-norm) keeps the measure norm-consistent.
    With a 2-norm numerator the Chu-weight dynamic range amplified
    parallel-FP jitter into a spurious ~1e-7 reading under MPI.

    Derivation: from ``HtrH v = sigma^2 v`` and ``u = H v / sigma``.

        H^H W (sigma u) = sigma^2 W v   →   invW H^H (W u) = sigma v.

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
    # resolvent_analysis(..., residuals=True).
    L_sigma = shifted_operator(L0, omega, gamma=gamma, matlab_parity=cfg.resolvent.matlab_parity)
    lu = factorize(L_sigma, env_prefix="LINSTABPY_RESOLVENT", solver_cfg=solver_cfg, cfg=cfg)
    try:
        op = ResolventOperator(lu, B_mat, C_mat, filter_fn, filter_ct_fn)

        return _compute_resolvent_svd_residuals_from_apply(
            gains,
            forcings,
            responses,
            W_vec,
            invW_vec,
            op.apply,
            op.apply_adjoint,
            cfg.resolvent.eps.forcing_norm,
        )
    finally:
        lu.destroy()
        if _is_petsc_matrix(L_sigma):
            L_sigma.destroy()
