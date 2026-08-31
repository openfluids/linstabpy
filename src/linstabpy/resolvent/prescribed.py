# SPDX-License-Identifier: GPL-3.0-only
"""Prescribed-forcing resolvent response about a steady base flow.

Solves the classical resolvent problem for a known forcing vector, instead
of the optimal-forcing SVD that resolvent_analysis computes.  Given a
discrete linear operator L0 and a right-hand side f_hat(beta, omega):

    q_hat = (L0 - (gamma + i*omega) I)^-1 (B f_hat)

with the gain

    gain = ||q_hat||_W / ||f_hat||_(chu or l2).

The pipeline reuses the classical machinery unchanged: shifted_operator,
factorize, and ResolventOperator.apply.  It never hand-rolls a parallel
solve path.

linstabpy does not derive leftover Navier-Stokes terms here. It does not
Fourier transform anything. It does not interpolate external data onto the
mesh. f_hat is the caller's right-hand side. It comes from outside
linstabpy (see docs/plan_1.1_prescribed_forcing.md).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol, TypeGuard

import numpy as np
import scipy.sparse as sp

from linstabpy.config import Config
from linstabpy.linalg import shifted_operator
from linstabpy.linalg.factorize import Factorization, factorize
from linstabpy.resolvent.operator import ResolventOperator
from linstabpy.utils.petsc_core import petsc_vec, vec_to_numpy

logger = logging.getLogger(__name__)

_VALID_FORCING_NORMS = ("chu", "l2")


class _PetscVecLike(Protocol):
    """Structural stand-in for ``petsc4py.PETSc.Vec``.

    Declared locally so this module type-checks without petsc4py
    installed. ``linstabpy.resolvent.operator`` uses the real PETSc type
    at runtime. This Protocol only describes the two calls made here.
    """

    def duplicate(self) -> _PetscVecLike: ...
    def destroy(self) -> None: ...


class _PetscMatLike(Protocol):
    """Structural stand-in for ``petsc4py.PETSc.Mat``.

    See ``_PetscVecLike`` for why this is a local Protocol rather than
    a petsc4py import. The shape accessor is PETSc's own camelCase method
    ``getSize()``.
    """

    def mult(self, x: _PetscVecLike, y: _PetscVecLike) -> None: ...
    def destroy(self) -> None: ...


def _petsc_shape(operator: _PetscMatLike) -> tuple[int, int]:
    """Return the (rows, cols) shape of a PETSc-like matrix.

    Returns
    -------
    tuple[int, int]
        The matrix shape.
    """
    nrows, ncols = operator.getSize()
    return int(nrows), int(ncols)


def _is_petsc_like(operator: object) -> TypeGuard[_PetscMatLike]:
    """Return whether ``operator`` is a PETSc matrix, by duck type.

    Returns
    -------
    bool
        True when ``operator`` looks like a ``petsc4py.PETSc.Mat``.
    """
    cls = type(operator)
    return cls.__module__.startswith("petsc4py") and cls.__name__ == "Mat"


def _operator_shape(operator: sp.spmatrix | _PetscMatLike) -> tuple[int, int]:
    """Return the (rows, cols) shape for a SciPy or PETSc operator.

    Returns
    -------
    tuple[int, int]
        The matrix shape.
    """
    if isinstance(operator, sp.spmatrix):
        return int(operator.shape[0]), int(operator.shape[1])
    return _petsc_shape(operator)


def _operator_ndof(operator: sp.spmatrix | _PetscMatLike) -> int:
    """Return the square-system size for a SciPy or PETSc operator.

    Returns
    -------
    int
        The operator size ``ndof``.

    Raises
    ------
    TypeError
        If ``operator`` is neither a SciPy sparse matrix nor a PETSc matrix.
    ValueError
        If ``operator`` is not square.
    """
    if not (_is_petsc_like(operator) or sp.issparse(operator)):
        message = "L0 must be a 2-D sparse matrix or PETSc Mat"
        raise TypeError(message)
    nrows, ncols = _operator_shape(operator)
    if nrows != ncols:
        message = "L0 must be a square matrix"
        raise ValueError(message)
    return nrows


def _apply_matrix(
    matrix: sp.spmatrix | _PetscMatLike | None,
    vector: np.ndarray,
) -> np.ndarray:
    """Apply an optional SciPy or PETSc matrix to a NumPy vector.

    Returns
    -------
    numpy.ndarray
        ``matrix @ vector``, or ``vector`` unchanged when ``matrix`` is None.
    """
    if matrix is None:
        return vector
    if isinstance(matrix, sp.spmatrix):
        return matrix @ vector
    x: _PetscVecLike = petsc_vec(vector)
    y: _PetscVecLike = x.duplicate()
    try:
        matrix.mult(x, y)
        result: np.ndarray = vec_to_numpy(y)
        return result
    finally:
        x.destroy()
        y.destroy()


def _w_norm(vector: np.ndarray, w_diag: np.ndarray) -> float:
    """Return the Chu-weighted 2-norm ``sqrt(sum(w_diag * |vector|**2))``.

    Returns
    -------
    float
        The weighted norm.
    """
    return float(np.sqrt(np.sum(w_diag * np.abs(vector) ** 2)))


def _l2_norm(vector: np.ndarray) -> float:
    """Return the plain Euclidean 2-norm of ``vector``.

    Returns
    -------
    float
        The unweighted norm.
    """
    return float(np.sqrt(np.real(np.vdot(vector, vector))))


def _validate_forcing_norm(forcing_norm: str) -> None:
    """Reject a ``forcing_norm`` outside the two supported values.

    Raises
    ------
    ValueError
        If ``forcing_norm`` is not ``"chu"`` or ``"l2"``.
    """
    if forcing_norm not in _VALID_FORCING_NORMS:
        message = f"forcing_norm must be one of {_VALID_FORCING_NORMS}, got {forcing_norm!r}"
        raise ValueError(message)


def _validate_f_hat(f_hat: np.ndarray, ndof: int) -> np.ndarray:
    """Check shape and finiteness of the prescribed forcing vector.

    Returns
    -------
    numpy.ndarray
        ``f_hat`` cast to complex.

    Raises
    ------
    ValueError
        If ``f_hat`` is not a length-``ndof`` finite 1-D vector.
    """
    f = np.asarray(f_hat)
    if f.ndim != 1:
        message = f"f_hat must be a 1-D vector, got shape {f.shape}"
        raise ValueError(message)
    if f.shape[0] != ndof:
        message = f"f_hat length {f.shape[0]} does not match operator ndof {ndof}"
        raise ValueError(message)
    f = f.astype(complex, copy=False)
    if not np.all(np.isfinite(f)):
        message = "f_hat contains NaN or Inf"
        raise ValueError(message)
    return f


def _validate_w_diag(w_diag: np.ndarray, ndof: int) -> np.ndarray:
    """Check shape and finiteness of the Chu-norm weight vector.

    Returns
    -------
    numpy.ndarray
        ``w_diag`` cast to float.

    Raises
    ------
    ValueError
        If ``w_diag`` is not a length-``ndof`` finite 1-D vector.
    """
    w = np.asarray(w_diag, dtype=float)
    if w.ndim != 1 or w.shape[0] != ndof:
        size = w.shape[0] if w.ndim == 1 else w.shape
        message = f"W_diag length {size} does not match operator ndof {ndof}"
        raise ValueError(message)
    if not np.all(np.isfinite(w)):
        message = "W_diag contains NaN or Inf"
        raise ValueError(message)
    return w


def _inner_solve_residual(
    L_sigma: sp.spmatrix | _PetscMatLike,
    lu: Factorization,
    B_mat: sp.spmatrix | _PetscMatLike | None,
    filt: Callable[[np.ndarray], np.ndarray],
    f: np.ndarray,
) -> float:
    """Return the inner-solve residual ``||L_sigma x - y|| / ||y||``.

    ``y`` is ``f`` after the input mask and filter; ``x = lu.solve(y)``.
    This is an extra solve, on top of ``ResolventOperator.apply``'s own.
    It stays separate so that method keeps its classical pipeline.
    The residual certifies only the linear-system solve: the B mask and
    filter apply exact matrix operations before and after, so the solve
    is the only inexact computational stage.

    Returns
    -------
    float
        The relative residual norm.
    """
    y = filt(_apply_matrix(B_mat, f))
    x = lu.solve(y)
    residual = _apply_matrix(L_sigma, x) - y
    y_norm = _l2_norm(y)
    if y_norm > 0.0:
        return _l2_norm(residual) / y_norm
    return _l2_norm(residual)


def _solve_pipeline(
    L0: sp.spmatrix | _PetscMatLike,
    omega: float,
    gamma: float,
    f: np.ndarray,
    B_mat: sp.spmatrix | _PetscMatLike | None,
    C_mat: sp.spmatrix | _PetscMatLike | None,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None,
    solver_cfg: dict[str, object] | None,
    cfg: Config,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Factorize, solve, and measure the inner-solve residual.

    One shifted-operator factorization backs both ``op.apply`` (the
    untouched classical pipeline) and the separate residual solve.
    The ResolventOperator pipeline must use identical instances of
    shifted_operator() and factorize() as the SVD path—any skew in
    setup or discretization between this path and resolvent_analysis
    voids the prescribed-gain <= sigma_1 inequality that both methods
    enforce.

    Returns
    -------
    tuple
        ``(q_hat, rho_fwd, timings)``, with ``timings`` holding
        ``factor_s``, ``solve_s``, and ``residual_s``.
    """
    L_sigma = shifted_operator(L0, omega, gamma=gamma)
    t0 = time.perf_counter()
    lu: Factorization = factorize(
        L_sigma, env_prefix="LINSTABPY_RESOLVENT", solver_cfg=solver_cfg, cfg=cfg
    )
    factor_s = time.perf_counter() - t0
    logger.info("  linear solver backend: %s", lu.backend)

    try:
        op = ResolventOperator(lu, B_mat, C_mat, filter_fn, filter_ct_fn)
        t0 = time.perf_counter()
        q_hat = op.apply(f)
        solve_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        filt: Callable[[np.ndarray], np.ndarray] = filter_fn or (lambda v: v)
        rho_fwd = _inner_solve_residual(L_sigma, lu, B_mat, filt, f)
        residual_s = time.perf_counter() - t0
    finally:
        if _is_petsc_like(L_sigma):
            L_sigma.destroy()
        lu.destroy()

    timings = {"factor_s": factor_s, "solve_s": solve_s, "residual_s": residual_s}
    return q_hat, rho_fwd, timings


def _gain_and_info(
    q_hat: np.ndarray,
    f: np.ndarray,
    w: np.ndarray,
    forcing_norm: str,
    gamma: float,
    rho_fwd: float,
    timings: dict[str, float],
) -> tuple[float, dict[str, float | str]]:
    """Compute the gain and assemble the diagnostics dict.

    The gain denominator branches on forcing_norm exactly like
    WeightedNormalOperator in the resolvent_analysis SVD path.
    When forcing_norm is "chu", both the response and forcing are
    normalized by the Chu-weighted norm (invW.H^H.W.H construction).
    When forcing_norm is "l2", the response stays Chu-weighted but
    the forcing uses the plain 2-norm. Mixing these norms across
    the two paths would violate the sigma_1 <= gain <= sigma_1
    bounds that both methods enforce.

    Returns
    -------
    tuple
        ``(gain, info)``.
    """
    q_w_norm = _w_norm(q_hat, w)
    f_w_norm = _w_norm(f, w)
    denom = f_w_norm if forcing_norm == "chu" else _l2_norm(f)
    gain = q_w_norm / denom if denom > 0.0 else q_w_norm
    info: dict[str, float | str] = {
        "rho_fwd": rho_fwd,
        "gain": gain,
        "forcing_norm": forcing_norm,
        "gamma": float(gamma),
        "q_hat_w_norm": q_w_norm,
        "f_hat_w_norm": f_w_norm,
        **timings,
    }
    return gain, info


def prescribed_forcing_response(
    L0: sp.spmatrix | _PetscMatLike,
    omega: float,
    W_diag: np.ndarray,
    f_hat: np.ndarray,
    *,
    gamma: float = 0.0,
    B_mat: sp.spmatrix | _PetscMatLike | None = None,
    C_mat: sp.spmatrix | _PetscMatLike | None = None,
    filter_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    forcing_norm: str = "chu",
    solver_cfg: dict[str, object] | None = None,
    cfg: Config | None = None,
) -> tuple[np.ndarray, float, dict[str, float | str]]:
    """Solve the classical resolvent problem for a prescribed forcing.

    Parameters
    ----------
    L0 : sparse matrix or PETSc.Mat, shape (ndof, ndof)
        LNSE operator, boundary conditions already applied.
    omega : float
        Frequency in ``L0 - (gamma + i*omega) I``.
    W_diag : ndarray, shape (ndof,)
        Diagonal of the Chu energy-norm matrix.
    f_hat : ndarray, shape (ndof,)
        Prescribed complex forcing vector, in the same DOF order as ``L0``.
    gamma : float, optional
        Real discount in ``L0 - (gamma + i*omega) I``. Default 0.
    B_mat, C_mat : sparse matrix or PETSc.Mat, optional
        Input/output masks, applied as in ``ResolventOperator``. Default
        identity.
    filter_fn, filter_ct_fn : Callable, optional
        Spatial filter and its adjoint, applied inside the masks. Default
        identity.
    forcing_norm : {"chu", "l2"}, optional
        Denominator norm for the gain. ``"chu"`` uses the Chu-weighted norm
        of ``f_hat``; ``"l2"`` uses its plain norm. Default ``"chu"``.
    solver_cfg, cfg : optional
        Forwarded to ``factorize``.

    Returns
    -------
    q_hat : ndarray, shape (ndof,)
        The resolvent response.
    gain : float
        ``||q_hat||_W`` over the ``forcing_norm``-selected norm of
        ``f_hat``.
    info : dict
        Holds ``rho_fwd``, ``gain``, ``forcing_norm``, and ``gamma``.
        Also holds the W-norms of ``q_hat`` and ``f_hat``, and the
        ``factor_s`` / ``solve_s`` / ``residual_s`` stage timings.
    """
    ndof = _operator_ndof(L0)
    _validate_forcing_norm(forcing_norm)
    f = _validate_f_hat(f_hat, ndof)
    w = _validate_w_diag(W_diag, ndof)
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    logger.info("Prescribed-forcing resolvent: omega=%.6g gamma=%.6g ndof=%d", omega, gamma, ndof)

    q_hat, rho_fwd, timings = _solve_pipeline(
        L0, omega, gamma, f, B_mat, C_mat, filter_fn, filter_ct_fn, solver_cfg, cfg
    )
    gain, info = _gain_and_info(q_hat, f, w, forcing_norm, gamma, rho_fwd, timings)
    return q_hat, gain, info
