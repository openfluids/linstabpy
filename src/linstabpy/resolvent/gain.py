# SPDX-License-Identifier: GPL-3.0-only
"""Weighted resolvent gain extraction."""

from __future__ import annotations

import logging

import numpy as np

from linstabpy.utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
# mpi4py must bind MPI before PETSc does: mpi4py requests MPI_THREAD_MULTIPLE,
# PETSc only MPI_THREAD_FUNNELED, and init cannot raise the level later.
# STRUMPACK/SLATE refuses to build a factor below MULTIPLE (PETSc error 57).
# Sorts above petsc4py alphabetically, so `ruff --fix` preserves the order.
# Enforced by tests/test_mpi_thread_level.py.
import mpi4py.MPI  # noqa: F401
from petsc4py import PETSc
from slepc4py import SLEPc

from linstabpy.config import Config
from linstabpy.resolvent.operator import ResolventOperator
from linstabpy.solvers._eps_helpers import (
    _install_eps_monitor,
    _resolve_eps_convergence_test,
)
from linstabpy.solvers.petsc_utils import vec_to_numpy

logger = logging.getLogger(__name__)


class WeightedNormalOperator:
    """PETSc MatShell context for weighted resolvent gain eigenproblem.

    Computes either invW * H^H * W * H (when forcing_norm="chu") or
    H^H * W * H (when forcing_norm="l2").
    """

    def __init__(
        self, op: ResolventOperator, W_diag: np.ndarray, ndof: int, forcing_norm: str = "chu"
    ) -> None:
        self.op = op
        W_vec = np.asarray(W_diag, dtype=float)
        if W_vec.ndim != 1 or W_vec.size != ndof:
            raise ValueError("W_diag must be a vector of length ndof")
        if np.any(W_vec == 0.0):
            raise ValueError("W_diag contains zeros; cannot build invW weighting")
        invW_vec = 1.0 / W_vec
        self.W_vec = W_vec
        self.invW_vec = invW_vec
        if forcing_norm not in ("chu", "l2"):
            raise ValueError(f"forcing_norm must be 'chu' or 'l2', got {forcing_norm!r}")
        self.forcing_norm = forcing_norm
        self.ndof = ndof

    def mult(self, mat, x, y):  # noqa: ANN001, ANN201
        op = self.op
        W_vec = self.W_vec
        invW_vec = self.invW_vec
        v = vec_to_numpy(x)
        v = op.apply(v)
        v = W_vec * v
        v = op.apply_adjoint(v)
        if self.forcing_norm == "chu":
            weighted = invW_vec * v
        else:  # forcing_norm == "l2"
            weighted = v
        if y.getComm().getSize() == 1:
            y.setArray(weighted)
        else:
            r0, r1 = y.getOwnershipRange()
            y.setArray(weighted[r0:r1])

    def create_shell(self) -> PETSc.Mat:
        """Create and set up the PETSc shell matrix for this operator."""
        htrh_shell = PETSc.Mat().createPython((self.ndof, self.ndof), context=self)
        htrh_shell.setUp()
        return htrh_shell


def _resolve_resolvent_ncv(ndof: int, n_modes: int, requested_ncv: int | None) -> int:
    """Resolve the EPS subspace size for the resolvent H*H eigenproblem."""
    default_ncv = min(max(2 * n_modes + 3, 20), ndof - 1)
    if requested_ncv is None:
        return default_ncv
    ncv = int(requested_ncv)
    if ncv <= n_modes:
        raise ValueError(f"Resolvent ncv must be > nev ({n_modes}), got {ncv}")
    return min(max(ncv, n_modes + 1), ndof - 1)


def compute_leading_gains(
    op: ResolventOperator,
    W_diag: np.ndarray,
    n_modes: int,
    ndof: int,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute leading weighted resolvent gains and modes."""
    # ------------------------------------------------------------------
    # 4. Build PETSc MatShell for the weighted normal operator.
    #    With forcing_norm="chu": HtrH(v) = invW * H^H * W * H(v)
    #    With forcing_norm="l2":  HtrH(v) = H^H * W * H(v)
    # ------------------------------------------------------------------
    forcing_norm = cfg.resolvent.eps.forcing_norm
    weighted_op = WeightedNormalOperator(op, W_diag, ndof, forcing_norm)
    W_vec = weighted_op.W_vec
    invW_vec = weighted_op.invW_vec

    htrh_shell = weighted_op.create_shell()

    # ------------------------------------------------------------------
    # 5. SLEPc EPS on the weighted normal operator.  This keeps the MATLAB-parity
    #    formulation: solve the eigenproblem, then recover response modes through H.
    # ------------------------------------------------------------------
    ncv = _resolve_resolvent_ncv(ndof, n_modes, cfg.resolvent.eps.ncv)
    eps_tol = cfg.resolvent.eps.eps_tol
    eps_max_it = cfg.resolvent.eps.eps_max_it
    conv_test_name = cfg.resolvent.eps.conv_test
    true_residual = cfg.resolvent.eps.true_residual
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
    if cfg.resolvent.eps.monitor:
        _install_eps_monitor(
            eps,
            cfg.resolvent.eps.monitor_every,
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
            if forcing_norm == "chu":
                # Normalize in W-inner product for Chu norm
                norm_sq = float(np.real(np.vdot(forcing, W_vec * forcing)))
            else:  # forcing_norm == "l2"
                # Normalize in plain L2 norm for l2
                norm_sq = float(np.real(np.vdot(forcing, forcing)))
            if norm_sq > 0.0:
                forcing = forcing / np.sqrt(norm_sq)
            forcings[:, i] = forcing

            response = op.apply(forcing)
            if sigma > 0.0:
                response = response / sigma
            responses[:, i] = response
    finally:
        vr.destroy()
        vi.destroy()
        eps.destroy()
        htrh_shell.destroy()

    order = np.argsort(gains)[::-1]
    gains = gains[order]
    forcings = forcings[:, order]
    responses = responses[:, order]

    return gains, responses, forcings, W_vec, invW_vec
