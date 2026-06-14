# SPDX-License-Identifier: GPL-3.0-only
"""Stability eigenvalue entry points."""

from __future__ import annotations

import dataclasses
import logging
import numbers
import sys
import time

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
from petsc4py import PETSc
from slepc4py import SLEPc

from linstabpy.config import Config
from linstabpy.linalg.backends import (
    PRECONDITIONER_PROBE_RESIDUAL_REL_MAX,
    backend_probe_is_preconditioner_role,
    factor_backend_available,
    resolve_auto_factor_backend,
)
from linstabpy.solvers import eigenvalue as _temporal_core
from linstabpy.solvers._eps_helpers import (
    _install_eps_monitor,
    _monitor_state_details,
    _resolve_eps_convergence_test,
    _resolve_eps_error_type,
    _resolve_solver_cfg_subsection,
)
from linstabpy.solvers.eigenvalue import (
    _check_inputs_shape,
    _configure_st_for_eigenvalue,
    _nearest_order,
    _normalize_in_weighted_inner,
    _raise_backend_probe_failure,
    _raise_eps_setup_failure,
    _raise_eps_solve_failure,
    _reorder_info_residuals,
    _resolve_ncv,
    _safe_petsc_info,
    _scaled_operator,
    _sort_nearest,
    _warn_if_shifted_operator_is_ill_conditioned,
)
from linstabpy.solvers.petsc_utils import (
    _is_petsc_matrix,
    _operator_shape,
    numpy_columns_to_petsc_vecs,
    petsc_to_scipy,
    require_complex,
    scipy_to_petsc,
    scipy_to_petsc_distributed_from_root,
    vec_to_numpy,
)
from linstabpy.solvers.spatial_eig import _eigs_generalized
from linstabpy.utils._solver_cfg import ITERATIVE_EIGEN_SOLVER_TYPES

logger = logging.getLogger(__name__)


def _validate_finite_number(name: str, value: float | complex) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        raise TypeError(f"{name} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_sparse_square_operator(
    name: str,
    operator: sp.spmatrix | PETSc.Mat,
) -> int:
    if not (_is_petsc_matrix(operator) or sp.issparse(operator) or isinstance(operator, np.ndarray)):
        raise TypeError(f"{name} must be a 2-D array, sparse matrix, or PETSc Mat")
    shape = _operator_shape(operator) if not isinstance(operator, np.ndarray) else operator.shape
    if len(shape) != 2:
        raise ValueError(f"{name} must be a 2-D operator")
    nrows, ncols = shape
    if nrows != ncols:
        raise ValueError(f"{name} must be a square matrix")
    return nrows


def _validate_n_modes(n_modes: int, ndof: int) -> None:
    if isinstance(n_modes, bool) or not isinstance(n_modes, numbers.Integral):
        raise TypeError("n_modes must be an integer >= 1")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if n_modes > ndof:
        raise ValueError("n_modes cannot exceed matrix size")


def _validate_weight_vector(name: str, value: np.ndarray, ndof: int) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if vector.size != ndof:
        raise ValueError(f"{name} length must match operator size {ndof}")
    return vector


def _solve_near_target(
    operator: sp.csc_matrix | PETSc.Mat,
    n_modes: int,
    target: complex,
    *,
    ncv: int | None = None,
    initial_space: np.ndarray | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
    return_info: bool = False,
    mass_operator: sp.spmatrix | PETSc.Mat | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve a shift-invert EVP near ``target`` using SLEPc Krylov-Schur."""
    if mass_operator is not None:
        return _eigs_generalized(
            operator,
            mass_operator,
            n_modes,
            target,
            ncv=ncv,
            solver_cfg=solver_cfg,
            cfg=cfg,
        )

    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    require_complex(solver_cfg)
    ndof = _operator_shape(operator)[0]
    _warn_if_shifted_operator_is_ill_conditioned(operator, target, cfg)

    if ndof <= 2 or n_modes >= ndof - 1:
        dense = np.asarray(
            petsc_to_scipy(operator).toarray() if _is_petsc_matrix(operator) else operator.toarray()
        )
        all_vals, all_vecs = np.linalg.eig(dense)
        order = np.argsort(np.abs(all_vals - target))
        sel = order[:n_modes]
        eigenvalues = all_vals[sel]
        eigenvectors = all_vecs[:, sel]
        if not return_info:
            return eigenvalues, eigenvectors
        dense_info: dict[str, object] = {
            "method": "dense",
            "ndof": int(ndof),
            "nev_requested": int(n_modes),
            "ncv_used": None,
            "nconv": int(eigenvalues.size),
            "eps_iterations": 0,
            "eps_reason": "dense",
            "ksp_reason": "dense",
            "eps_tol": None,
            "eps_max_it": None,
            "st_solver": "dense",
            "residual_estimates": np.full(eigenvalues.size, np.nan, dtype=float),
        }
        return eigenvalues, eigenvectors, dense_info

    _ecfg = _resolve_solver_cfg_subsection(solver_cfg, "LINSTABPY_EIGEN")
    st_solver = cfg.eigen.st_solver
    use_root_scatter = (
        not _is_petsc_matrix(operator)
        and PETSc.COMM_WORLD.getSize() > 1
        and st_solver in ITERATIVE_EIGEN_SOLVER_TYPES
        and (cfg.mpi.root_scatter)
    )
    if _is_petsc_matrix(operator):
        A = operator.copy()
    elif use_root_scatter:
        rank = PETSc.COMM_WORLD.getRank()
        A = scipy_to_petsc_distributed_from_root(
            operator if rank == 0 else None,
            shape=operator.shape,
            root=0,
            comm=PETSc.COMM_WORLD,
        )
    else:
        A = scipy_to_petsc(operator, solver_cfg)

    eps = SLEPc.EPS().create()
    eps.setOperators(A)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    default_ncv = min(max(2 * n_modes + 3, 30), ndof - 1)
    requested_ncv = ncv if ncv is not None else cfg.eigen.ncv
    ncv = _resolve_ncv(
        ndof=ndof,
        n_modes=n_modes,
        default_ncv=default_ncv,
        requested_ncv=requested_ncv,
    )
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    conv_test_name = cfg.eigen.conv_test
    if conv_test_name:
        eps.setConvergenceTest(_resolve_eps_convergence_test(conv_test_name))
    eps_tol = cfg.eigen.eps_tol
    eps_max_it = cfg.eigen.eps_max_it(ndof)
    eps.setTolerances(tol=eps_tol, max_it=eps_max_it)
    residual_error_type_name = cfg.eigen.residual_error_type
    residual_error_type = _resolve_eps_error_type(residual_error_type_name)
    monitor_state: dict[str, float | int] | None = None
    if cfg.eigen.monitor:
        monitor_state = _install_eps_monitor(eps, cfg.eigen.monitor_every)
    initial_vecs: list[PETSc.Vec] = []
    if initial_space is not None:
        initial_arr = np.asarray(initial_space, dtype=complex)
        if initial_arr.ndim == 1:
            initial_arr = initial_arr.reshape(-1, 1)
        if initial_arr.ndim != 2:
            raise ValueError("initial_space must be a 1-D vector or a 2-D column matrix")
        if initial_arr.shape[0] != ndof:
            raise ValueError(
                f"initial_space row count {initial_arr.shape[0]} does not match operator size {ndof}"
            )
        n_seed = min(initial_arr.shape[1], max(1, ncv - 1))
        if n_seed > 0:
            initial_vecs = numpy_columns_to_petsc_vecs(A, initial_arr[:, :n_seed])
            eps.setInitialSpace(initial_vecs)

    # Shift-invert spectral transformation; linear solver backend is selectable.
    #
    # LINSTABPY_EIGEN_ST_SOLVER:
    #   mumps (default) -> direct LU factorization (robust, memory-heavy)
    #   iterative       -> Krylov linear solve in ST (scalable, may require tuning)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    # Keep the ST shift explicit instead of relying only on EPS.setTarget().
    # In crowded target neighborhoods like Martini Example 3, this makes the
    # intended shift-invert center unambiguous in PETSc/SLEPc.
    st.setShift(target)
    ksp = st.getKSP()
    pc = ksp.getPC()

    # Auto-route: when factor_backend="auto", resolve to a concrete backend now
    # that ndof is known.  The config carries intent "auto"; this is the only
    # site where ndof is available before the backend is applied to the PC.
    if cfg.eigen.st_factor_backend == "auto":
        _strumpack_avail = factor_backend_available("strumpack")
        _threshold = cfg.eigen.auto_route_dof_threshold
        _concrete = resolve_auto_factor_backend(
            ndof,
            threshold=_threshold,
            strumpack_available=_strumpack_avail,
        )
        logger.info(
            "factor_backend=auto routing: ndof=%d threshold=%d strumpack_available=%s -> %s",
            ndof,
            _threshold,
            _strumpack_avail,
            _concrete,
        )
        # Replace the frozen cfg with a copy that carries the concrete backend.
        # Only the eigen sub-object changes; no other fields are touched.
        _eigen_concrete = dataclasses.replace(cfg.eigen, st_factor_backend=_concrete)
        cfg = dataclasses.replace(cfg, eigen=_eigen_concrete)

    _configure_st_for_eigenvalue(st_solver, _ecfg, ksp, pc, st, cfg=cfg)
    # Note: the "iterative" branch inside _configure_st_for_eigenvalue handles
    # the FieldSplit physics-split when pc_type in {"fieldsplit", "fs"}.
    # Re-apply FieldSplit IS setup here since the helper cannot access ndof/A.
    if st_solver in ITERATIVE_EIGEN_SOLVER_TYPES:
        pc_type_used = cfg.eigen.pc_type
        if pc_type_used in {"fieldsplit", "fs"} and ndof % 5 == 0:
            ngp = ndof // 5
            fluid_idx = np.concatenate(
                (
                    np.arange(0, 3 * ngp, dtype=np.int32),
                    np.arange(4 * ngp, 5 * ngp, dtype=np.int32),
                )
            )
            w_idx = np.arange(3 * ngp, 4 * ngp, dtype=np.int32)
            rstart, rend = A.getOwnershipRange()
            fluid_local = fluid_idx[(fluid_idx >= rstart) & (fluid_idx < rend)]
            w_local = w_idx[(w_idx >= rstart) & (w_idx < rend)]
            is_fluid = PETSc.IS().createGeneral(fluid_local, comm=PETSc.COMM_WORLD)
            is_w = PETSc.IS().createGeneral(w_local, comm=PETSc.COMM_WORLD)
            pc.setFieldSplitIS(("fluid", is_fluid), ("w", is_w))
            pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # Allow PETSc/SLEPc options DB overrides from command-line flags.
    # For direct solvers (mumps/lu) ksp.setFromOptions() is a no-op but harmless.
    # For iterative/inexact solvers the options DB can override JSONC-level settings.
    ksp.setFromOptions()
    st.setFromOptions()
    eps.setFromOptions()
    logger.info(
        "ST KSP configuration: type=%s, pc=%s, factor_solver=%s, pc_side=%s",
        _safe_petsc_info(ksp.getType),
        _safe_petsc_info(pc.getType),
        _safe_petsc_info(pc.getFactorSolverType, "?"),
        _safe_petsc_info(ksp.getPCSide, "?"),
    )
    # "inexact" inner solves can satisfy the preconditioned residual while
    # the true residual remains large; default true_residual=True for that path.
    true_residual = cfg.eigen.true_residual
    if hasattr(eps, "setTrueResidual") and true_residual:
        eps.setTrueResidual(True)

    eps_setup_seconds = 0.0
    eps_solve_seconds = 0.0
    try:
        setup_t0 = time.perf_counter()
        eps.setUp()
        eps_setup_seconds = time.perf_counter() - setup_t0
    except Exception as exc:
        _raise_eps_setup_failure(
            exc=exc,
            A=A,
            eps=eps,
            st=st,
            ksp=ksp,
            pc=pc,
            ndof=ndof,
            n_modes=n_modes,
            ncv=ncv,
            target=target,
            st_solver=st_solver,
        )

    resolved_factor_backend = cfg.eigen.st_factor_backend
    probe_enabled = cfg.eigen.backend_probe.enabled
    probe_cfg = {"enabled": probe_enabled} if probe_enabled is not None else {}
    backend_probe_result: dict[str, float | int | str | None] | None = None
    if _temporal_core.should_probe_backend(resolved_factor_backend, probe_cfg, st_solver=st_solver):
        probe = _temporal_core.shifted_ksp_probe_residual(st.getKSP())
        backend_probe_result = probe
        residual_rel = float(probe["residual_rel"])
        residual_rel_max = cfg.eigen.backend_probe.residual_rel_max
        preconditioner_role = backend_probe_is_preconditioner_role(st_solver)
        probe_role = "preconditioner" if preconditioner_role else "direct"
        logger.info(
            "Factor backend accuracy probe: backend=%s role=%s residual_abs=%.3e residual_rel=%.3e "
            "tol=%.3e ksp_type=%s pc_type=%s factor_solver=%s iterations=%s reason=%s",
            resolved_factor_backend,
            probe_role,
            float(probe["residual_abs"]),
            residual_rel,
            residual_rel_max,
            probe["ksp_type"],
            probe["pc_type"],
            probe["pc_factor_solver_type"],
            probe["ksp_iterations"],
            probe["ksp_reason"],
        )
        rejection_limit = residual_rel_max
        if preconditioner_role:
            # A factor backend used under an iterative outer KSP/SLEPc solve is
            # accepted primarily by that outer convergence test.  The bare LU
            # apply can be inexact (for example STRUMPACK compression) while a
            # healthy FGMRES outer still drives the true shifted residual down.
            # Keep a relaxed sanity ceiling so a completely ineffective
            # preconditioner still fails fast without weakening the direct gate.
            rejection_limit = max(residual_rel_max, PRECONDITIONER_PROBE_RESIDUAL_REL_MAX)
            if residual_rel > residual_rel_max:
                logger.info(
                    "Factor backend probe residual %.3e exceeds direct-solve tol %.3e, "
                    "but backend is in preconditioner role; applying relaxed sanity tol %.3e",
                    residual_rel,
                    residual_rel_max,
                    rejection_limit,
                )
        if residual_rel > rejection_limit:
            _raise_backend_probe_failure(
                backend=resolved_factor_backend,
                residual_rel=residual_rel,
                residual_rel_max=residual_rel_max,
            )

    logger.info(
        "  SLEPc EPS: sigma = %.6g%+.6gj, nev=%d, ncv=%d, tol=%g, conv_test=%s, true_residual=%s ...",
        np.real(target),
        np.imag(target),
        n_modes,
        ncv,
        eps_tol,
        conv_test_name or "default",
        true_residual,
    )
    try:
        solve_t0 = time.perf_counter()
        eps.solve()
        eps_solve_seconds = time.perf_counter() - solve_t0
    except Exception as exc:
        _raise_eps_solve_failure(
            exc=exc,
            eps=eps,
            ksp=ksp,
            target=target,
            monitor_state=monitor_state,
        )

    nconv = eps.getConverged()
    if nconv == 0:
        details = [
            f"target={target}",
            f"nev={n_modes}",
            f"ncv={ncv}",
            f"eps_tol={eps_tol:g}",
            f"eps_iterations={_safe_petsc_info(eps.getIterationNumber, '?')}",
            f"eps_reason={_safe_petsc_info(eps.getConvergedReason, '?')}",
            f"ksp_reason={_safe_petsc_info(ksp.getConvergedReason, '?')}",
            f"eps_setup_seconds={eps_setup_seconds:.2f}",
            f"eps_solve_seconds={eps_solve_seconds:.2f}",
            f"true_residual={bool(true_residual)}",
            f"conv_test={conv_test_name or 'default'}",
        ]
        details.extend(_monitor_state_details(monitor_state))
        raise RuntimeError("SLEPc EPS did not converge for any eigenpair: " + ", ".join(details))
    if nconv < n_modes:
        logger.warning(
            "SLEPc EPS converged %d/%d eigenpairs; returning partial result",
            nconv,
            n_modes,
        )

    extract_all_converged = cfg.eigen.extract_all_converged
    n_extract = int(nconv) if extract_all_converged else min(int(nconv), int(n_modes))
    eigenvalues = np.empty(n_extract, dtype=complex)
    eigenvectors = np.empty((ndof, n_extract), dtype=complex)
    residual_estimates = np.full(n_extract, np.nan, dtype=float)
    residual_absolute = np.full(n_extract, np.nan, dtype=float)
    residual_relative = np.full(n_extract, np.nan, dtype=float)
    residual_backward = np.full(n_extract, np.nan, dtype=float)

    vr = A.createVecRight()
    vi = A.createVecRight()
    try:
        for i in range(n_extract):
            eig = eps.getEigenpair(i, vr, vi)
            eigenvalues[i] = eig
            eigenvectors[:, i] = vec_to_numpy(vr)
            residual_absolute[i] = float(
                _safe_petsc_info(
                    lambda: eps.computeError(i, SLEPc.EPS.ErrorType.ABSOLUTE),
                    np.nan,
                )
            )
            residual_relative[i] = float(
                _safe_petsc_info(
                    lambda: eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE),
                    np.nan,
                )
            )
            residual_backward[i] = float(
                _safe_petsc_info(
                    lambda: eps.computeError(i, SLEPc.EPS.ErrorType.BACKWARD),
                    np.nan,
                )
            )
            residual_estimates[i] = float(
                _safe_petsc_info(lambda: eps.computeError(i, residual_error_type), np.nan)
            )
    finally:
        vr.destroy()
        vi.destroy()
        for vec in initial_vecs:
            vec.destroy()

    if extract_all_converged:
        order = _nearest_order(eigenvalues, target)
        keep = order[: min(n_modes, order.size)]
        eigenvalues = eigenvalues[keep]
        eigenvectors = eigenvectors[:, keep]
        residual_estimates = residual_estimates[keep]
        residual_absolute = residual_absolute[keep]
        residual_relative = residual_relative[keep]
        residual_backward = residual_backward[keep]

    info: dict[str, object] = {
        "method": "slepc",
        "ndof": int(ndof),
        "nev_requested": int(n_modes),
        "ncv_used": int(ncv),
        "nconv": int(nconv),
        "nextract_total": int(n_extract),
        "extract_all_converged": bool(extract_all_converged),
        "eps_iterations": int(_safe_petsc_info(eps.getIterationNumber, 0)),
        "eps_reason": str(_safe_petsc_info(eps.getConvergedReason, "?")),
        "ksp_reason": str(_safe_petsc_info(ksp.getConvergedReason, "?")),
        "conv_test": str(_safe_petsc_info(eps.getConvergenceTest, "?")),
        "eps_tol": float(eps_tol),
        "eps_max_it": int(eps_max_it),
        "eps_setup_seconds": float(eps_setup_seconds),
        "eps_solve_seconds": float(eps_solve_seconds),
        "st_solver": st_solver,
        "true_residual": bool(true_residual),
        "residual_error_type": str(residual_error_type_name),
        "residual_estimates": residual_estimates,
        "residual_absolute": residual_absolute,
        "residual_relative": residual_relative,
        "residual_backward": residual_backward,
    }
    info["backend_probe_factor_backend"] = resolved_factor_backend
    info["backend_probe_ran"] = backend_probe_result is not None
    if backend_probe_result is not None:
        info.update({f"backend_probe_{key}": value for key, value in backend_probe_result.items()})
    info.update({key: value for key, value in (monitor_state or {}).items()})
    eps.destroy()
    A.destroy()

    if return_info:
        return eigenvalues, eigenvectors, info
    return eigenvalues, eigenvectors


def eigenvalues(
    L0: sp.spmatrix | PETSc.Mat,
    n_modes: int,
    target: float | complex,
    W_diag: np.ndarray | None = None,
    ncv: int | None = None,
    initial_space: np.ndarray | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Compute temporal eigenvalues ``omega = eig(i*L0)`` near a target.

    The eigensolver operates on ``i*L0`` so that ``omega = eig(i*L0)``.
    With the ``exp(-i ω t)`` convention, temporally unstable modes satisfy
    ``Im(omega) > 0``.

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
    ncv:
        Optional Arnoldi subspace dimension. When omitted, use the generic
        default ``max(2*n_modes+3, 30)`` clipped to the matrix size.
    solver_cfg:
        Optional solver config dict (from JSONC ``"solver"`` section).

    Returns
    -------
    omega:
        Temporal eigenvalues ``omega = eig(i*L0)``.
    modes:
        Right eigenvectors corresponding to ``omega``.
    """
    ndof = _validate_sparse_square_operator("L0", L0)
    _validate_n_modes(n_modes, ndof)
    _validate_finite_number("target", target)
    if W_diag is not None:
        _validate_weight_vector("W_diag", W_diag, ndof)
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    operator, _ = _check_inputs_shape(L0, n_modes)

    temporal_operator = _scaled_operator(operator, 1j)
    omega = complex(target)
    if return_info:
        eigenvalues, eigenvectors, info = _solve_near_target(
            temporal_operator,
            n_modes,
            omega,
            ncv=ncv,
            initial_space=initial_space,
            solver_cfg=solver_cfg,
            cfg=cfg,
            return_info=True,
        )
    else:
        eigenvalues, eigenvectors = _solve_near_target(
            temporal_operator,
            n_modes,
            omega,
            ncv=ncv,
            initial_space=initial_space,
            solver_cfg=solver_cfg,
            cfg=cfg,
        )
        info = {}
    order = _nearest_order(eigenvalues, omega)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    _reorder_info_residuals(info, order)

    if W_diag is not None:
        eigenvectors = _normalize_in_weighted_inner(eigenvectors, np.asarray(W_diag))
        if eigenvectors.shape[1] > 0:
            v0 = eigenvectors[:, 0]
            weighted_norm_error = np.abs(np.vdot(v0, np.asarray(W_diag) * v0) - 1.0)
            weighted_norm_tol = cfg.eigen.weighted_norm_tol
            if weighted_norm_error > weighted_norm_tol:
                logger.warning(
                    "Leading weighted norm check failed: |v^H W v - 1|=%.3e > %.3e",
                    weighted_norm_error,
                    weighted_norm_tol,
                )
            info["leading_weighted_norm_error"] = float(weighted_norm_error)
            info["weighted_norm_tol"] = float(weighted_norm_tol)

    if return_info:
        return eigenvalues, eigenvectors, info
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
    L0: sp.spmatrix | PETSc.Mat,
    n_modes: int,
    target: float | complex,
    W_diag: np.ndarray,
    invW_diag: np.ndarray,
    ncv: int | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
    temporal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute adjoint temporal eigenvalues near a target with W weighting.

    The adjoint operator is formed as ``W^{-1} L0^* W`` and eigendecomposed
    with shift-invert at ``target``.

    Parameters
    ----------
    L0 : sparse matrix or PETSc.Mat
        State matrix from ``build_lnse_operator``.
    n_modes : int
        Number of eigenpairs requested.
    target : float or complex
        Target eigenvalue for the shift-invert search.
    W_diag : ndarray
        Diagonal of the energy-norm weight matrix.
    invW_diag : ndarray
        Diagonal of the inverse weight matrix.
    ncv : int or None, optional
        Arnoldi subspace dimension.
    solver_cfg : dict or None, optional
        Solver configuration dictionary.
    temporal : bool, optional
        When True, interpret *target* as a temporal eigenvalue omega (from
        ``temporal_eigenvalues``) and return eigenvalues in the same
        convention.  The target is converted internally to the adjoint
        eigenvalue space (``target_internal = 1j * conj(target)``), and
        the returned eigenvalues are the conjugates of the direct temporal
        eigenvalues so that ``adjoint_omegas == conj(temporal_omegas)``.
        Default is False (raw adjoint eigenvalue convention).
    """
    ndof = _validate_sparse_square_operator("L0", L0)
    _validate_n_modes(n_modes, ndof)
    _validate_finite_number("target", target)
    _validate_weight_vector("W_diag", W_diag, ndof)
    _validate_weight_vector("invW_diag", invW_diag, ndof)
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    operator, _ = _check_inputs_shape(L0, n_modes)
    if _is_petsc_matrix(operator):
        raise NotImplementedError(
            "adjoint_eigenvalues() does not yet support PETSc matrices; "
            "assemble with backend='scipy' for adjoint workflows"
        )
    adjoint_op = _build_adjoint_operator(operator, W_diag, invW_diag)

    if temporal:
        # eigenvalues returns omega = eig(i*L0).
        # adjoint_eigenvalues operates on W^{-1} L0^H W whose eigenvalues
        # are lambda_adj = conj(eig(L0)) = conj(-i*omega) = i*conj(omega).
        # To search near the adjoint eigenvalue corresponding to omega_target,
        # convert: target_internal = i * conj(omega_target).
        omega_internal = 1j * np.conj(complex(target))
    else:
        omega_internal = complex(target)

    eigenvalues, eigenvectors = _solve_near_target(
        adjoint_op,
        n_modes,
        omega_internal,
        ncv=ncv,
        solver_cfg=solver_cfg,
        cfg=cfg,
    )
    eigenvalues, eigenvectors = _sort_nearest(eigenvalues, eigenvectors, omega_internal)
    eigenvectors = _normalize_in_weighted_inner(eigenvectors, np.asarray(W_diag))

    if temporal:
        # Convert back to temporal omega convention:
        # omega_temporal = -i * conj(lambda_adj)
        eigenvalues = -1j * np.conj(eigenvalues)

    return eigenvalues, eigenvectors


def spatial_eigenvalues(
    Lc: sp.spmatrix | PETSc.Mat,
    Lw: sp.spmatrix | PETSc.Mat,
    R: sp.spmatrix | PETSc.Mat,
    omega: float | complex,
    n_modes: int,
    target: float | complex,
    ncv: int | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the spatial EVP `(Lc + omega Lw) @ v = alpha R @ v`.

    Returns:
        alpha : ndarray, shape (n_modes,)
            Spatial eigenvalues.
        modes : ndarray, shape (N, n_modes)
            Right eigenvectors.
    """
    ndof = _validate_sparse_square_operator("Lc", Lc)
    _validate_n_modes(n_modes, ndof)
    for name, operator in (("Lw", Lw), ("R", R)):
        other_ndof = _validate_sparse_square_operator(name, operator)
        if other_ndof != ndof:
            raise ValueError(f"{name} shape must match Lc shape")
    _validate_finite_number("omega", omega)
    _validate_finite_number("target", target)
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    lhs = Lc.copy() if _is_petsc_matrix(Lc) else sp.csc_matrix(Lc, dtype=complex)
    if _is_petsc_matrix(lhs):
        lhs.axpy(omega, Lw, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    else:
        lhs = lhs + omega * sp.csc_matrix(Lw, dtype=complex)

    eigenvalues, eigenvectors = _solve_near_target(
        lhs,
        n_modes,
        complex(target),
        ncv=ncv,
        solver_cfg=solver_cfg,
        cfg=cfg,
        mass_operator=R,
    )
    return eigenvalues, eigenvectors


_parent_module = sys.modules.get(__package__)
if _parent_module is not None:
    _parent_module.eigenvalues = eigenvalues
    _parent_module.adjoint_eigenvalues = adjoint_eigenvalues
    _parent_module.spatial_eigenvalues = spatial_eigenvalues
