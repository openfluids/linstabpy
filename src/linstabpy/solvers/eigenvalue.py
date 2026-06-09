# SPDX-License-Identifier: GPL-3.0-only
"""Temporal and adjoint eigenvalue solvers using SLEPc EPS.

Solver configuration — ``solver.eigen`` section in JSONC
---------------------------------------------------------

``st_solver`` selects the shift-invert inner linear solver:

* ``"mumps"`` (default) — direct LU via MUMPS.  Robust, memory-heavy.
  Requires ~3 TiB at stride-1 for the SWBLI LNSE operator.
* ``"iterative"`` / ``"krylov"`` / ``"slepc"`` — configurable Krylov inner
  solver with a wide choice of preconditioners (including ASM, BJacobi,
  FieldSplit).  Parameters are read from the *top-level* ``solver.eigen``
  section keys (``ksp_type``, ``pc_type``, …).
* ``"inexact"`` — FGMRES + ASM + local ILU inner solver tuned for
  memory-constrained large-DOF runs.  Parameters live in the nested
  ``solver.eigen.inexact`` sub-dict so they do not collide with the
  ``"iterative"`` key names.  Memory scales with preconditioner storage
  instead of the LU factor, giving 5–20× lower peak RSS at stride-1.

JSONC schema for ``"inexact"`` mode::

    "solver": {
      "eigen": {
        "st_solver": "inexact",
        "inexact": {
          "ksp_type":         "fgmres",  // inner Krylov method
          "ksp_rtol":          1e-8,     // relative tolerance
          "ksp_atol":          1e-10,    // absolute tolerance
          "ksp_max_it":        200,      // max inner iterations
          "pc_type":          "asm",     // additive Schwarz (MPI-parallel)
          "asm_overlap":       1,        // subdomain overlap
          "sub_pc_type":      "ilu",     // local subdomain factor
          "pc_factor_levels":  2         // ILU fill level
        }
      }
    }

``true_residual`` defaults to ``True`` when ``st_solver == "inexact"``.
Inexact inner solves can satisfy the preconditioned residual while the
true residual remains large; forcing true-residual checks at convergence
time catches this case.
"""

from __future__ import annotations

import logging
import time
from typing import NoReturn

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
from petsc4py import PETSc
from slepc4py import SLEPc

from linstabpy.config import Config

from ..utils._solver_cfg import (
    DIRECT_SOLVER_TYPES,
    INEXACT_EIGEN_SOLVER_TYPES,
    ITERATIVE_EIGEN_SOLVER_TYPES,
)
from ._eps_helpers import (
    _install_eps_monitor,
    _monitor_state_details,
    _resolve_eps_convergence_test,
    _resolve_eps_error_type,
    _resolve_solver_cfg_subsection,
)
from .backends import (
    apply_factor_backend,
    shifted_ksp_probe_residual,
    should_probe_backend,
)
from .petsc_utils import (
    _is_petsc_matrix,
    _operator_shape,
    numpy_columns_to_petsc_vecs,
    petsc_to_scipy,
    require_complex,
    scipy_to_petsc,
    scipy_to_petsc_distributed_from_root,
    vec_to_numpy,
)

logger = logging.getLogger(__name__)


def _apply_ksp_jsonc_options(ksp: PETSc.KSP, cfg: dict | None) -> None:
    """Apply PETSc KSP options from JSONC via the object's options prefix.

    This keeps solver choices in JSONC while still using PETSc's native
    options-based hooks for GMRES orthogonalization/refinement details that are
    not consistently exposed as direct petsc4py methods.
    """
    if not cfg:
        return

    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix() or ""
    temp_keys: list[str] = []

    def _set_opt(key_suffix: str, value: str) -> None:
        full_key = f"{prefix}{key_suffix}"
        opts.setValue(full_key, value)
        temp_keys.append(full_key)

    orth = cfg.get("gmres_orthogonalization")
    if orth is not None:
        orth_name = str(orth).strip().lower()
        if orth_name in {"modified", "modifiedgramschmidt", "mgs"}:
            _set_opt("ksp_gmres_modifiedgramschmidt", "true")
        elif orth_name in {"classical", "classicalgramschmidt", "cgs"}:
            _set_opt("ksp_gmres_classicalgramschmidt", "true")
        else:
            raise ValueError(
                f"Invalid gmres_orthogonalization={orth!r}; expected modified|mgs|classical|cgs"
            )

    refinement = cfg.get("gmres_cgs_refinement_type")
    if refinement is not None:
        refinement_name = str(refinement).strip().lower()
        valid = {"refine_never", "refine_ifneeded", "refine_always"}
        if refinement_name not in valid:
            raise ValueError(
                f"Invalid gmres_cgs_refinement_type={refinement!r}; "
                "expected refine_never|refine_ifneeded|refine_always"
            )
        _set_opt("ksp_gmres_cgs_refinement_type", refinement_name)

    restart = cfg.get("gmres_restart")
    if restart is not None:
        _set_opt("ksp_gmres_restart", str(int(restart)))

    try:
        ksp.setFromOptions()
    finally:
        for key in temp_keys:
            try:
                opts.delValue(key)
            except PETSc.Error:
                pass


def _check_inputs_shape(
    operator: sp.spmatrix | PETSc.Mat,
    n_modes: int,
) -> tuple[sp.csc_matrix | PETSc.Mat, int]:
    """Validate input dimensions for eigensolvers and return normalized operator."""
    if n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")
    nrows, ncols = _operator_shape(operator)
    if nrows != ncols:
        raise ValueError("L0 must be a square matrix")
    if n_modes > nrows:
        raise ValueError("n_modes cannot exceed matrix size")
    if _is_petsc_matrix(operator):
        # No copy here — _eigs_near_target makes its own working copy.
        return operator, nrows
    return sp.csc_matrix(operator, dtype=complex), nrows


def _normalize_in_weighted_inner(
    vectors: np.ndarray,
    W_diag: np.ndarray | None,
) -> np.ndarray:
    """Normalise columns in the weighted inner product using ``W_diag``."""
    if W_diag is None:
        return vectors

    W = np.asarray(W_diag)
    if W.ndim != 1:
        raise ValueError("W_diag must be one-dimensional")
    if vectors.shape[0] != W.size:
        raise ValueError("W_diag length must match eigenvector length")
    if np.any(W == 0):
        raise ValueError("W_diag contains zero entries; cannot normalise")

    normed = np.array(vectors, dtype=complex, copy=True)
    for i in range(vectors.shape[1]):
        v = vectors[:, i]
        norm_sq = np.real(np.vdot(v, W * v))
        if norm_sq <= 0.0:
            continue
        normed[:, i] = v / np.sqrt(norm_sq)
    return normed


def _nearest_order(values: np.ndarray, target: complex) -> np.ndarray:
    """Return the stable ordering of eigenvalues nearest to ``target``."""
    distances = np.abs(values - target)
    # Bucket tolerance for tie-breaking: eigenvalues closer than this are
    # considered equidistant from the target and sorted by value instead.
    tie_tol = 1e-10
    distance_buckets = np.rint(distances / tie_tol)
    return np.lexsort((np.imag(values), np.real(values), distance_buckets))


def _sort_nearest(
    values: np.ndarray, vectors: np.ndarray, target: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Sort eigenpairs by distance to ``target`` with deterministic tie-breaking."""
    order = _nearest_order(values, target)
    return values[order], vectors[:, order]


def _warn_if_shifted_operator_is_ill_conditioned(
    operator: sp.csc_matrix | PETSc.Mat,
    target: complex,
    cfg: Config,
) -> None:
    """Warn when ``cond(operator - target*I)`` exceeds a configurable threshold.

    The exact dense condition number can be expensive for large sparse systems, so
    this check is limited to moderate sizes by ``cond_warn_max_n``.
    """
    if not cfg.eigen.cond_warn:
        return

    max_n = cfg.eigen.cond_warn_max_n
    threshold = cfg.eigen.cond_warn_threshold
    ndof = _operator_shape(operator)[0]
    if ndof > max_n:
        return

    if _is_petsc_matrix(operator):
        try:
            operator_sp = petsc_to_scipy(operator, format="csc")
        except (NotImplementedError, TypeError):
            return
    else:
        operator_sp = operator

    shifted = operator_sp - target * sp.identity(ndof, format="csc", dtype=complex)
    try:
        kappa = float(np.linalg.cond(np.asarray(shifted.toarray())))
    except np.linalg.LinAlgError:
        logger.warning(
            "Could not estimate cond(L-sigma*M) for warning gate (ndof=%d, sigma=%s)",
            ndof,
            target,
        )
        return

    if np.isfinite(kappa) and kappa > threshold:
        logger.warning(
            "Estimated cond(L-sigma*M)=%.3e exceeds %.3e; iterative shift-invert may stall",
            kappa,
            threshold,
        )


def _resolve_ncv(
    *,
    ndof: int,
    n_modes: int,
    default_ncv: int,
    requested_ncv: int | None,
) -> int:
    """Choose a valid Arnoldi subspace size for SLEPc EPS."""
    max_ncv = max(1, ndof - 1)
    ncv = default_ncv if requested_ncv is None else int(requested_ncv)
    return min(max(ncv, n_modes + 1), max_ncv)


def _scaled_operator(
    operator: sp.csc_matrix | PETSc.Mat,
    scale: complex,
) -> sp.csc_matrix | PETSc.Mat:
    """Return a copy of ``operator`` multiplied by ``scale``."""
    if _is_petsc_matrix(operator):
        scaled = operator.copy()
        scaled.scale(scale)
        return scaled
    return scale * operator


def _safe_petsc_info(callable_obj, default: object = "?") -> object:
    """Best-effort PETSc/SLEPc query helper for diagnostics."""
    try:
        return callable_obj()
    except Exception:
        return default


def _raise_eps_setup_failure(
    *,
    exc: Exception,
    A: PETSc.Mat,
    eps: SLEPc.EPS,
    st: SLEPc.ST,
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    ndof: int,
    n_modes: int,
    ncv: int,
    target: complex,
    st_solver: str,
) -> NoReturn:
    """Raise a richer error when SLEPc/PETSc setup fails."""
    details: list[str] = [
        f"type={type(exc).__name__}",
        f"message={exc}",
        f"ndof={ndof}",
        f"nev={n_modes}",
        f"ncv={ncv}",
        f"target={target}",
        f"eps_type={_safe_petsc_info(eps.getType)}",
        f"st_type={_safe_petsc_info(st.getType)}",
        f"st_solver={st_solver}",
        f"ksp_type={_safe_petsc_info(ksp.getType)}",
        f"pc_type={_safe_petsc_info(pc.getType)}",
        f"pc_side={_safe_petsc_info(ksp.getPCSide)}",
        f"mat_nnz={_safe_petsc_info(lambda: int(A.getInfo()['nz_used']))}",
    ]

    factor_solver = _safe_petsc_info(pc.getFactorSolverType, None)
    if factor_solver is not None and factor_solver != "?":
        details.append(f"factor_solver={factor_solver}")

    factor_matrix = _safe_petsc_info(pc.getFactorMatrix, None)
    if factor_matrix is not None and factor_matrix != "?":
        icntl7 = _safe_petsc_info(lambda: factor_matrix.getMumpsIcntl(7), None)
        infog1 = _safe_petsc_info(lambda: factor_matrix.getMumpsInfog(1), None)
        infog2 = _safe_petsc_info(lambda: factor_matrix.getMumpsInfog(2), None)
        if icntl7 not in {None, "?"}:
            details.append(f"mumps_icntl7={icntl7}")
        if infog1 not in {None, "?"}:
            details.append(f"mumps_infog1={infog1}")
        if infog2 not in {None, "?"}:
            details.append(f"mumps_infog2={infog2}")

    raise RuntimeError(
        "SLEPc EPS.setUp failed: " + ", ".join(str(item) for item in details)
    ) from exc


def _raise_eps_solve_failure(
    *,
    exc: Exception,
    eps: SLEPc.EPS,
    ksp: PETSc.KSP,
    target: complex,
    monitor_state: dict[str, float | int] | None = None,
) -> NoReturn:
    """Raise a richer error when SLEPc solve fails after setup."""
    details = [
        f"type={type(exc).__name__}",
        f"message={exc}",
        f"target={target}",
        f"eps_iterations={_safe_petsc_info(eps.getIterationNumber, '?')}",
        f"nconv={_safe_petsc_info(eps.getConverged, '?')}",
        f"eps_reason={_safe_petsc_info(eps.getConvergedReason)}",
        f"ksp_reason={_safe_petsc_info(ksp.getConvergedReason)}",
    ]
    details.extend(_monitor_state_details(monitor_state))
    raise RuntimeError(
        "SLEPc EPS.solve failed: " + ", ".join(str(item) for item in details)
    ) from exc


def _reorder_info_residuals(info: dict[str, object], order: np.ndarray) -> None:
    """Keep saved residual arrays aligned with the returned eigenpair order."""
    for key in (
        "residual_estimates",
        "residual_absolute",
        "residual_relative",
        "residual_backward",
    ):
        arr = np.asarray(info.get(key, np.array([], dtype=float)), dtype=float)
        if arr.size == order.size:
            info[key] = arr[order]


def _configure_st_for_eigenvalue(
    st_solver: str,
    ecfg: dict | None,
    ksp: "PETSc.KSP",
    pc: "PETSc.PC",
    st: "SLEPc.ST",
    cfg: Config | None = None,
) -> None:
    """Configure the SLEPc ST KSP/PC for the chosen ``st_solver`` backend.

    Parameters
    ----------
    st_solver:
        One of ``"mumps"``, ``"direct"``, ``"lu"``, ``"iterative"``,
        ``"krylov"``, ``"slepc"``, or ``"inexact"``.
    ecfg:
        The ``solver.eigen`` JSONC sub-dict (may be empty or ``None``).
    ksp:
        KSP object obtained from ``eps.getST().getKSP()``.
    pc:
        PC object obtained from ``ksp.getPC()``.
    st:
        The SLEPc ST object (unused in this helper but kept for future use).
    """
    if ecfg is None:
        ecfg = {}
    cfg = cfg if cfg is not None else Config.load({"solver": {"eigen": ecfg}})

    if st_solver in DIRECT_SOLVER_TYPES:
        ignored_iterative_keys = sorted(
            key
            for key in (
                "ksp_type",
                "pc_type",
                "pc_factor_solver_type",
                "ksp_pc_side",
                "ksp_rtol",
                "ksp_max_it",
                "asm_overlap",
                "sub_ksp_type",
                "sub_pc_type",
                "gmres_orthogonalization",
                "gmres_cgs_refinement_type",
                "gmres_restart",
            )
            if key in ecfg
        )
        if ignored_iterative_keys:
            logger.warning(
                "Ignoring iterative ST-KSP options for direct eigensolve path "
                "(st_solver=%s uses KSPPREONLY+PCLU): %s",
                st_solver,
                ", ".join(ignored_iterative_keys),
            )
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc.setType(PETSc.PC.Type.LU)
        backend = cfg.eigen.st_factor_backend
        apply_factor_backend(pc, backend, options_prefix=ksp.getOptionsPrefix() or "")

    elif st_solver in ITERATIVE_EIGEN_SOLVER_TYPES:
        ksp_type = cfg.eigen.ksp_type
        pc_type = cfg.eigen.pc_type
        pc_factor_solver_type = cfg.eigen.pc_factor_solver_type
        pc_side = cfg.eigen.ksp_pc_side
        ksp_rtol = cfg.eigen.ksp_rtol
        ksp_max_it = cfg.eigen.ksp_max_it
        asm_overlap = cfg.eigen.asm_overlap
        sub_ksp_type = cfg.eigen.sub_ksp_type
        sub_pc_type = cfg.eigen.sub_pc_type
        ksp.setType(ksp_type)
        if pc_side == "left":
            ksp.setPCSide(PETSc.PC.Side.LEFT)
        elif pc_side == "right":
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
        elif pc_side in {"symmetric", "symm"}:
            ksp.setPCSide(PETSc.PC.Side.SYMMETRIC)
        else:
            raise ValueError(
                f"Invalid LINSTABPY_EIGEN_KSP_PC_SIDE '{pc_side}'; expected left|right|symmetric"
            )
        if pc_type in {"fieldsplit", "fs"}:
            pc.setType(PETSc.PC.Type.FIELDSPLIT)
        else:
            pc.setType(pc_type)
            if pc_type == "lu":
                factor_solver_type = cfg.eigen.st_factor_backend
                apply_factor_backend(
                    pc,
                    factor_solver_type,
                    options_prefix=ksp.getOptionsPrefix() or "",
                )
        ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)
        ksp.setErrorIfNotConverged(True)
        if ksp_type in {"gmres", "fgmres"}:
            _apply_ksp_jsonc_options(ksp, ecfg)
        if pc_type == "asm":
            pc.setASMOverlap(asm_overlap)
            # Rank-local coarse solves for ASM blocks (parallel-safe default).
            # A user can still override from PETSc options DB.
            try:
                sub_ksps = pc.getASMSubKSP()
                for sub_ksp in sub_ksps:
                    sub_ksp.setType(sub_ksp_type)
                    sub_pc = sub_ksp.getPC()
                    sub_pc.setType(sub_pc_type)
            except PETSc.Error:
                pass

    elif st_solver in INEXACT_EIGEN_SOLVER_TYPES:
        # Inexact shift-invert: FGMRES outer + ASM subdomain + local ILU.
        # Memory scales with preconditioner storage, not with the LU factor.
        # All parameters are read from the nested "inexact" sub-dict to avoid
        # key collision with the "iterative" branch options.
        inexact_cfg = cfg.eigen.inexact
        ksp_type = inexact_cfg.ksp_type
        ksp_rtol = inexact_cfg.ksp_rtol
        ksp_atol = inexact_cfg.ksp_atol
        ksp_max_it = inexact_cfg.ksp_max_it
        pc_type = inexact_cfg.pc_type
        asm_overlap = inexact_cfg.asm_overlap
        sub_pc_type = inexact_cfg.sub_pc_type
        factor_levels = inexact_cfg.pc_factor_levels

        ksp.setType(ksp_type)
        ksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)
        ksp.setErrorIfNotConverged(True)
        pc.setType(pc_type)
        if pc_type in {"asm", "gasm"}:
            pc.setASMOverlap(asm_overlap)

            # Sub-KSP configuration (type and sub-PC) must be deferred until after
            # the matrix is available (i.e. after eps.setUp()).  We store the
            # desired sub-PC settings on the helper object so _eigs_near_target
            # can apply them post-setUp.  For the common single-rank / test-matrix
            # case we attempt the call immediately but catch the "no matrix" error.
            def _apply_asm_sub_pc(pc_obj=pc):
                try:
                    sub_ksps = pc_obj.getASMSubKSP()
                    for sk in sub_ksps:
                        sub_pc = sk.getPC()
                        sub_pc.setType(sub_pc_type)
                        if sub_pc_type == "ilu":
                            sub_pc.setFactorLevels(factor_levels)
                        sk.setType("preonly")
                except PETSc.Error:
                    pass

            _apply_asm_sub_pc()
        elif pc_type == "bjacobi":

            def _apply_bjacobi_sub_pc(pc_obj=pc):
                try:
                    sub_ksps = pc_obj.getBJacobiSubKSP()
                    for sk in sub_ksps:
                        sub_pc = sk.getPC()
                        sub_pc.setType(sub_pc_type)
                        if sub_pc_type == "ilu":
                            sub_pc.setFactorLevels(factor_levels)
                        sk.setType("preonly")
                except PETSc.Error:
                    pass

            _apply_bjacobi_sub_pc()

    else:
        raise ValueError(
            f"Unsupported st_solver={st_solver!r}; "
            "expected mumps|direct|lu|iterative|krylov|slepc|inexact"
        )



def _resolve_st_factor_backend(st_solver: str, ecfg: dict | None) -> str | None:
    """Resolve the factor backend used by the configured ST KSP/PC (compatibility helper)."""
    cfg = Config.load({"solver": {"eigen": ecfg or {}}})
    if st_solver in DIRECT_SOLVER_TYPES:
        return cfg.eigen.st_factor_backend
    if st_solver in ITERATIVE_EIGEN_SOLVER_TYPES:
        if cfg.eigen.pc_type != "lu":
            return None
        return cfg.eigen.st_factor_backend
    return None


def _probe_residual_rel_max(ecfg: dict | None) -> float:
    cfg = Config.load({"solver": {"eigen": ecfg or {}}})
    return cfg.eigen.backend_probe.residual_rel_max


def _raise_backend_probe_failure(
    *,
    backend: str | None,
    residual_rel: float,
    residual_rel_max: float,
) -> NoReturn:
    raise RuntimeError(
        f"factor backend '{backend}' failed accuracy probe: "
        f"||Kx-b||/||b|| = {residual_rel:.3e} > {residual_rel_max:.3e}; "
        "refusing to run EVP (set eigen.backend_probe.enabled=false to override)"
    )


def _eigs_near_target(
    operator: sp.csc_matrix | PETSc.Mat,
    n_modes: int,
    target: complex,
    *,
    ncv: int | None = None,
    initial_space: np.ndarray | None = None,
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve a shift-invert EVP near ``target`` using SLEPc Krylov-Schur."""
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
        and (
            cfg.mpi.root_scatter
        )
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
    if should_probe_backend(resolved_factor_backend, probe_cfg):
        probe = shifted_ksp_probe_residual(st.getKSP())
        backend_probe_result = probe
        residual_rel = float(probe["residual_rel"])
        residual_rel_max = cfg.eigen.backend_probe.residual_rel_max
        logger.info(
            "Factor backend accuracy probe: backend=%s residual_abs=%.3e residual_rel=%.3e "
            "tol=%.3e ksp_type=%s pc_type=%s factor_solver=%s iterations=%s reason=%s",
            resolved_factor_backend,
            float(probe["residual_abs"]),
            residual_rel,
            residual_rel_max,
            probe["ksp_type"],
            probe["pc_type"],
            probe["pc_factor_solver_type"],
            probe["ksp_iterations"],
            probe["ksp_reason"],
        )
        if residual_rel > residual_rel_max:
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


def temporal_eigenvalues(
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
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    operator, _ = _check_inputs_shape(L0, n_modes)

    temporal_operator = _scaled_operator(operator, 1j)
    omega = complex(target)
    if return_info:
        eigenvalues, eigenvectors, info = _eigs_near_target(
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
        eigenvalues, eigenvectors = _eigs_near_target(
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
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    operator, _ = _check_inputs_shape(L0, n_modes)
    if _is_petsc_matrix(operator):
        raise NotImplementedError(
            "adjoint_eigenvalues() does not yet support PETSc matrices; "
            "assemble with backend='scipy' for adjoint workflows"
        )
    adjoint_op = _build_adjoint_operator(operator, W_diag, invW_diag)

    if temporal:
        # temporal_eigenvalues returns omega = eig(i*L0).
        # adjoint_eigenvalues operates on W^{-1} L0^H W whose eigenvalues
        # are lambda_adj = conj(eig(L0)) = conj(-i*omega) = i*conj(omega).
        # To search near the adjoint eigenvalue corresponding to omega_target,
        # convert: target_internal = i * conj(omega_target).
        omega_internal = 1j * np.conj(complex(target))
    else:
        omega_internal = complex(target)

    eigenvalues, eigenvectors = _eigs_near_target(
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
