"""Temporal and adjoint eigenvalue solvers using SLEPc EPS."""

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
from ..utils._solver_cfg import _scfg, _scfg_bool, _scfg_float, _scfg_int

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
                f"Invalid gmres_orthogonalization={orth!r}; "
                "expected modified|mgs|classical|cgs"
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


def _sort_nearest(values: np.ndarray, vectors: np.ndarray, target: complex) -> tuple[np.ndarray, np.ndarray]:
    """Sort eigenpairs by distance to ``target`` with deterministic tie-breaking."""
    order = _nearest_order(values, target)
    return values[order], vectors[:, order]


def _warn_if_shifted_operator_is_ill_conditioned(
    operator: sp.csc_matrix | PETSc.Mat,
    target: complex,
    solver_cfg: dict | None = None,
) -> None:
    """Warn when ``cond(operator - target*I)`` exceeds a configurable threshold.

    The exact dense condition number can be expensive for large sparse systems, so
    this check is limited to moderate sizes by ``cond_warn_max_n``.
    """
    _ecfg = solver_cfg.get("eigen", {}) if solver_cfg is not None else None
    if not _scfg_bool(_ecfg, "cond_warn", "LINSTAB2D_EIGEN_COND_WARN", True):
        return

    max_n = _scfg_int(_ecfg, "cond_warn_max_n", "LINSTAB2D_EIGEN_COND_WARN_MAX_N", 1500)
    threshold = _scfg_float(_ecfg, "cond_warn_threshold", "LINSTAB2D_EIGEN_COND_WARN_THRESHOLD", 1e12)
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


def _resolve_eps_convergence_test(name: str) -> SLEPc.EPS.Conv:
    """Map config strings to SLEPc EPS convergence tests."""
    key = str(name).strip().lower()
    mapping = {
        "abs": SLEPc.EPS.Conv.ABS,
        "absolute": SLEPc.EPS.Conv.ABS,
        "rel": SLEPc.EPS.Conv.REL,
        "relative": SLEPc.EPS.Conv.REL,
        "norm": SLEPc.EPS.Conv.NORM,
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(
            f"Invalid EPS convergence test {name!r}; expected abs|rel|norm"
        ) from exc


def _resolve_eps_error_type(name: str) -> SLEPc.EPS.ErrorType:
    """Map config strings to SLEPc EPS residual/error types."""
    key = str(name).strip().lower()
    mapping = {
        "abs": SLEPc.EPS.ErrorType.ABSOLUTE,
        "absolute": SLEPc.EPS.ErrorType.ABSOLUTE,
        "rel": SLEPc.EPS.ErrorType.RELATIVE,
        "relative": SLEPc.EPS.ErrorType.RELATIVE,
        "bwd": SLEPc.EPS.ErrorType.BACKWARD,
        "backward": SLEPc.EPS.ErrorType.BACKWARD,
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(
            f"Invalid EPS error type {name!r}; expected absolute|relative|backward"
        ) from exc


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


def _install_eps_monitor(eps: SLEPc.EPS, every: int) -> dict[str, float | int]:
    """Log sparse progress updates from SLEPc EPS iterations."""
    interval = max(int(every), 1)
    state: dict[str, float | int] = {
        "last_nconv": -1,
        "last_iteration": 0,
        "last_min_err": float("nan"),
        "last_max_err": float("nan"),
    }

    def _monitor(
        eps_obj: SLEPc.EPS,
        iteration: int,
        nconv: int,
        eigs: list[complex],
        errs: list[float],
    ) -> None:
        should_log = iteration == 1 or iteration % interval == 0 or nconv != state["last_nconv"]
        if not should_log:
            return
        finite_errs = [float(val) for val in errs if np.isfinite(val)]
        max_err = max(finite_errs) if finite_errs else float("nan")
        min_err = min(finite_errs) if finite_errs else float("nan")
        state["last_iteration"] = int(iteration)
        state["last_nconv"] = int(nconv)
        state["last_min_err"] = float(min_err)
        state["last_max_err"] = float(max_err)
        PETSc.Sys.Print(
            f"[eigs] EPS iter={int(iteration)} nconv={int(nconv)} "
            f"err_range=[{min_err:.3e}, {max_err:.3e}]"
        )

    eps.setTrackAll(True)
    eps.setMonitor(_monitor)
    return state


def _monitor_state_details(state: dict[str, float | int] | None) -> list[str]:
    """Format the latest EPS monitor state for logs and errors."""
    if not state:
        return []
    details: list[str] = []
    last_iteration = int(state.get("last_iteration", 0))
    if last_iteration > 0:
        details.append(f"monitor_iter={last_iteration}")
    last_nconv = int(state.get("last_nconv", -1))
    if last_nconv >= 0:
        details.append(f"monitor_nconv={last_nconv}")
    last_min_err = float(state.get("last_min_err", np.nan))
    if np.isfinite(last_min_err):
        details.append(f"monitor_min_err={last_min_err:.3e}")
    last_max_err = float(state.get("last_max_err", np.nan))
    if np.isfinite(last_max_err):
        details.append(f"monitor_max_err={last_max_err:.3e}")
    return details


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

    raise RuntimeError("SLEPc EPS.setUp failed: " + ", ".join(str(item) for item in details)) from exc


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
    raise RuntimeError("SLEPc EPS.solve failed: " + ", ".join(str(item) for item in details)) from exc


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


def _eigs_near_target(
    operator: sp.csc_matrix | PETSc.Mat,
    n_modes: int,
    target: complex,
    *,
    ncv: int | None = None,
    initial_space: np.ndarray | None = None,
    solver_cfg: dict | None = None,
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve a shift-invert EVP near ``target`` using SLEPc Krylov-Schur."""
    require_complex(solver_cfg)
    ndof = _operator_shape(operator)[0]
    _warn_if_shifted_operator_is_ill_conditioned(operator, target, solver_cfg)

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

    _ecfg = solver_cfg.get("eigen", {}) if solver_cfg is not None else None
    st_solver = _scfg(_ecfg, "st_solver", "LINSTAB2D_EIGEN_ST_SOLVER", "mumps")
    use_root_scatter = (
        not _is_petsc_matrix(operator)
        and PETSc.COMM_WORLD.getSize() > 1
        and st_solver in {"iterative", "krylov", "slepc"}
        and _scfg_bool(solver_cfg, "root_scatter", "LINSTAB2D_PETSC_ROOT_SCATTER", False)
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
    requested_ncv = ncv
    if requested_ncv is None and _ecfg is not None and "ncv" in _ecfg:
        requested_ncv = int(_ecfg["ncv"])
    ncv = _resolve_ncv(
        ndof=ndof,
        n_modes=n_modes,
        default_ncv=default_ncv,
        requested_ncv=requested_ncv,
    )
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    conv_test_name = _scfg(_ecfg, "conv_test", "LINSTAB2D_EIGEN_CONV_TEST", "")
    if conv_test_name:
        eps.setConvergenceTest(_resolve_eps_convergence_test(conv_test_name))
    eps_tol = _scfg_float(_ecfg, "eps_tol", "LINSTAB2D_EIGEN_EPS_TOL", 1e-10)
    eps_max_it = _scfg_int(_ecfg, "eps_max_it", "LINSTAB2D_EIGEN_EPS_MAX_IT", max(500, 10 * ndof))
    eps.setTolerances(tol=eps_tol, max_it=eps_max_it)
    residual_error_type_name = _scfg(
        _ecfg,
        "residual_error_type",
        "LINSTAB2D_EIGEN_RESIDUAL_ERROR_TYPE",
        "relative",
    )
    residual_error_type = _resolve_eps_error_type(residual_error_type_name)
    monitor_state: dict[str, float | int] | None = None
    if _scfg_bool(_ecfg, "monitor", "LINSTAB2D_EIGEN_MONITOR", False):
        monitor_state = _install_eps_monitor(
            eps,
            _scfg_int(_ecfg, "monitor_every", "LINSTAB2D_EIGEN_MONITOR_EVERY", 10),
        )
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
    # LINSTAB2D_EIGEN_ST_SOLVER:
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

    if st_solver in {"mumps", "direct", "lu"}:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc.setType(PETSc.PC.Type.LU)
        if st_solver == "mumps":
            pc.setFactorSolverType("mumps")
    elif st_solver in {"iterative", "krylov", "slepc"}:
        ksp_type = _scfg(_ecfg, "ksp_type", "LINSTAB2D_EIGEN_KSP_TYPE", "gmres")
        pc_type = _scfg(_ecfg, "pc_type", "LINSTAB2D_EIGEN_PC_TYPE", "asm")
        pc_factor_solver_type = _scfg(
            _ecfg,
            "pc_factor_solver_type",
            "LINSTAB2D_EIGEN_PC_FACTOR_SOLVER_TYPE",
            "",
        )
        pc_side = _scfg(_ecfg, "ksp_pc_side", "LINSTAB2D_EIGEN_KSP_PC_SIDE", "left")
        ksp_rtol = _scfg_float(_ecfg, "ksp_rtol", "LINSTAB2D_EIGEN_KSP_RTOL", 1e-8)
        ksp_max_it = _scfg_int(_ecfg, "ksp_max_it", "LINSTAB2D_EIGEN_KSP_MAX_IT", 20000)
        asm_overlap = _scfg_int(_ecfg, "asm_overlap", "LINSTAB2D_EIGEN_ASM_OVERLAP", 2)
        sub_ksp_type = _scfg(_ecfg, "sub_ksp_type", "LINSTAB2D_EIGEN_SUB_KSP_TYPE", "preonly")
        sub_pc_type = _scfg(_ecfg, "sub_pc_type", "LINSTAB2D_EIGEN_SUB_PC_TYPE", "lu")
        ksp.setType(ksp_type)
        if pc_side == "left":
            ksp.setPCSide(PETSc.PC.Side.LEFT)
        elif pc_side == "right":
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
        elif pc_side in {"symmetric", "symm"}:
            ksp.setPCSide(PETSc.PC.Side.SYMMETRIC)
        else:
            raise ValueError(
                "Invalid LINSTAB2D_EIGEN_KSP_PC_SIDE "
                f"'{pc_side}'; expected left|right|symmetric"
            )
        if pc_type in {"fieldsplit", "fs"}:
            pc.setType(PETSc.PC.Type.FIELDSPLIT)
        else:
            pc.setType(pc_type)
            if pc_type == "lu":
                factor_solver_type = pc_factor_solver_type if pc_factor_solver_type else "mumps"
                pc.setFactorSolverType(factor_solver_type)
        ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)
        ksp.setErrorIfNotConverged(True)
        if ksp_type in {"gmres", "fgmres"}:
            _apply_ksp_jsonc_options(ksp, _ecfg)
        if pc_type in {"fieldsplit", "fs"}:
            # Physics-aware 5-variable split for compressible LNSE state:
            # fluid = [rho, u, v, T], spanwise = [w]
            # Assumes state layout [rho(ngp), u(ngp), v(ngp), w(ngp), T(ngp)].
            if ndof % 5 == 0:
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
    else:
        raise ValueError(
            f"Unsupported LINSTAB2D_EIGEN_ST_SOLVER={st_solver!r}; "
            "expected mumps|direct|lu|iterative|krylov|slepc"
        )

    # Allow PETSc/SLEPc options DB overrides from command-line flags.
    if not (st_solver in {"iterative", "krylov", "slepc"} and ksp_type in {"gmres", "fgmres"}):
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
    true_residual = _scfg_bool(_ecfg, "true_residual", "LINSTAB2D_EIGEN_TRUE_RESIDUAL", False)
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

    logger.info(
        "  SLEPc EPS: sigma = %.6g%+.6gj, nev=%d, ncv=%d, tol=%g, conv_test=%s, true_residual=%s ...",
        np.real(target), np.imag(target), n_modes, ncv, eps_tol,
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
        raise RuntimeError(
            "SLEPc EPS did not converge for any eigenpair: " + ", ".join(details)
        )
    if nconv < n_modes:
        logger.warning(
            "SLEPc EPS converged %d/%d eigenpairs; returning partial result",
            nconv, n_modes,
        )

    extract_all_converged = _scfg_bool(
        _ecfg, "extract_all_converged", "LINSTAB2D_EIGEN_EXTRACT_ALL_CONVERGED", False,
    )
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
        Temporal eigenvalues ``omega = eig(i*L0) = -ω_OS``.
    modes:
        Right eigenvectors corresponding to ``omega``.
    """
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
        )
        info = {}
    order = _nearest_order(eigenvalues, omega)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    _reorder_info_residuals(info, order)

    if W_diag is not None:
        eigenvectors = _normalize_in_weighted_inner(eigenvectors, np.asarray(W_diag))
        if eigenvectors.shape[1] > 0:
            _ecfg = solver_cfg.get("eigen", {}) if solver_cfg is not None else None
            v0 = eigenvectors[:, 0]
            weighted_norm_error = np.abs(np.vdot(v0, np.asarray(W_diag) * v0) - 1.0)
            weighted_norm_tol = _scfg_float(
                _ecfg, "weighted_norm_tol", "LINSTAB2D_EIGEN_WEIGHTED_NORM_TOL", 1e-12,
            )
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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute adjoint temporal eigenvalues near a target with W weighting.

    The adjoint operator is formed as ``W^{-1} L0^* W`` and eigendecomposed
    with shift-invert at ``target``.
    """
    operator, _ = _check_inputs_shape(L0, n_modes)
    if _is_petsc_matrix(operator):
        raise NotImplementedError(
            "adjoint_eigenvalues() does not yet support PETSc matrices; "
            "assemble with backend='scipy' for adjoint workflows"
        )
    adjoint_op = _build_adjoint_operator(operator, W_diag, invW_diag)

    omega = complex(target)
    eigenvalues, eigenvectors = _eigs_near_target(
        adjoint_op,
        n_modes,
        omega,
        ncv=ncv,
        solver_cfg=solver_cfg,
    )
    eigenvalues, eigenvectors = _sort_nearest(eigenvalues, eigenvectors, omega)
    eigenvectors = _normalize_in_weighted_inner(eigenvectors, np.asarray(W_diag))

    return eigenvalues, eigenvectors
