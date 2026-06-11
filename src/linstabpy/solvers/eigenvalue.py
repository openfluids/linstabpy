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
from typing import NoReturn

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
from petsc4py import PETSc
from slepc4py import SLEPc

from linstabpy.config import Config
from linstabpy.linalg import backends as _backends
from linstabpy.linalg.backends import (
    apply_factor_backend,
)

shifted_ksp_probe_residual = _backends.shifted_ksp_probe_residual
should_probe_backend = _backends.should_probe_backend

from ..utils._solver_cfg import (
    DIRECT_SOLVER_TYPES,
    INEXACT_EIGEN_SOLVER_TYPES,
    ITERATIVE_EIGEN_SOLVER_TYPES,
)
from ._eps_helpers import (
    _monitor_state_details,
)
from .petsc_utils import (
    _is_petsc_matrix,
    _operator_shape,
    petsc_to_scipy,
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
    """Deprecated compatibility shim for :func:`linstabpy.stability._solve_near_target`."""
    import warnings

    from linstabpy.stability._eigenvalues import _solve_near_target

    warnings.warn(
        "linstabpy.solvers.eigenvalue._eigs_near_target is deprecated; "
        "use linstabpy.stability._eigenvalues._solve_near_target",
        DeprecationWarning,
        stacklevel=2,
    )
    return _solve_near_target(
        operator,
        n_modes,
        target,
        ncv=ncv,
        initial_space=initial_space,
        solver_cfg=solver_cfg,
        cfg=cfg,
        return_info=return_info,
    )


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


_DEPRECATED_EXPORTS = {
    "temporal_eigenvalues": ("linstabpy.stability._eigenvalues", "eigenvalues"),
    "adjoint_eigenvalues": ("linstabpy.stability._eigenvalues", "adjoint_eigenvalues"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _DEPRECATED_EXPORTS:
        import importlib
        import warnings

        module_name, attr_name = _DEPRECATED_EXPORTS[name]
        warnings.warn(
            f"linstabpy.solvers.eigenvalue.{name} is deprecated; "
            f"use linstabpy.stability.{attr_name}",
            DeprecationWarning,
            stacklevel=2,
        )
        attr = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEPRECATED_EXPORTS))
