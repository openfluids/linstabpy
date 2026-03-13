"""Temporal and adjoint eigenvalue solvers using SLEPc EPS."""

from __future__ import annotations

import logging
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
    petsc_to_scipy,
    require_complex,
    scipy_to_petsc,
    scipy_to_petsc_distributed_from_root,
    vec_to_numpy,
)
from ..utils._solver_cfg import _scfg, _scfg_bool, _scfg_float, _scfg_int

logger = logging.getLogger(__name__)


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


def _sort_nearest(values: np.ndarray, vectors: np.ndarray, target: complex) -> tuple[np.ndarray, np.ndarray]:
    """Sort eigenpairs by distance to ``target`` with deterministic tie-breaking.

    Eigenvalues within ``tie_tol`` of each other are considered equidistant
    and broken by real then imaginary part for reproducibility.
    """
    distances = np.abs(values - target)
    # Bucket tolerance for tie-breaking: eigenvalues closer than this are
    # considered equidistant from the target and sorted by value instead.
    tie_tol = 1e-10
    distance_buckets = np.rint(distances / tie_tol)
    order = np.lexsort((np.imag(values), np.real(values), distance_buckets))
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

    raise RuntimeError("SLEPc EPS.setUp failed: " + ", ".join(str(item) for item in details)) from exc


def _raise_eps_solve_failure(
    *,
    exc: Exception,
    eps: SLEPc.EPS,
    ksp: PETSc.KSP,
    target: complex,
) -> NoReturn:
    """Raise a richer error when SLEPc solve fails after setup."""
    details = [
        f"type={type(exc).__name__}",
        f"message={exc}",
        f"target={target}",
        f"eps_reason={_safe_petsc_info(eps.getConvergedReason)}",
        f"ksp_reason={_safe_petsc_info(ksp.getConvergedReason)}",
    ]
    raise RuntimeError("SLEPc EPS.solve failed: " + ", ".join(str(item) for item in details)) from exc


def _eigs_near_target(
    operator: sp.csc_matrix | PETSc.Mat,
    n_modes: int,
    target: complex,
    *,
    ncv: int | None = None,
    solver_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
        return all_vals[sel], all_vecs[:, sel]

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
    ncv = _resolve_ncv(
        ndof=ndof,
        n_modes=n_modes,
        default_ncv=default_ncv,
        requested_ncv=ncv,
    )
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps_tol = _scfg_float(_ecfg, "eps_tol", "LINSTAB2D_EIGEN_EPS_TOL", 1e-10)
    eps_max_it = _scfg_int(_ecfg, "eps_max_it", "LINSTAB2D_EIGEN_EPS_MAX_IT", max(500, 10 * ndof))
    eps.setTolerances(tol=eps_tol, max_it=eps_max_it)

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
        ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)
        ksp.setErrorIfNotConverged(True)
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
    ksp.setFromOptions()
    st.setFromOptions()
    eps.setFromOptions()
    if (
        st_solver in {"iterative", "krylov", "slepc"}
        and hasattr(eps, "setTrueResidual")
        and _scfg_bool(_ecfg, "true_residual", "LINSTAB2D_EIGEN_TRUE_RESIDUAL", False)
    ):
        eps.setTrueResidual(True)

    try:
        eps.setUp()
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
        "  SLEPc EPS: sigma = %.6g%+.6gj, nev=%d ...",
        np.real(target), np.imag(target), n_modes,
    )
    try:
        eps.solve()
    except Exception as exc:
        _raise_eps_solve_failure(exc=exc, eps=eps, ksp=ksp, target=target)

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError(
            f"SLEPc EPS did not converge for any eigenpair (target={target})"
        )
    if nconv < n_modes:
        logger.warning(
            "SLEPc EPS converged %d/%d eigenpairs; returning partial result",
            nconv, n_modes,
        )

    n_extract = min(nconv, n_modes)
    eigenvalues = np.empty(n_extract, dtype=complex)
    eigenvectors = np.empty((ndof, n_extract), dtype=complex)

    vr = A.createVecRight()
    vi = A.createVecRight()
    try:
        for i in range(n_extract):
            eig = eps.getEigenpair(i, vr, vi)
            eigenvalues[i] = eig
            eigenvectors[:, i] = vec_to_numpy(vr)
    finally:
        vr.destroy()
        vi.destroy()
        eps.destroy()
        A.destroy()

    return eigenvalues, eigenvectors


def temporal_eigenvalues(
    L0: sp.spmatrix | PETSc.Mat,
    n_modes: int,
    target: float | complex,
    W_diag: np.ndarray | None = None,
    ncv: int | None = None,
    solver_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    eigenvalues, eigenvectors = _eigs_near_target(
        temporal_operator,
        n_modes,
        omega,
        ncv=ncv,
        solver_cfg=solver_cfg,
    )
    eigenvalues, eigenvectors = _sort_nearest(eigenvalues, eigenvectors, omega)

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
