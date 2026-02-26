"""Temporal and adjoint eigenvalue solvers using SLEPc EPS."""

from __future__ import annotations

import os
import logging

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
from petsc4py import PETSc
from slepc4py import SLEPc

from .petsc_utils import (
    require_complex,
    scipy_to_petsc,
    scipy_to_petsc_distributed_from_root,
    vec_to_numpy,
)

logger = logging.getLogger(__name__)


def _check_inputs_shape(operator: sp.spmatrix, n_modes: int) -> tuple[sp.csc_matrix, int]:
    """Validate input dimensions for eigensolvers and return CSC copy."""
    if n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("L0 must be a square matrix")
    if n_modes > operator.shape[0]:
        raise ValueError("n_modes cannot exceed matrix size")
    return sp.csc_matrix(operator, dtype=complex), operator.shape[0]


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
    order = np.argsort(np.abs(values - target))
    return values[order], vectors[:, order]


def _eigs_near_target(
    operator: sp.csc_matrix,
    n_modes: int,
    target: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a shift-invert EVP near ``target`` using SLEPc Krylov-Schur."""
    require_complex()
    ndof = operator.shape[0]

    if ndof <= 2 or n_modes >= ndof - 1:
        dense = np.asarray(operator.toarray())
        all_vals, all_vecs = np.linalg.eig(dense)
        order = np.argsort(np.abs(all_vals - target))
        sel = order[:n_modes]
        return all_vals[sel], all_vecs[:, sel]

    st_solver = os.getenv("LINSTAB2D_EIGEN_ST_SOLVER", "mumps").strip().lower()
    use_root_scatter = (
        PETSc.COMM_WORLD.getSize() > 1
        and st_solver in {"iterative", "krylov", "slepc"}
        and os.getenv("LINSTAB2D_PETSC_ROOT_SCATTER", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    if use_root_scatter:
        rank = PETSc.COMM_WORLD.getRank()
        A = scipy_to_petsc_distributed_from_root(
            operator if rank == 0 else None,
            shape=operator.shape,
            root=0,
            comm=PETSc.COMM_WORLD,
        )
    else:
        A = scipy_to_petsc(operator)

    eps = SLEPc.EPS().create()
    eps.setOperators(A)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    # Match MATLAB eigs default used in Hamada workflow:
    # opts.p = 2*nEig + 3
    ncv = min(max(2 * n_modes + 3, 30), ndof - 1)
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTolerances(tol=1e-8, max_it=max(500, 10 * ndof))

    # Shift-invert spectral transformation; linear solver backend is selectable.
    #
    # LINSTAB2D_EIGEN_ST_SOLVER:
    #   mumps (default) -> direct LU factorization (robust, memory-heavy)
    #   iterative       -> Krylov linear solve in ST (scalable, may require tuning)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    pc = ksp.getPC()

    if st_solver in {"mumps", "direct", "lu"}:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc.setType(PETSc.PC.Type.LU)
        if st_solver == "mumps":
            pc.setFactorSolverType("mumps")
    elif st_solver in {"iterative", "krylov", "slepc"}:
        ksp_type = os.getenv("LINSTAB2D_EIGEN_KSP_TYPE", "gmres").strip().lower()
        pc_type = os.getenv("LINSTAB2D_EIGEN_PC_TYPE", "asm").strip().lower()
        ksp_rtol = float(os.getenv("LINSTAB2D_EIGEN_KSP_RTOL", "1e-8"))
        ksp_max_it = int(os.getenv("LINSTAB2D_EIGEN_KSP_MAX_IT", "20000"))
        asm_overlap = int(os.getenv("LINSTAB2D_EIGEN_ASM_OVERLAP", "2"))
        sub_ksp_type = os.getenv("LINSTAB2D_EIGEN_SUB_KSP_TYPE", "preonly").strip().lower()
        sub_pc_type = os.getenv("LINSTAB2D_EIGEN_SUB_PC_TYPE", "lu").strip().lower()
        ksp.setType(ksp_type)
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
        and os.getenv("LINSTAB2D_EIGEN_TRUE_RESIDUAL", "0").strip().lower() in {"1", "true", "yes", "on"}
    ):
        eps.setTrueResidual(True)

    eps.setUp()

    logger.info(
        "  SLEPc EPS: sigma = %.6g%+.6gj, nev=%d ...",
        np.real(target), np.imag(target), n_modes,
    )
    eps.solve()

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
    for i in range(n_extract):
        eig = eps.getEigenpair(i, vr, vi)
        eigenvalues[i] = eig
        eigenvectors[:, i] = vec_to_numpy(vr)

    return eigenvalues, eigenvectors


def temporal_eigenvalues(
    L0: sp.spmatrix,
    n_modes: int,
    target: float | complex,
    W_diag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute temporal eigenvalues ``omega = eig(i*L0)`` near a target.

    The physical PDE is ``dq/dt = -L0 q``, so ``eig(L0) = iω_OS`` and
    ``eig(i*L0) = -ω_OS``.  Unstable modes have ``Im(omega) < 0``.

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

    Returns
    -------
    omega:
        Temporal eigenvalues ``omega = eig(i*L0) = -ω_OS``.
    modes:
        Right eigenvectors corresponding to ``omega``.
    """
    operator, _ = _check_inputs_shape(L0, n_modes)

    temporal_operator = 1j * operator
    omega = complex(target)
    eigenvalues, eigenvectors = _eigs_near_target(temporal_operator, n_modes, omega)
    eigenvalues, eigenvectors = _sort_nearest(eigenvalues, eigenvectors, omega)

    if W_diag is not None:
        eigenvectors = _normalize_in_weighted_inner(eigenvectors, np.asarray(W_diag))

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
    L0: sp.spmatrix,
    n_modes: int,
    target: float | complex,
    W_diag: np.ndarray,
    invW_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute adjoint temporal eigenvalues near a target with W weighting.

    The adjoint operator is formed as ``W^{-1} L0^* W`` and eigendecomposed
    with shift-invert at ``target``.
    """
    operator, _ = _check_inputs_shape(L0, n_modes)
    adjoint_op = _build_adjoint_operator(operator, W_diag, invW_diag)

    omega = complex(target)
    eigenvalues, eigenvectors = _eigs_near_target(adjoint_op, n_modes, omega)
    eigenvalues, eigenvectors = _sort_nearest(eigenvalues, eigenvectors, omega)
    eigenvectors = _normalize_in_weighted_inner(eigenvectors, np.asarray(W_diag))

    return eigenvalues, eigenvectors
