"""Direct linear solver via PETSc KSP with MUMPS.

Provides a factorize-once, solve-many interface for the LNSE shift-invert
workflows.  Both forward and conjugate-transpose solves reuse the same
MUMPS factorisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from .petsc_utils import _is_petsc_matrix, _operator_shape, require_complex, scipy_to_petsc, vec_to_numpy
from ..utils._solver_cfg import _scfg, _scfg_float, _scfg_int

logger = logging.getLogger(__name__)


def _matrix_nnz(operator: sp.spmatrix | PETSc.Mat) -> int:
    """Return the stored nonzero count for SciPy or PETSc matrices."""
    if _is_petsc_matrix(operator):
        return int(operator.getInfo().get("nz_used", 0.0))
    return int(operator.nnz)


@dataclass
class LinearSolver:
    """Callable wrapper for forward and adjoint solves."""

    backend: str
    solve: Callable[[np.ndarray], np.ndarray]
    solve_transpose: Callable[[np.ndarray], np.ndarray]
    solve_vec: Callable[[PETSc.Vec], PETSc.Vec]
    solve_transpose_vec: Callable[[PETSc.Vec], PETSc.Vec]
    _ksp: PETSc.KSP | None = None
    _mat: PETSc.Mat | None = None

    def destroy(self) -> None:
        """Release PETSc KSP and matrix resources."""
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        if self._mat is not None:
            self._mat.destroy()
            self._mat = None


def make_linear_solver(
    operator: sp.spmatrix | PETSc.Mat,
    *,
    env_prefix: str = "LINSTAB2D_LINSOLVE",
    solver_cfg: dict | None = None,
) -> LinearSolver:
    """Build a reusable PETSc linear solver for ``operator``.

    Parameters
    ----------
    operator:
        Square sparse matrix to factor (typically a shifted LNSE operator).
    env_prefix:
        Environment prefix controlling solver options.
    solver_cfg:
        Optional solver config dict (from JSONC ``"solver"`` section).
        When provided, the appropriate subsection is selected based on
        ``env_prefix``: ``"resolvent"`` subsection for
        ``LINSTAB2D_RESOLVENT``, ``"linsolve"`` otherwise. Keys in the
        subsection override the corresponding env vars.
    """
    require_complex(solver_cfg)

    nrows, ncols = _operator_shape(operator)
    if nrows != ncols:
        raise ValueError("operator must be a square matrix")
    n = nrows
    nnz = _matrix_nnz(operator)

    if _is_petsc_matrix(operator):
        A = operator.copy()
    else:
        A = scipy_to_petsc(operator, solver_cfg)

    # Resolve the config subsection for this linear-solver instance.
    _sub: dict | None = None
    if solver_cfg is not None:
        if env_prefix == "LINSTAB2D_RESOLVENT":
            _sub = solver_cfg.get("resolvent", solver_cfg.get("linsolve", {}))
        else:
            _sub = solver_cfg.get("linsolve", {})

    solver_type = _scfg(_sub, "type", f"{env_prefix}_TYPE", "mumps")

    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    pc = ksp.getPC()

    if solver_type in {"mumps", "direct", "lu"}:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        ksp.setUp()

        # Log MUMPS ordering method (one-time verification that METIS is active)
        try:
            F = pc.getFactorMatrix()
            icntl7 = F.getMumpsIcntl(7)
            _ordering_names = {
                0: "AMD",
                2: "AMF",
                3: "scotch",
                4: "PORD",
                5: "METIS",
                6: "QAMD",
                7: "auto",
            }
            ordering = _ordering_names.get(icntl7, f"unknown({icntl7})")
            logger.info(
                "Linear solver: MUMPS direct (n=%d, nnz=%d, ordering=%s [ICNTL(7)=%d])",
                n,
                nnz,
                ordering,
                icntl7,
            )
        except (AttributeError, RuntimeError) as e:
            logger.debug("Could not query MUMPS ordering: %s", e)
            logger.info("Linear solver: MUMPS direct (n=%d, nnz=%d)", n, nnz)
        backend = "MUMPS"
    elif solver_type in {"iterative", "krylov", "petsc"}:
        ksp_type = _scfg(_sub, "ksp_type", f"{env_prefix}_KSP_TYPE", "fgmres")
        pc_type = _scfg(_sub, "pc_type", f"{env_prefix}_PC_TYPE", "asm")
        pc_side = _scfg(_sub, "ksp_pc_side", f"{env_prefix}_KSP_PC_SIDE", "right")
        ksp_rtol = _scfg_float(_sub, "ksp_rtol", f"{env_prefix}_KSP_RTOL", 1e-10)
        ksp_atol = _scfg_float(_sub, "ksp_atol", f"{env_prefix}_KSP_ATOL", 1e-14)
        ksp_max_it = _scfg_int(_sub, "ksp_max_it", f"{env_prefix}_KSP_MAX_IT", 20000)
        asm_overlap = _scfg_int(_sub, "asm_overlap", f"{env_prefix}_ASM_OVERLAP", 2)
        sub_ksp_type = _scfg(_sub, "sub_ksp_type", f"{env_prefix}_SUB_KSP_TYPE", "preonly")
        sub_pc_type = _scfg(_sub, "sub_pc_type", f"{env_prefix}_SUB_PC_TYPE", "lu")

        ksp.setType(ksp_type)
        if pc_side == "left":
            ksp.setPCSide(PETSc.PC.Side.LEFT)
        elif pc_side == "right":
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
        elif pc_side in {"symmetric", "symm"}:
            ksp.setPCSide(PETSc.PC.Side.SYMMETRIC)
        else:
            raise ValueError(
                f"Invalid {env_prefix}_KSP_PC_SIDE '{pc_side}'; expected left|right|symmetric"
            )
        pc.setType(pc_type)
        ksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)
        ksp.setErrorIfNotConverged(True)

        if pc_type == "asm":
            pc.setASMOverlap(asm_overlap)

        # Configure local block solvers for ASM/BJACOBI (parallel-safe defaults).
        if pc_type in {"asm", "bjacobi"}:
            try:
                sub_ksps = pc.getASMSubKSP() if pc_type == "asm" else pc.getBJacobiSubKSP()
                for sub_ksp in sub_ksps:
                    sub_ksp.setType(sub_ksp_type)
                    # Hard-stop on any local block Krylov divergence so iterative
                    # runs cannot silently continue with invalid sub-solves.
                    sub_ksp.setErrorIfNotConverged(True)
                    sub_pc = sub_ksp.getPC()
                    sub_pc.setType(sub_pc_type)
            except PETSc.Error:
                pass

        # Allow runtime override via PETSc options DB.
        ksp.setFromOptions()
        ksp.setUp()
        logger.info(
            "Linear solver: iterative (n=%d, nnz=%d, ksp=%s, pc=%s, side=%s, rtol=%g, max_it=%d)",
            n,
            nnz,
            ksp_type,
            pc_type,
            pc_side,
            ksp_rtol,
            ksp_max_it,
        )
        backend = "PETSc-iterative"
    else:
        raise ValueError(
            f"Unsupported {env_prefix}_TYPE={solver_type!r}; "
            "expected mumps|direct|lu|iterative|krylov|petsc"
        )

    def _check_vec_size(rhs: PETSc.Vec) -> None:
        if int(rhs.getSize()) != n:
            raise ValueError(f"rhs must have size {n}, got {rhs.getSize()}")

    def _rhs_vec_from_numpy(rhs: np.ndarray) -> PETSc.Vec:
        rhs_arr = np.asarray(rhs, dtype=complex)
        if rhs_arr.ndim != 1 or rhs_arr.size != n:
            raise ValueError(f"rhs must be a 1-D array of length {n}, got shape {rhs_arr.shape}")

        b = A.createVecLeft()
        r0, r1 = b.getOwnershipRange()
        if r1 > r0:
            rows = np.arange(r0, r1, dtype=PETSc.IntType)
            b.setValues(rows, rhs_arr[r0:r1], addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        return b

    def solve_vec(rhs: PETSc.Vec) -> PETSc.Vec:
        """Solve ``A x = rhs`` while preserving PETSc vector distribution."""
        _check_vec_size(rhs)
        x = rhs.duplicate()
        ksp.solve(rhs, x)
        return x

    def solve_transpose_vec(rhs: PETSc.Vec) -> PETSc.Vec:
        """Solve ``A^H x = rhs`` while preserving PETSc vector distribution."""
        _check_vec_size(rhs)
        b = rhs.copy()
        x = rhs.duplicate()
        try:
            b.conjugate()
            ksp.solveTranspose(b, x)
            x.conjugate()
            return x
        finally:
            b.destroy()

    def solve(rhs: np.ndarray) -> np.ndarray:
        b = _rhs_vec_from_numpy(rhs)
        try:
            x = solve_vec(b)
            try:
                return vec_to_numpy(x)
            finally:
                x.destroy()
        finally:
            b.destroy()

    def solve_transpose(rhs: np.ndarray) -> np.ndarray:
        b = _rhs_vec_from_numpy(rhs)
        try:
            x = solve_transpose_vec(b)
            try:
                return vec_to_numpy(x)
            finally:
                x.destroy()
        finally:
            b.destroy()

    return LinearSolver(
        backend=backend,
        solve=solve,
        solve_transpose=solve_transpose,
        solve_vec=solve_vec,
        solve_transpose_vec=solve_transpose_vec,
        _ksp=ksp,
        _mat=A,
    )
