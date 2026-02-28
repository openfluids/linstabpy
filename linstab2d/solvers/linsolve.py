"""Direct linear solver via PETSc KSP with MUMPS.

Provides a factorize-once, solve-many interface for the LNSE shift-invert
workflows.  Both forward and conjugate-transpose solves reuse the same
MUMPS factorisation.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from .petsc_utils import require_complex, scipy_to_petsc, petsc_vec, vec_to_numpy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearSolver:
    """Callable wrapper for forward and adjoint solves."""

    backend: str
    solve: Callable[[np.ndarray], np.ndarray]
    solve_transpose: Callable[[np.ndarray], np.ndarray]


def make_linear_solver(
    operator: sp.spmatrix,
    *,
    env_prefix: str = "LINSTAB2D_LINSOLVE",
) -> LinearSolver:
    """Build a reusable PETSc linear solver for ``operator``.

    Parameters
    ----------
    operator:
        Square sparse matrix to factor (typically a shifted LNSE operator).
    env_prefix:
        Environment prefix controlling solver options.

        Supported keys:
        - ``{prefix}_TYPE``: ``mumps`` (default) or ``iterative``
        - ``{prefix}_KSP_TYPE`` (iterative): default ``gmres``
        - ``{prefix}_PC_TYPE`` (iterative): default ``bjacobi``
        - ``{prefix}_KSP_RTOL`` (iterative): default ``1e-10``
        - ``{prefix}_KSP_ATOL`` (iterative): default ``1e-14``
        - ``{prefix}_KSP_MAX_IT`` (iterative): default ``20000``
        - ``{prefix}_ASM_OVERLAP`` (iterative + ASM): default ``2``
        - ``{prefix}_SUB_KSP_TYPE`` (iterative + ASM/BJACOBI): default ``preonly``
        - ``{prefix}_SUB_PC_TYPE`` (iterative + ASM/BJACOBI): default ``lu``
    """
    require_complex()

    A = scipy_to_petsc(operator)
    n = operator.shape[0]
    solver_type = os.getenv(f"{env_prefix}_TYPE", "mumps").strip().lower()

    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
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
                operator.nnz,
                ordering,
                icntl7,
            )
        except (AttributeError, RuntimeError) as e:
            logger.debug("Could not query MUMPS ordering: %s", e)
            logger.info("Linear solver: MUMPS direct (n=%d, nnz=%d)", n, operator.nnz)
        backend = "MUMPS"
    elif solver_type in {"iterative", "krylov", "petsc"}:
        ksp_type = os.getenv(f"{env_prefix}_KSP_TYPE", "fgmres").strip().lower()
        pc_type = os.getenv(f"{env_prefix}_PC_TYPE", "asm").strip().lower()
        ksp_rtol = float(os.getenv(f"{env_prefix}_KSP_RTOL", "1e-10"))
        ksp_atol = float(os.getenv(f"{env_prefix}_KSP_ATOL", "1e-14"))
        ksp_max_it = int(os.getenv(f"{env_prefix}_KSP_MAX_IT", "20000"))
        asm_overlap = int(os.getenv(f"{env_prefix}_ASM_OVERLAP", "2"))
        sub_ksp_type = os.getenv(f"{env_prefix}_SUB_KSP_TYPE", "preonly").strip().lower()
        sub_pc_type = os.getenv(f"{env_prefix}_SUB_PC_TYPE", "lu").strip().lower()

        ksp.setType(ksp_type)
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
                    sub_pc = sub_ksp.getPC()
                    sub_pc.setType(sub_pc_type)
            except PETSc.Error:
                pass

        # Allow runtime override via PETSc options DB.
        ksp.setFromOptions()
        ksp.setUp()
        logger.info(
            "Linear solver: iterative (n=%d, nnz=%d, ksp=%s, pc=%s, rtol=%g, max_it=%d)",
            n,
            operator.nnz,
            ksp_type,
            pc_type,
            ksp_rtol,
            ksp_max_it,
        )
        backend = "PETSc-iterative"
    else:
        raise ValueError(
            f"Unsupported {env_prefix}_TYPE={solver_type!r}; "
            "expected mumps|direct|lu|iterative|krylov|petsc"
        )

    def solve(rhs: np.ndarray) -> np.ndarray:
        b = petsc_vec(rhs)
        x = b.duplicate()
        ksp.solve(b, x)
        return vec_to_numpy(x)

    def solve_transpose(rhs: np.ndarray) -> np.ndarray:
        """Solve A^H x = rhs via conjugation: A^T conj(x) = conj(rhs)."""
        b = petsc_vec(rhs.conj())
        x = b.duplicate()
        ksp.solveTranspose(b, x)
        return vec_to_numpy(x).conj()

    return LinearSolver(backend=backend, solve=solve, solve_transpose=solve_transpose)
