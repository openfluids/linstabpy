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

from .petsc_utils import require_complex, scipy_to_petsc, petsc_vec, vec_to_numpy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearSolver:
    """Callable wrapper for forward and adjoint solves."""

    backend: str
    solve: Callable[[np.ndarray], np.ndarray]
    solve_transpose: Callable[[np.ndarray], np.ndarray]


def make_linear_solver(operator: sp.spmatrix) -> LinearSolver:
    """Factor *operator* using PETSc KSP with MUMPS and return a solver.

    Parameters
    ----------
    operator:
        Square sparse matrix to factor (typically a shifted LNSE operator).
    """
    require_complex()

    A = scipy_to_petsc(operator)
    n = operator.shape[0]

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setUp()

    # Log MUMPS ordering method (one-time verification that METIS is active)
    try:
        F = pc.getFactorMatrix()
        icntl7 = F.getMumpsIcntl(7)
        _ordering_names = {0: "AMD", 2: "AMF", 3: "scotch", 4: "PORD",
                           5: "METIS", 6: "QAMD", 7: "auto"}
        ordering = _ordering_names.get(icntl7, f"unknown({icntl7})")
        logger.info("MUMPS factorisation complete (n=%d, nnz=%d, ordering=%s [ICNTL(7)=%d])",
                     n, operator.nnz, ordering, icntl7)
    except (AttributeError, RuntimeError) as e:
        logger.debug("Could not query MUMPS ordering: %s", e)
        logger.info("MUMPS factorisation complete (n=%d, nnz=%d)", n, operator.nnz)

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

    return LinearSolver(backend="MUMPS", solve=solve, solve_transpose=solve_transpose)
