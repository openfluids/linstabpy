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

    A = scipy_to_petsc(sp.csc_matrix(operator, dtype=complex))
    n = operator.shape[0]

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setUp()

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
