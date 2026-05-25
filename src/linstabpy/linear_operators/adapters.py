# SPDX-License-Identifier: GPL-3.0-only
"""Adapters bridging existing linstabpy sparse matrices to the operator layer.

Provides factory functions that convert SciPy sparse matrices, MUMPS-factorized
solvers, numpy callables, and diagonal weight vectors into LinearOperator
instances.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from ..utils.petsc_core import scipy_to_petsc
from .base import LinearOperator
from .matrix import MatrixOperator

# ---------------------------------------------------------------------------
# SciPy sparse → MatrixOperator
# ---------------------------------------------------------------------------


def scipy_to_operator(
    mat: sp.spmatrix,
    *,
    solver_cfg: dict | None = None,
) -> MatrixOperator:
    """Wrap a SciPy sparse matrix as a MatrixOperator (apply only, no solve).

    Parameters
    ----------
    mat :
        SciPy sparse matrix (will be converted to PETSc AIJ).
    solver_cfg :
        Optional solver config for PETSc matrix conversion mode.
    """
    A = scipy_to_petsc(mat, solver_cfg)
    return MatrixOperator(A)


def scipy_to_solve_operator(
    mat: sp.spmatrix,
    *,
    solver_cfg: dict | None = None,
) -> MatrixOperator:
    """Wrap a SciPy sparse matrix as a MatrixOperator with MUMPS direct solver.

    The returned operator supports all four actions: apply, apply_adjoint,
    solve, and solve_adjoint.

    Parameters
    ----------
    mat :
        Square invertible SciPy sparse matrix.
    solver_cfg :
        Optional solver config for PETSc matrix conversion and MUMPS options.
    """
    A = scipy_to_petsc(mat, solver_cfg)
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setUp()
    return MatrixOperator(A, ksp)


def build_shifted_operator(
    L0: sp.spmatrix,
    omega: float | complex,
    *,
    solver_cfg: dict | None = None,
) -> MatrixOperator:
    """Build the shifted LNSE operator L_sigma = L0 - i*omega*I and factorize.

    Returns a MatrixOperator with MUMPS KSP, supporting apply, solve, and
    their adjoint counterparts.

    Parameters
    ----------
    L0 :
        LNSE operator (sparse, square, BCs applied).
    omega :
        Angular frequency for the resolvent shift.
    solver_cfg :
        Optional solver config.
    """
    L_sigma = sp.csc_matrix(L0, copy=True)
    L_sigma.setdiag(L_sigma.diagonal() - 1j * omega)
    return scipy_to_solve_operator(L_sigma, solver_cfg=solver_cfg)


# ---------------------------------------------------------------------------
# Diagonal scaling operator
# ---------------------------------------------------------------------------


class DiagonalOperator(LinearOperator):
    """Linear operator representing diagonal scaling: y = diag(d) @ x.

    Self-adjoint when *d* is real, Hermitian-adjoint uses conj(d).

    Parameters
    ----------
    d :
        Diagonal entries (1-D numpy array, real or complex).
    """

    def __init__(self, d: np.ndarray) -> None:
        self._d = np.asarray(d, dtype=complex).ravel()
        n = self._d.size
        comm = PETSc.COMM_WORLD
        super().__init__(comm, "DiagonalOperator", ((n, n), (n, n)))

    def apply(self, x, y=None):
        if y is None:
            y = self.create_left_vector()
        arr = np.array(x.array_r, copy=True)
        y.setArray(self._d * arr)
        return y

    def apply_adjoint(self, x, y=None):
        if y is None:
            y = self.create_right_vector()
        arr = np.array(x.array_r, copy=True)
        y.setArray(self._d.conj() * arr)
        return y

    def solve(self, x, y=None):
        if y is None:
            y = self.create_left_vector()
        arr = np.array(x.array_r, copy=True)
        y.setArray(arr / self._d)
        return y

    def solve_adjoint(self, x, y=None):
        if y is None:
            y = self.create_right_vector()
        arr = np.array(x.array_r, copy=True)
        y.setArray(arr / self._d.conj())
        return y

    def destroy(self):
        pass  # No PETSc resources to release.


# ---------------------------------------------------------------------------
# Numpy callable operator
# ---------------------------------------------------------------------------


class NumpyCallableOperator(LinearOperator):
    """Wrap numpy-level callables as a LinearOperator.

    Useful for spatial filters and other transformations that operate on
    numpy arrays.

    Parameters
    ----------
    fn :
        Forward action: takes 1-D complex numpy array, returns same.
    fn_adjoint :
        Adjoint action. If None, ``apply_adjoint`` raises NotImplementedError.
    n :
        Vector dimension.
    """

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        n: int,
        *,
        fn_adjoint: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._fn = fn
        self._fn_adj = fn_adjoint
        comm = PETSc.COMM_WORLD
        super().__init__(comm, "NumpyCallableOperator", ((n, n), (n, n)))

    def apply(self, x, y=None):
        if y is None:
            y = self.create_left_vector()
        arr = np.array(x.array_r, copy=True)
        y.setArray(self._fn(arr))
        return y

    def apply_adjoint(self, x, y=None):
        if self._fn_adj is None:
            raise NotImplementedError(f"'{self.name}' has no adjoint callable")
        if y is None:
            y = self.create_right_vector()
        arr = np.array(x.array_r, copy=True)
        y.setArray(self._fn_adj(arr))
        return y

    def destroy(self):
        pass  # No PETSc resources to release.
