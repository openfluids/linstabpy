# SPDX-License-Identifier: GPL-3.0-only
"""Matrix-backed linear operator.

Wraps a PETSc Mat and an optional KSP for solve operations.
Adapted from resolvent4py's MatrixLinearOperator (Padovan et al.).
"""

from __future__ import annotations

from typing import Optional, Union

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from .base import LinearOperator


class MatrixOperator(LinearOperator):
    r"""Linear operator backed by a PETSc matrix: :math:`L = A`.

    Parameters
    ----------
    A :
        PETSc matrix (any type: AIJ, MatShell, etc.).
    ksp :
        Optional KSP solver to enable ``solve()`` and ``solve_adjoint()``.
    nblocks :
        Number of harmonic blocks (None for non-block operators).
    """

    def __init__(
        self,
        A: PETSc.Mat,
        ksp: Optional[Union[PETSc.KSP, None]] = None,
        nblocks: Optional[Union[int, None]] = None,
    ) -> None:
        self.A = A
        self.ksp = ksp
        super().__init__(
            A.getComm(), "MatrixOperator", A.getSizes(), nblocks
        )

    # -- Forward application -------------------------------------------------

    def apply(self, x, y=None):
        if y is None:
            y = self.create_left_vector()
        self.A.mult(x, y)
        return y

    def apply_adjoint(self, x, y=None):
        if y is None:
            y = self.create_right_vector()
        self.A.multHermitian(x, y)
        return y

    # -- Solve ---------------------------------------------------------------

    def solve(self, x, y=None):
        if self.ksp is None:
            raise RuntimeError(
                f"{self.name}.solve() requires a KSP; none was provided"
            )
        if y is None:
            y = self.create_left_vector()
        self.ksp.solve(x, y)
        return y

    def solve_adjoint(self, x, y=None):
        r"""Compute :math:`y = A^{-H} x` via conjugation trick.

        Uses ``A^{-H} x = conj(A^{-T} conj(x))``.
        """
        if self.ksp is None:
            raise RuntimeError(
                f"{self.name}.solve_adjoint() requires a KSP; none was provided"
            )
        if y is None:
            y = self.create_right_vector()
        x.conjugate()
        self.ksp.solveTranspose(x, y)
        x.conjugate()  # restore input
        y.conjugate()
        return y

    # -- Cleanup -------------------------------------------------------------

    def destroy(self):
        self.A.destroy()
        if self.ksp is not None:
            self.ksp.destroy()
