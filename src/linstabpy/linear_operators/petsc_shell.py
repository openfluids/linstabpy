# SPDX-License-Identifier: GPL-3.0-only
"""Expose a LinearOperator as a PETSc Mat (shell matrix).

Useful for passing operator compositions to SLEPc EPS/SVD solvers.
Adapted from resolvent4py's PetscPythonLinearOperator (Padovan et al.).
"""

from __future__ import annotations

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from .base import LinearOperator


class PetscShellOperator:
    """PETSc MatShell context wrapping a :class:`LinearOperator`.

    This is *not* a subclass of :class:`LinearOperator` — it implements
    the PETSc ``Mat().createPython()`` callback protocol.

    Usage
    -----
    >>> shell = PetscShellOperator.as_mat(linop)
    >>> eps.setOperators(shell)
    """

    def __init__(self, L: LinearOperator) -> None:
        self.L = L

    def mult(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """PETSc callback: y = L x."""
        self.L.apply(x, y)

    def multHermitian(
        self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec
    ) -> None:
        """PETSc callback: y = L^H x."""
        self.L.apply_adjoint(x, y)

    def multTranspose(
        self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec
    ) -> None:
        r"""PETSc callback: y = L^T x.

        SLEPc SVD with ``setImplicitTranspose(True)`` calls ``multTranspose``
        expecting :math:`A^T`, not :math:`A^H`.  We use the identity
        :math:`A^T x = \overline{A^H \bar{x}}`.

        Note: ``x`` is read-locked inside MatShell callbacks, so we must
        copy before conjugating.
        """
        x_copy = x.duplicate()
        x.copy(x_copy)
        x_copy.conjugate()
        self.L.apply_adjoint(x_copy, y)
        x_copy.destroy()
        y.conjugate()

    @classmethod
    def as_mat(cls, L: LinearOperator) -> PETSc.Mat:
        """Create a PETSc shell matrix wrapping *L*.

        Returns
        -------
        PETSc.Mat
            Shell matrix whose ``mult`` / ``multHermitian`` /
            ``multTranspose`` delegate to *L*.
        """
        mat = PETSc.Mat().create(L.comm)
        mat.setSizes(L.dimensions)
        mat.setType("python")
        mat.setPythonContext(cls(L))
        mat.setUp()
        return mat
