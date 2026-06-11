# SPDX-License-Identifier: GPL-3.0-only
"""Shifted linear operators used by stability and resolvent solvers."""

from __future__ import annotations

import scipy.sparse as sp

from linstabpy.utils.petsc_core import _is_petsc_matrix
from linstabpy.utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
from petsc4py import PETSc


def shifted_operator(
    operator: sp.spmatrix | PETSc.Mat,
    omega: complex,
    gamma: float = 0.0,
) -> sp.csc_matrix | PETSc.Mat:
    """Return ``operator - (gamma + 1j*omega)*I`` preserving the backend."""
    if _is_petsc_matrix(operator):
        shifted = operator.copy()
        if gamma == 0.0:
            shifted.shift(-1j * omega)
        else:
            shifted.shift(-(float(gamma) + 1j * omega))
        shifted.assemble()
        return shifted
    shifted = sp.csc_matrix(operator, copy=True)
    if gamma == 0.0:
        shifted.setdiag(shifted.diagonal() - 1j * omega)
    else:
        shifted.setdiag(shifted.diagonal() - (float(gamma) + 1j * omega))
    return shifted
