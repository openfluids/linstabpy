# SPDX-License-Identifier: GPL-3.0-only
"""Resolvent operator actions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from linstabpy.utils.petsc_core import _is_petsc_matrix, petsc_vec, vec_to_numpy

if TYPE_CHECKING:
    # Type-only import keeps ``import linstabpy.resolvent`` PETSc-free at
    # runtime (same contract as ``linstabpy.utils.petsc_core``).
    from petsc4py import PETSc


class ResolventOperator:
    """Forward and adjoint actions for ``H = F C L_sigma^-1 B F``."""

    def __init__(
        self,
        lu,
        B_mat: sp.spmatrix | PETSc.Mat | None,
        C_mat: sp.spmatrix | PETSc.Mat | None,
        filter_fn: Callable[[np.ndarray], np.ndarray] | None,
        filter_ct_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        self.lu = lu
        self.B_mat = B_mat
        self.C_mat = C_mat
        self._filt = filter_fn if filter_fn is not None else (lambda v: v)
        self._filt_ct = filter_ct_fn if filter_ct_fn is not None else (lambda v: v)

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Apply resolvent: H v = FILTER(C inv(L_sigma) B FILTER(v))."""
        w = self._filt(v)
        w = _apply_optional_matrix(self.B_mat, w)
        w = self.lu.solve(w)
        w = _apply_optional_matrix(self.C_mat, w)
        w = self._filt(w)
        return w

    def apply_adjoint(self, v: np.ndarray) -> np.ndarray:
        """Apply adjoint resolvent: H^H v = FILTER_ct(B^H inv(L_sigma^H) C^H FILTER_ct(v))."""
        w = self._filt_ct(v)
        w = _apply_optional_adjoint_matrix(self.C_mat, w)
        w = self.lu.solve_transpose(w)
        w = _apply_optional_adjoint_matrix(self.B_mat, w)
        w = self._filt_ct(w)
        return w


def _apply_optional_matrix(
    matrix: sp.spmatrix | PETSc.Mat | None,
    vector: np.ndarray,
) -> np.ndarray:
    """Apply an optional SciPy/PETSc matrix to a NumPy vector."""
    if matrix is None:
        return vector
    if _is_petsc_matrix(matrix):
        x = petsc_vec(vector)
        y = x.duplicate()
        try:
            matrix.mult(x, y)
            return vec_to_numpy(y)
        finally:
            x.destroy()
            y.destroy()
    return matrix @ vector


def _apply_optional_adjoint_matrix(
    matrix: sp.spmatrix | PETSc.Mat | None,
    vector: np.ndarray,
) -> np.ndarray:
    """Apply an optional conjugate-transpose matrix to a NumPy vector."""
    if matrix is None:
        return vector
    if _is_petsc_matrix(matrix):
        x = petsc_vec(vector)
        y = x.duplicate()
        try:
            matrix.multHermitian(x, y)
            return vec_to_numpy(y)
        finally:
            x.destroy()
            y.destroy()
    return matrix.conj().T @ vector
