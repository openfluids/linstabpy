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
    """Forward and adjoint actions for ``H = C F L_sigma^-1 F B``.

    The spatial filter sits INSIDE the input/output masks, matching the
    MATLAB reference (resolvent.m: ``H = C*FILTER(R(FILTER(B*v)))``).
    The order matters: the filter does not commute with a diagonal mask.
    The previous ``F C R B F`` composition shifted the leading jet gain
    by 0.5% (411.71 vs 413.72).
    """

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
        """Apply resolvent: H v = C FILTER(inv(L_sigma) FILTER(B v))."""
        w = _apply_optional_matrix(self.B_mat, v)
        w = self._filt(w)
        w = self.lu.solve(w)
        w = self._filt(w)
        w = _apply_optional_matrix(self.C_mat, w)
        return w

    def apply_adjoint(self, v: np.ndarray) -> np.ndarray:
        """Apply adjoint resolvent: H^H v = B^H FILTER_ct(inv(L_sigma^H) FILTER_ct(C^H v))."""
        w = _apply_optional_adjoint_matrix(self.C_mat, v)
        w = self._filt_ct(w)
        w = self.lu.solve_transpose(w)
        w = self._filt_ct(w)
        w = _apply_optional_adjoint_matrix(self.B_mat, w)
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
