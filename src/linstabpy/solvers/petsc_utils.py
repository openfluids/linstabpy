# SPDX-License-Identifier: GPL-3.0-only
"""Backward-compatible re-export of PETSc utilities.

The canonical definitions live in ``linstabpy.utils.petsc_core``.
Solver modules can continue to import from here without changes.
"""

from linstabpy.utils.petsc_core import (  # noqa: F401
    _is_petsc_matrix,
    _matrix_nnz,
    _operator_shape,
    numpy_columns_to_petsc_vecs,
    petsc_to_scipy,
    petsc_vec,
    require_complex,
    scipy_to_petsc,
    scipy_to_petsc_distributed,
    scipy_to_petsc_distributed_from_root,
    vec_to_numpy,
)
