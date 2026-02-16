"""Minimal SciPy ↔ PETSc conversion utilities."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from petsc4py import PETSc


_petsc_configured = False


def require_complex() -> None:
    """Assert PETSc was built with complex scalar type and set defaults."""
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError(
            "PETSc must be built with --with-scalar-type=complex for stability analysis. "
            f"Current scalar type: {PETSc.ScalarType}"
        )
    global _petsc_configured
    if not _petsc_configured:
        PETSc.Options().setValue("-mat_mumps_icntl_7", "5")  # METIS ordering
        _petsc_configured = True


def scipy_to_petsc(mat: sp.spmatrix) -> PETSc.Mat:
    """Convert a SciPy sparse matrix to a PETSc AIJ Mat."""
    csr = sp.csr_matrix(mat, dtype=complex)
    A = PETSc.Mat().createAIJ(
        size=csr.shape,
        csr=(csr.indptr, csr.indices, csr.data),
    )
    A.assemble()
    return A


def petsc_vec(arr: np.ndarray) -> PETSc.Vec:
    """Convert a 1-D numpy array to a PETSc Vec."""
    v = PETSc.Vec().createWithArray(np.ascontiguousarray(arr, dtype=complex))
    return v


def vec_to_numpy(v: PETSc.Vec) -> np.ndarray:
    """Copy a PETSc Vec to a numpy array.

    Uses read-only access (``array_r``) so the function works correctly
    inside PETSc MatShell callbacks where the input vector is locked.
    """
    return np.array(v.array_r, copy=True)
