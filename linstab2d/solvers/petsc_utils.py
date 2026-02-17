"""Backward-compatible re-export of PETSc utilities.

The canonical definitions live in ``linstab2d.utils.petsc_core``.
Solver modules can continue to import from here without changes.
"""

from linstab2d.utils.petsc_core import (  # noqa: F401
    require_complex,
    scipy_to_petsc,
    petsc_vec,
    vec_to_numpy,
)
