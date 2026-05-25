# SPDX-License-Identifier: GPL-3.0-only
"""Operator abstraction layer for linstabpy.

Provides a composable interface for linear operators on top of PETSc.
Adapted from resolvent4py (Padovan et al.).
"""

from .base import LinearOperator
from .matrix import MatrixOperator
from .product import ProductOperator
from .petsc_shell import PetscShellOperator
from .projection import ProjectionOperator
from .adapters import (
    scipy_to_operator,
    scipy_to_solve_operator,
    build_shifted_operator,
    DiagonalOperator,
    NumpyCallableOperator,
)
from .harmonic import (
    HarmonicLayout,
    build_harmonic_weights,
    assemble_harmonic_operator,
    assemble_harmonic_operator_petsc,
    apply_harmonic_bcs,
    harmonic_resolvent_analysis,
    projected_harmonic_resolvent_analysis,
)

__all__ = [
    "LinearOperator",
    "MatrixOperator",
    "ProductOperator",
    "PetscShellOperator",
    "ProjectionOperator",
    "scipy_to_operator",
    "scipy_to_solve_operator",
    "build_shifted_operator",
    "DiagonalOperator",
    "NumpyCallableOperator",
    "HarmonicLayout",
    "build_harmonic_weights",
    "assemble_harmonic_operator",
    "assemble_harmonic_operator_petsc",
    "apply_harmonic_bcs",
    "harmonic_resolvent_analysis",
    "projected_harmonic_resolvent_analysis",
]
