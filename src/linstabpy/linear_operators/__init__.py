# SPDX-License-Identifier: GPL-3.0-only
"""Operator abstraction layer for linstabpy.

Provides a composable interface for linear operators on top of PETSc.
Adapted from resolvent4py (Padovan et al.).
"""

from .adapters import (
    DiagonalOperator,
    NumpyCallableOperator,
    build_shifted_operator,
    scipy_to_operator,
    scipy_to_solve_operator,
)
from .base import LinearOperator
from .harmonic import (
    HarmonicLayout,
    apply_harmonic_bcs,
    assemble_harmonic_operator,
    assemble_harmonic_operator_petsc,
    build_harmonic_weights,
    harmonic_resolvent_analysis,
    projected_harmonic_resolvent_analysis,
)
from .matrix import MatrixOperator
from .petsc_shell import PetscShellOperator
from .product import ProductOperator
from .projection import ProjectionOperator

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
