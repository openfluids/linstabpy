"""Boundary condition enforcement."""

from .dirichlet import (
    apply_dirichlet,
    apply_dirichlet_identity_pair,
    apply_dirichlet_spatial,
    build_dirichlet_restriction_matrices,
)
from .neumann import apply_neumann

__all__ = [
    "apply_dirichlet",
    "apply_dirichlet_spatial",
    "build_dirichlet_restriction_matrices",
    "apply_dirichlet_identity_pair",
    "apply_neumann",
]
