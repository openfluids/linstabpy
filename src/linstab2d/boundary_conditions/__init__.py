"""Boundary condition enforcement."""

from .dirichlet import (
    apply_dirichlet,
    apply_dirichlet_identity_pair,
    apply_dirichlet_spatial,
    build_dirichlet_restriction_matrices,
)
from .neumann import apply_neumann
from .recipe import apply_bc_recipe

__all__ = [
    "apply_dirichlet",
    "apply_dirichlet_spatial",
    "build_dirichlet_restriction_matrices",
    "apply_dirichlet_identity_pair",
    "apply_neumann",
    "apply_bc_recipe",
]
