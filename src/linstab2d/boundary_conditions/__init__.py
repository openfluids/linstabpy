"""Boundary condition enforcement."""

from .dirichlet import (
    apply_dirichlet,
    apply_dirichlet_identity_pair,
    apply_dirichlet_petsc,
    apply_dirichlet_spatial,
    apply_dirichlet_spatial_petsc,
    build_dirichlet_restriction_matrices,
)
from .neumann import apply_neumann, apply_neumann_petsc
from .recipe import apply_bc_recipe, apply_bc_recipe_petsc

__all__ = [
    "apply_dirichlet",
    "apply_dirichlet_petsc",
    "apply_dirichlet_spatial",
    "apply_dirichlet_spatial_petsc",
    "build_dirichlet_restriction_matrices",
    "apply_dirichlet_identity_pair",
    "apply_neumann",
    "apply_neumann_petsc",
    "apply_bc_recipe",
    "apply_bc_recipe_petsc",
]
