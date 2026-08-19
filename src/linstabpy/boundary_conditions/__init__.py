# SPDX-License-Identifier: GPL-3.0-only
"""Boundary condition enforcement.

Public names resolve lazily via PEP 562 ``__getattr__``.
The submodules (``dirichlet``, ``neumann``, ``recipe``) import
``petsc4py`` at module load to support the ``*_petsc`` variants.
Eager re-exports here would force PETSc on every consumer of
pure-math entry points like ``apply_dirichlet`` or ``analyze_bc_recipe``.
"""

_LAZY_EXPORTS = {
    "apply_dirichlet": "linstabpy.boundary_conditions.dirichlet",
    "apply_dirichlet_petsc": "linstabpy.boundary_conditions.dirichlet",
    "apply_dirichlet_spatial": "linstabpy.boundary_conditions.dirichlet",
    "apply_dirichlet_spatial_petsc": "linstabpy.boundary_conditions.dirichlet",
    "apply_dirichlet_identity_pair": "linstabpy.boundary_conditions.dirichlet",
    "build_dirichlet_restriction_matrices": "linstabpy.boundary_conditions.dirichlet",
    "dirichlet_free_dofs": "linstabpy.boundary_conditions.dirichlet",
    "apply_neumann": "linstabpy.boundary_conditions.neumann",
    "apply_neumann_petsc": "linstabpy.boundary_conditions.neumann",
    "BCRule": "linstabpy.boundary_conditions.recipe",
    "BCRecipe": "linstabpy.boundary_conditions.recipe",
    "analyze_bc_recipe": "linstabpy.boundary_conditions.recipe",
    "apply_bc_recipe": "linstabpy.boundary_conditions.recipe",
    "apply_bc_recipe_petsc": "linstabpy.boundary_conditions.recipe",
    "apply_bc_recipe_spatial": "linstabpy.boundary_conditions.recipe",
    "apply_bc_recipe_spatial_petsc": "linstabpy.boundary_conditions.recipe",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        attr = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
