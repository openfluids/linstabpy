# SPDX-License-Identifier: GPL-3.0-only
"""Mesh infrastructure: SBP finite differences, structured grids, integration weights."""

from .fdweights import fdweights
from .sbp import (
    build_sbp_dissipation_matrices,
    build_sbp_matrix,
    build_sbp_pole_matrices,
    min_points,
    sbp_stencils,
)

__all__ = [
    "Mesh",
    "MeshGeometry",
    "MeshOperators",
    "MeshRuntimeState",
    "MeshFilters",
    "build_mesh_filters",
    "sbp_stencils",
    "build_sbp_matrix",
    "build_sbp_pole_matrices",
    "build_sbp_dissipation_matrices",
    "min_points",
    "fdweights",
]


def __getattr__(name: str):
    """Lazily import PETSc-dependent mesh objects."""
    if name in {"Mesh", "MeshGeometry", "MeshOperators", "MeshRuntimeState"}:
        from .mesh import Mesh, MeshGeometry, MeshOperators, MeshRuntimeState

        return {
            "Mesh": Mesh,
            "MeshGeometry": MeshGeometry,
            "MeshOperators": MeshOperators,
            "MeshRuntimeState": MeshRuntimeState,
        }[name]
    if name in {"MeshFilters", "build_mesh_filters"}:
        from .filters import MeshFilters, build_mesh_filters

        return {"MeshFilters": MeshFilters, "build_mesh_filters": build_mesh_filters}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
