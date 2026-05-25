# SPDX-License-Identifier: GPL-3.0-only
"""Mesh infrastructure: SBP finite differences, structured grids, integration weights."""

from .distance import min_distance
from .fdweights import fdweights
from .grids import segmspace
from .interpolation import interp_mat_2d
from .sbp import (
    build_sbp_dissipation_matrices,
    build_sbp_matrix,
    build_sbp_pole_matrices,
    min_points,
    sbp_stencils,
)

__all__ = [
    "Mesh",
    "MeshFilters",
    "build_mesh_filters",
    "sbp_stencils",
    "build_sbp_matrix",
    "build_sbp_pole_matrices",
    "build_sbp_dissipation_matrices",
    "min_points",
    "fdweights",
    "segmspace",
    "min_distance",
    "interp_mat_2d",
]


def __getattr__(name: str):
    """Lazily import PETSc-dependent mesh objects."""
    if name == "Mesh":
        from .mesh import Mesh

        return Mesh
    if name in {"MeshFilters", "build_mesh_filters"}:
        from .filters import MeshFilters, build_mesh_filters

        return {"MeshFilters": MeshFilters, "build_mesh_filters": build_mesh_filters}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
