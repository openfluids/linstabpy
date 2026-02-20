"""Mesh infrastructure: SBP finite differences, structured grids, integration weights."""

from .sbp import (
    sbp_stencils,
    build_sbp_matrix,
    build_sbp_pole_matrices,
    min_points,
)
from .mesh import Mesh
from .filters import MeshFilters, build_mesh_filters
from .fdweights import fdweights
from .grids import segmspace
from .distance import min_distance
from .interpolation import interp_mat_2d
