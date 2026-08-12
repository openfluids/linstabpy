# SPDX-License-Identifier: GPL-3.0-only
"""LNSE operator construction: coefficients, assembly, energy norms."""

from .assembly import (
    LNSESystem,
    build_lnse_operator,
    build_lnse_operator_petsc,
    build_lnse_system,
    build_lnse_system_result,
)
from .coefficients import (
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
)
from .energy_norm import compute_chu_norm
from .spatial import (
    build_spatial_operator,
    build_spatial_operator_petsc,
    spatial_lhs_with_dirichlet,
)
