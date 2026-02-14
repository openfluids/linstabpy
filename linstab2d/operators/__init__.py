"""LNSE operator construction: coefficients, assembly, energy norms."""

from .coefficients import (
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
)
from .assembly import build_lnse_operator
from .spatial import build_spatial_operator
from .energy_norm import compute_chu_norm
