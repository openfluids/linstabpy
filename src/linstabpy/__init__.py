# SPDX-License-Identifier: GPL-3.0-only
"""linstabpy: Stability and resolvent analysis of compressible viscous flows."""

__version__ = "0.1.0"

# Public API re-exports
from linstabpy.operators.assembly import build_lnse_system, build_lnse_operator
from linstabpy.operators.energy_norm import compute_chu_norm
from linstabpy.solvers.eigenvalue import temporal_eigenvalues, adjoint_eigenvalues
from linstabpy.solvers.resolvent import resolvent_analysis
from linstabpy.baseflow import (
    BaseFlow,
    load_blasius,
    load_cavity,
    load_jet,
    load_paraboloid,
    load_streaky_bl,
)
