# SPDX-License-Identifier: GPL-3.0-only
"""linstabpy: Stability and resolvent analysis of compressible viscous flows."""

from importlib import metadata

try:
    __version__ = metadata.version("linstabpy")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+dev"

# Public API — lazy via PEP 562 __getattr__. Importing linstabpy must NOT
# initialize PETSc/MPI, because pure-math test jobs run without an MPI stack
# installed. Names below are resolved on first attribute access and cached on
# the module for subsequent lookups.
_LAZY_EXPORTS = {
    "build_lnse_system": "linstabpy.operators.assembly",
    "build_lnse_operator": "linstabpy.operators.assembly",
    "compute_chu_norm": "linstabpy.operators.energy_norm",
    "temporal_eigenvalues": ("linstabpy.stability._eigenvalues", "eigenvalues"),
    "adjoint_eigenvalues": "linstabpy.stability._eigenvalues",
    "wavemaker": "linstabpy.stability.sensitivity",
    "resolvent_analysis": "linstabpy.resolvent.analysis",
    "resolvent_map": "linstabpy.solvers.router.resolvent_map",
    "BaseFlow": "linstabpy.baseflow",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        target = _LAZY_EXPORTS[name]
        module_name, attr_name = target if isinstance(target, tuple) else (target, name)
        attr = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'linstabpy' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
