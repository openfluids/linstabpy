# SPDX-License-Identifier: GPL-3.0-only
"""Stability and resolvent analysis of compressible viscous flows.

The linearized Navier-Stokes operator is assembled on a two-dimensional
structured mesh. Summation-by-parts finite differences cover orders 2 to 8.
Temporal and adjoint eigenvalue problems, structural sensitivity, and
resolvent gain use PETSc, SLEPc, and a sparse direct factor (MUMPS by default).

A plain ``import linstabpy`` does not start PETSc or MPI. Public names load
on first use. The loader caches each name on this module.

Public names
------------
build_lnse_system, build_lnse_operator
    Sparse LNSE operator. State ``q = [rho, u, v, w, T]``.
compute_chu_norm
    Chu compressible energy inner product.
temporal_eigenvalues, adjoint_eigenvalues
    SLEPc temporal eigenvalue problems.
wavemaker
    Structural sensitivity from direct and adjoint modes.
resolvent_analysis, resolvent_map
    Leading gain at one frequency, or a frequency walk.
BaseFlow
    Mean flow, sponge damping, and viscosity.
"""

from __future__ import annotations

import importlib
import re
from importlib import metadata
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def _package_version() -> str:
    """Return the version from pyproject.toml, then installed metadata."""
    if _PYPROJECT.is_file():
        match = _VERSION_RE.search(_PYPROJECT.read_text(encoding="utf-8"))
        if match is not None:
            return match.group(1)
    try:
        return metadata.version("linstabpy")
    except metadata.PackageNotFoundError:
        return "0.0.0+dev"


__version__ = _package_version()

# Each public name maps to (module, attribute). ``temporal_eigenvalues`` is
# the public alias of ``stability._eigenvalues.eigenvalues``.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "build_lnse_system": ("linstabpy.operators.assembly", "build_lnse_system"),
    "build_lnse_operator": ("linstabpy.operators.assembly", "build_lnse_operator"),
    "compute_chu_norm": ("linstabpy.operators.energy_norm", "compute_chu_norm"),
    "temporal_eigenvalues": ("linstabpy.stability._eigenvalues", "eigenvalues"),
    "adjoint_eigenvalues": ("linstabpy.stability._eigenvalues", "adjoint_eigenvalues"),
    "wavemaker": ("linstabpy.stability.sensitivity", "wavemaker"),
    "resolvent_analysis": ("linstabpy.resolvent.analysis", "resolvent_analysis"),
    "resolvent_map": ("linstabpy.solvers.router.resolvent_map", "resolvent_map"),
    "BaseFlow": ("linstabpy.baseflow", "BaseFlow"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    """Load a public name on first use and cache it on this module."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'linstabpy' has no attribute {name!r}")
    module_name, attr_name = target
    attr = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    """List bound names and public names that load on first use."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
