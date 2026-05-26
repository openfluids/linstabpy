# SPDX-License-Identifier: GPL-3.0-only
"""Eigenvalue and resolvent solvers.

Public names are resolved lazily via PEP 562 ``__getattr__`` so that
``import linstabpy.solvers`` does not trigger ``petsc4py`` / ``slepc4py``
at module load. The submodules ``eigenvalue``, ``resolvent``, and
``spatial_eig`` all call ``init_slepc_from_argv()`` at module top, so
eager re-exports here would defeat the lazy contract from
``src/linstabpy/__init__.py``.
"""

_LAZY_EXPORTS = {
    "resolvent_analysis": "linstabpy.solvers.resolvent",
    "adjoint_eigenvalues": "linstabpy.solvers.eigenvalue",
    "temporal_eigenvalues": "linstabpy.solvers.eigenvalue",
    "spatial_eigenvalues": "linstabpy.solvers.spatial_eig",
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
