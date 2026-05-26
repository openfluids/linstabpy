# SPDX-License-Identifier: GPL-3.0-only
"""Operator abstraction layer for linstabpy.

Provides a composable interface for linear operators on top of PETSc.
Adapted from resolvent4py (Padovan et al.).

Imports of public names are deferred via PEP 562 ``__getattr__``: the
submodules (``base``, ``adapters``, ``matrix``, ``petsc_shell``, ``product``,
``projection``, ``harmonic``) eagerly import ``petsc4py``/``slepc4py`` at
module load, so eager re-exports here would defeat the lazy contract
established at ``src/linstabpy/__init__.py``. With this lazy layer in place,
``import linstabpy.linear_operators`` is PETSc-free; the heavy chain only
runs when a specific name is accessed.
"""

_LAZY_EXPORTS = {
    "LinearOperator": "linstabpy.linear_operators.base",
    "MatrixOperator": "linstabpy.linear_operators.matrix",
    "ProductOperator": "linstabpy.linear_operators.product",
    "PetscShellOperator": "linstabpy.linear_operators.petsc_shell",
    "ProjectionOperator": "linstabpy.linear_operators.projection",
    "scipy_to_operator": "linstabpy.linear_operators.adapters",
    "scipy_to_solve_operator": "linstabpy.linear_operators.adapters",
    "build_shifted_operator": "linstabpy.linear_operators.adapters",
    "DiagonalOperator": "linstabpy.linear_operators.adapters",
    "NumpyCallableOperator": "linstabpy.linear_operators.adapters",
    "HarmonicLayout": "linstabpy.linear_operators.harmonic",
    "build_harmonic_weights": "linstabpy.linear_operators.harmonic",
    "assemble_harmonic_operator": "linstabpy.linear_operators.harmonic",
    "assemble_harmonic_operator_petsc": "linstabpy.linear_operators.harmonic",
    "apply_harmonic_bcs": "linstabpy.linear_operators.harmonic",
    "harmonic_resolvent_analysis": "linstabpy.linear_operators.harmonic",
    "projected_harmonic_resolvent_analysis": "linstabpy.linear_operators.harmonic",
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
