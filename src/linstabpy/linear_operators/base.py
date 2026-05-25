# SPDX-License-Identifier: GPL-3.0-only
"""Abstract base class for linear operators.

Adapted from resolvent4py's LinearOperator design (Padovan et al.).
Simplified for linstabpy: no BV methods, no runtime probes, always complex.
"""

from __future__ import annotations

import abc
import functools
from typing import Optional, Union

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc


def _not_implemented(method):
    """Decorator: raise NotImplementedError with a descriptive message."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        raise NotImplementedError(
            f"'{type(self).__name__}' does not implement '{method.__name__}'"
        )

    return wrapper


class LinearOperator(abc.ABC):
    r"""Abstract base class for linear operators :math:`L`.

    Parameters
    ----------
    comm :
        PETSc communicator.
    name :
        Human-readable label for logging / error messages.
    dimensions :
        ``((local_rows, global_rows), (local_cols, global_cols))``.
    nblocks :
        Number of harmonic blocks (None for non-block operators).
    """

    def __init__(
        self,
        comm: PETSc.Comm,
        name: str,
        dimensions: tuple[tuple[int, int], tuple[int, int]],
        nblocks: Optional[Union[int, None]] = None,
    ) -> None:
        self._comm = comm
        self._name = name
        self._dimensions = dimensions
        self._nblocks = nblocks

    # -- Read-only properties ------------------------------------------------

    @property
    def comm(self) -> PETSc.Comm:
        return self._comm

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self._dimensions

    @property
    def nblocks(self) -> int | None:
        return self._nblocks

    # -- Vector constructors -------------------------------------------------

    def create_right_vector(self) -> PETSc.Vec:
        """Create a vector in the domain of *L*."""
        v = PETSc.Vec().create(comm=self._comm)
        v.setSizes(self._dimensions[1])
        v.setType("standard")
        return v

    def create_left_vector(self) -> PETSc.Vec:
        """Create a vector in the range of *L*."""
        v = PETSc.Vec().create(comm=self._comm)
        v.setSizes(self._dimensions[0])
        v.setType("standard")
        return v

    # -- Required: subclasses must implement ---------------------------------

    @abc.abstractmethod
    def apply(self, x: PETSc.Vec, y: PETSc.Vec | None = None) -> PETSc.Vec:
        r"""Compute :math:`y = L x`."""

    @abc.abstractmethod
    def destroy(self) -> None:
        """Release PETSc resources."""

    # -- Optional: default raises NotImplementedError ------------------------

    @_not_implemented
    def apply_adjoint(
        self, x: PETSc.Vec, y: PETSc.Vec | None = None
    ) -> PETSc.Vec:
        r"""Compute :math:`y = L^H x`."""

    @_not_implemented
    def solve(
        self, x: PETSc.Vec, y: PETSc.Vec | None = None
    ) -> PETSc.Vec:
        r"""Compute :math:`y = L^{-1} x`."""

    @_not_implemented
    def solve_adjoint(
        self, x: PETSc.Vec, y: PETSc.Vec | None = None
    ) -> PETSc.Vec:
        r"""Compute :math:`y = L^{-H} x`."""
