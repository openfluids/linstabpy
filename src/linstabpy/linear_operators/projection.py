# SPDX-License-Identifier: GPL-3.0-only
"""Projection operators for harmonic resolvent phase-direction removal.

Implements oblique projectors of the form

    P = Φ (Ψ* Φ)^{-1} Ψ*

and their complements I - P, where Φ and Ψ are N × r matrices stored as
numpy arrays.  The Gram matrix (Ψ* Φ) is a small r × r dense matrix
inverted at construction time.

The primary use case is rank-1 phase-direction removal in harmonic
resolvent analysis (Padovan et al., 2020, 2022): Φ = dQ/dτ (normalised
time derivative of the base flow) and Ψ = w (the adjoint direction
satisfying T^H w ∝ dQ/dτ).

Adapted from resolvent4py's ProjectionLinearOperator (Padovan et al.).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from .base import LinearOperator


class ProjectionOperator(LinearOperator):
    r"""Oblique projection operator.

    When ``complement=False``:

    .. math::
        L = \Phi \, (\Psi^* \Phi)^{-1} \, \Psi^*

    When ``complement=True``:

    .. math::
        L = I - \Phi \, (\Psi^* \Phi)^{-1} \, \Psi^*

    The operator is idempotent: L² = L.

    Parameters
    ----------
    Phi :
        Column basis for the range of the projector, shape (N, r).
        For a single vector, pass shape (N, 1) or (N,).
    Psi :
        Column basis for the co-range, shape (N, r).
    complement :
        If True, return I - P instead of P.
    """

    def __init__(
        self,
        Phi: np.ndarray,
        Psi: np.ndarray,
        *,
        complement: bool = False,
    ) -> None:
        Phi = np.asarray(Phi, dtype=complex)
        Psi = np.asarray(Psi, dtype=complex)

        # Accept 1-D vectors as rank-1 projectors.
        if Phi.ndim == 1:
            Phi = Phi[:, np.newaxis]
        if Psi.ndim == 1:
            Psi = Psi[:, np.newaxis]

        if Phi.shape[0] != Psi.shape[0]:
            raise ValueError(
                f"Phi and Psi must have the same number of rows: {Phi.shape[0]} != {Psi.shape[0]}"
            )
        if Phi.shape[1] != Psi.shape[1]:
            raise ValueError(
                f"Phi and Psi must have the same number of columns: "
                f"{Phi.shape[1]} != {Psi.shape[1]}"
            )

        N = Phi.shape[0]
        self._Phi = Phi  # (N, r)
        self._Psi = Psi  # (N, r)
        self._complement = complement

        # Gram matrix S = Ψ* Φ  (r × r), precompute its inverse.
        S = Psi.conj().T @ Phi  # (r, r)
        self._Sinv = la.inv(S)  # (r, r)
        # For adjoint: (P)^H = Ψ S^{-H} Φ*
        self._Sinv_H = self._Sinv.conj().T  # (r, r)

        comm = PETSc.COMM_WORLD
        super().__init__(comm, "ProjectionOperator", ((N, N), (N, N)))

    @property
    def rank(self) -> int:
        """Rank of the projector."""
        return self._Phi.shape[1]

    def apply(self, x, y=None):
        """Apply the projection: y = P x  or  y = (I-P) x."""
        if y is None:
            y = self.create_left_vector()
        arr = np.array(x.array_r, copy=True)

        # P x = Φ S^{-1} (Ψ* x)
        Psi_x = self._Psi.conj().T @ arr  # (r,)
        Px = self._Phi @ (self._Sinv @ Psi_x)  # (N,)

        if self._complement:
            y.setArray(arr - Px)
        else:
            y.setArray(Px)
        return y

    def apply_adjoint(self, x, y=None):
        """Apply the adjoint: y = P^H x  or  y = (I-P^H) x.

        P^H = Ψ S^{-H} Φ*  so  (I-P)^H = I - Ψ S^{-H} Φ*.
        """
        if y is None:
            y = self.create_right_vector()
        arr = np.array(x.array_r, copy=True)

        # P^H x = Ψ S^{-H} (Φ* x)
        Phi_x = self._Phi.conj().T @ arr  # (r,)
        PHx = self._Psi @ (self._Sinv_H @ Phi_x)  # (N,)

        if self._complement:
            y.setArray(arr - PHx)
        else:
            y.setArray(PHx)
        return y

    def destroy(self):
        pass  # No PETSc resources to release.
