# SPDX-License-Identifier: GPL-3.0-only
"""Chu's compressible energy norm for perturbation analysis.

Translates ``GetCompEnergyNorm.m`` from the MATLAB LinStab2D codebase.
Supports Cartesian and axisymmetric (axy) models. In the axisymmetric
formulation, integration weights are additionally multiplied by radial
coordinate *r*.
"""

from __future__ import annotations

import numpy as np


def compute_chu_norm(
    W_int: np.ndarray,
    RHO: np.ndarray,
    T: np.ndarray,
    kappa: float,
    Ma: float,
    ngp: int,
    *,
    model: str = "2D",
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute Chu's compressible energy norm weights.

    The energy norm is diagonal with five blocks corresponding to the
    primitive perturbation variables :math:`[\rho', u', v', w', T']`:

    .. math::

        W_{ii} = \frac{1}{2}
        \begin{bmatrix}
            \bar{T} / (\bar{\rho}\,\kappa\,\mathrm{Ma}^2) \\
            \bar{\rho} \\
            \bar{\rho} \\
            \bar{\rho} \\
            \bar{\rho} / [\kappa(\kappa-1)\bar{T}\,\mathrm{Ma}^2]
        \end{bmatrix}
        \odot w_{\mathrm{int}}

    Parameters
    ----------
    W_int : ndarray, shape (ngp,)
        Integration weights (from the mesh, already restricted to active
        grid points).
    RHO : ndarray, shape (ngp,)
        Base flow density at grid points.
    T : ndarray, shape (ngp,)
        Base flow temperature at grid points.
    kappa : float
        Heat capacity ratio (:math:`\gamma`).
    Ma : float
        Mach number.
    ngp : int
        Number of grid points.
    model : str, optional
        ``"2D"`` or ``"axy"``. For ``"axy"``, set ``R`` to radial
        coordinates to include the cylindrical volume factor.
    R : ndarray, optional
        Radial coordinate array used only for ``model="axy"``. Required
        when using axisymmetric weights.

    Returns
    -------
    W_diag : ndarray, shape (5*ngp,)
        Diagonal of the energy norm matrix.
    invW_diag : ndarray, shape (5*ngp,)
        Diagonal of the inverse energy norm matrix.
    """
    if model.lower() == "axy":
        if R is None:
            raise ValueError("R is required when model='axy'")
        R = np.asarray(R, dtype=float)
        W_int = np.asarray(W_int) * np.maximum(R, 1e-14)

    Ma2 = Ma**2

    W_diag = 0.5 * np.concatenate(
        [
            T / (RHO * kappa * Ma2) * W_int,  # rho block
            RHO * W_int,  # u block
            RHO * W_int,  # v block
            RHO * W_int,  # w block
            RHO / (kappa * (kappa - 1.0) * T * Ma2) * W_int,  # T block
        ]
    )

    invW_diag = 1.0 / W_diag

    return W_diag, invW_diag
