"""Viscosity models for compressible flow.

Translates ``sutherland_air.m`` from the MATLAB LinStab2D codebase.
"""

from __future__ import annotations

import numpy as np

from .baseflow import BaseFlow


def sutherland_air(bf: BaseFlow) -> None:
    """Set viscosity fields on *bf* using Sutherland's law for air.

    The function modifies *bf* **in-place**, populating :pyattr:`MU`,
    :pyattr:`dmudT`, :pyattr:`d2mudT2`, :pyattr:`cv`, :pyattr:`c1`, and
    :pyattr:`c2`.

    Parameters
    ----------
    bf : BaseFlow
        Base flow whose temperature field ``T`` is already set.  The
        non-dimensional temperature is converted to dimensional form via
        ``T_0``.
    """
    kappa = bf.kappa
    T_0 = bf.T_0
    Re = bf.Re
    Pr = bf.Pr
    Ma = bf.Ma

    # -- Derived coefficients -------------------------------------------
    bf.cv = 1.0 / (kappa * (kappa - 1.0) * Ma**2)
    bf.c1 = (kappa - 1.0) * Re * Pr * Ma**2
    bf.c2 = kappa * Ma**2

    # -- Sutherland's law -----------------------------------------------
    S1: float = 110.4  # Sutherland temperature [K]
    T_ref: float = 280.0  # reference temperature [K]
    mu_ref: float = 1.735e-5  # reference dynamic viscosity [Pa.s]

    Tstar = bf.T * T_0  # dimensional temperature

    # Reference viscosity at T_0
    mu_0 = mu_ref * (T_ref + S1) / (T_0 + S1) * (T_0 / T_ref) ** 1.5

    # Viscosity and its first two derivatives w.r.t. dimensional T
    mu = mu_ref * (T_ref + S1) / (Tstar + S1) * (Tstar / T_ref) ** 1.5

    dmudT = (
        -mu_ref * (T_ref + S1) / (Tstar + S1) ** 2 * (Tstar / T_ref) ** 1.5
        + 1.5 * mu_ref * (T_ref + S1) / (Tstar + S1)
        * np.sqrt(Tstar / T_ref) / T_ref
    )

    d2mudT2 = (
        2.0 * mu_ref * (T_ref + S1) / (Tstar + S1) ** 3
        * (Tstar / T_ref) ** 1.5
        - 3.0 * mu_ref * (T_ref + S1) / (Tstar + S1) ** 2
        * np.sqrt(Tstar / T_ref) / T_ref
        + 0.75 * mu_ref * (T_ref + S1) / (Tstar + S1)
        * (Tstar / T_ref) ** (-0.5) / T_ref**2
    )

    # -- Non-dimensionalize ---------------------------------------------
    bf.MU = mu / mu_0
    bf.dmudT = dmudT / mu_0 * T_0
    bf.d2mudT2 = d2mudT2 / mu_0 * T_0**2
