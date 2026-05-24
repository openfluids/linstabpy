"""Base flow data container for linear stability analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaseFlow:
    """Base flow fields and parameters for linear stability analysis.

    All flow field arrays have shape ``(ny, nx)`` matching the computational
    mesh.  Viscosity fields and derived coefficients are populated later by
    :func:`~linstab2d.baseflow.viscosity.sutherland_air`.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Ma : float
        Mach number.
    Pr : float
        Prandtl number.
    kappa : float
        Heat capacity ratio (gamma).
    T_0 : float
        Reference temperature [K].
    RHO, U, V, W, T : np.ndarray
        Density, streamwise velocity, wall-normal velocity, spanwise
        velocity, and temperature fields.
    """

    # Physical parameters
    Re: float
    Ma: float
    Pr: float
    kappa: float
    T_0: float

    # Flow fields -- shape (ny, nx) matching mesh
    RHO: np.ndarray
    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    T: np.ndarray

    # Viscosity fields (populated by sutherland_air)
    MU: np.ndarray | None = None
    dmudT: np.ndarray | None = None
    d2mudT2: np.ndarray | None = None

    # Derived coefficients (populated by sutherland_air)
    cv: float | None = None
    c1: float | None = None
    c2: float | None = None
