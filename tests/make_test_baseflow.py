# SPDX-License-Identifier: GPL-3.0-only
"""Tiny manufactured base flow for operator tests."""

from __future__ import annotations

import numpy as np

from linstabpy.baseflow import BaseFlow, sutherland_air


def make_test_baseflow(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    Re: float = 1.0e4,
    Ma: float = 0.3,
    Pr: float = 0.7,
    kappa: float = 1.4,
    T_0: float = 288.15,
) -> BaseFlow:
    """Return a shear layer with unit temperature and density."""
    eta = np.clip(Y / (np.max(np.abs(Y)) + 1e-15), 0.0, 1.0)
    U = Ma * (1.5 * eta - 0.5 * eta**3)
    V = np.zeros_like(U)
    W = np.zeros_like(U)
    T = np.ones_like(U)
    rho = np.ones_like(U)
    bf = BaseFlow(
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        RHO=rho,
        U=U,
        V=V,
        W=W,
        T=T,
    )
    sutherland_air(bf)
    return bf
