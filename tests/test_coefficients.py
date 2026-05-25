# SPDX-License-Identifier: GPL-3.0-only
"""Coefficient-level regression tests."""

from __future__ import annotations

import numpy as np

from linstabpy.operators.coefficients import compute_cylindrical_coefficients


def test_cylindrical_singularity_bounded() -> None:
    """Operator entries near the axis should not blow up to 1e28 scale."""
    ngp = 6
    R = np.array([0.0, 0.0, 0.01, 0.1, 0.5, 1.0])
    U = np.zeros(ngp)
    V = np.zeros(ngp)
    W = np.ones(ngp) * 0.5
    RHO = np.ones(ngp)
    T = np.ones(ngp)
    MU = np.ones(ngp) * 1e-3
    m = 0.0
    Re = 1e5
    Ma = 0.5
    Pr = 0.72
    kappa = 1.4
    zeros = np.zeros(ngp)
    dmudT = np.zeros(ngp)
    d2mudT2 = np.zeros(ngp)

    coeffs = compute_cylindrical_coefficients(
        U,
        V,
        W,
        R,
        RHO,
        T,
        MU,
        m,
        Re,
        Ma,
        Pr,
        kappa,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        dmudT,
        d2mudT2,
    )

    threshold = 1e8
    for attr in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy"):
        arr = getattr(coeffs, attr)
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size > 0:
            max_val = float(np.max(np.abs(finite_vals)))
            assert max_val < threshold, (
                f"Coefficient {attr} has entry {max_val:.3e} > {threshold:.3e} "
                "near axis (likely 1/r or 1/r^2 blow-up)"
            )
