# SPDX-License-Identifier: GPL-3.0-only
"""Coefficient-level regression tests."""

from __future__ import annotations

import numpy as np

from linstabpy.operators.coefficients import compute_cylindrical_coefficients


def test_cylindrical_singularity_bounded() -> None:
    """A grid containing r = 0 is refused; strictly positive radii stay bounded.

    The old mesh-spacing floor silently rewrote 1/R coefficients at real
    near-axis points (jet example 2: gain 411.72 vs the 413.72 reference).
    r = 0 now raises, and a staggered grid keeps its true, finite 1/R terms.
    """
    import pytest

    ngp = 6
    R = np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
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

    threshold = 1e8  # bounded by 1/r_min^2 = 4e4 with margin
    for attr in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy"):
        arr = getattr(coeffs, attr)
        assert np.all(np.isfinite(arr)), f"{attr} has non-finite entries for r > 0"
        max_val = float(np.max(np.abs(arr)))
        assert max_val < threshold, (
            f"Coefficient {attr} has entry {max_val:.3e} > {threshold:.3e} "
            "for strictly positive radii"
        )

    with pytest.raises(ValueError, match="pole treatment"):
        compute_cylindrical_coefficients(
            U,
            V,
            W,
            np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0]),
            RHO,
            T,
            MU,
            m,
            Re,
            Ma,
            Pr,
            kappa,
            *([zeros] * 21),
            dmudT,
            d2mudT2,
        )
