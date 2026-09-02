# SPDX-License-Identifier: GPL-3.0-only
"""Focused tests for viscosity models that do not need mesh/PETSc setup."""

from __future__ import annotations

import numpy as np

from linstabpy.baseflow.baseflow import BaseFlow
from linstabpy.baseflow.viscosity import sutherland_air


def _sutherland_mu_dimensional(temperature: np.ndarray | float) -> np.ndarray:
    sutherland_temperature = 110.4
    reference_temperature = 280.0
    reference_mu = 1.735e-5
    temperature = np.asarray(temperature, dtype=float)
    return (
        reference_mu
        * (reference_temperature + sutherland_temperature)
        / (temperature + sutherland_temperature)
        * (temperature / reference_temperature) ** 1.5
    )


def test_sutherland_mu_is_scaled_by_mu_at_T0_not_mu_ref() -> None:
    t0 = 300.0
    dimensional_temperature = np.array([[280.0, 300.0, 450.0]], dtype=float)
    bf = BaseFlow(
        Re=1.0e4,
        Ma=0.3,
        Pr=0.72,
        kappa=1.4,
        T_0=t0,
        RHO=np.ones_like(dimensional_temperature),
        U=np.ones_like(dimensional_temperature),
        V=np.zeros_like(dimensional_temperature),
        W=np.zeros_like(dimensional_temperature),
        T=dimensional_temperature / t0,
    )

    sutherland_air(bf)

    mu0 = _sutherland_mu_dimensional(t0)
    expected = _sutherland_mu_dimensional(dimensional_temperature) / mu0
    np.testing.assert_allclose(bf.MU, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(bf.MU[0, 1], 1.0, rtol=1e-14, atol=1e-14)
    assert not np.isclose(bf.MU[0, 0], 1.0)


def test_sutherland_energy_coefficients_follow_gamma_re_pr_ma() -> None:
    """cv, c1 and c2 are the LNSE groups from gamma, Re, Pr and Ma."""
    cases = (
        {"ma": 0.3, "kappa": 1.4, "re": 1.0e4, "pr": 0.72},
        {"ma": 2.15, "kappa": 1.4, "re": 1.0e5, "pr": 0.72},
    )
    cv_values: list[float] = []
    c2_values: list[float] = []
    for case in cases:
        temperature = np.array([[1.0]], dtype=float)
        bf = BaseFlow(
            Re=case["re"],
            Ma=case["ma"],
            Pr=case["pr"],
            kappa=case["kappa"],
            T_0=300.0,
            RHO=np.ones_like(temperature),
            U=np.ones_like(temperature),
            V=np.zeros_like(temperature),
            W=np.zeros_like(temperature),
            T=temperature,
        )
        sutherland_air(bf)
        gamma = case["kappa"]
        mach = case["ma"]
        np.testing.assert_allclose(
            bf.cv, 1.0 / (gamma * (gamma - 1.0) * mach**2), rtol=1e-14, atol=0.0
        )
        np.testing.assert_allclose(
            bf.c1, (gamma - 1.0) * case["re"] * case["pr"] * mach**2, rtol=1e-14, atol=0.0
        )
        np.testing.assert_allclose(bf.c2, gamma * mach**2, rtol=1e-14, atol=0.0)
        assert bf.cv > 0.0
        assert bf.c1 > 0.0
        assert bf.c2 > 0.0
        cv_values.append(float(bf.cv))
        c2_values.append(float(bf.c2))
    assert c2_values[1] > c2_values[0]
    assert cv_values[1] < cv_values[0]


def test_sutherland_derivatives_are_with_respect_to_nondimensional_temperature() -> None:
    t0 = 288.15
    nondim_temperature = np.array([[0.85, 1.0, 1.35]], dtype=float)
    shape = nondim_temperature.shape
    bf = BaseFlow(
        Re=1.0e4,
        Ma=0.3,
        Pr=0.72,
        kappa=1.4,
        T_0=t0,
        RHO=np.ones(shape),
        U=np.ones(shape),
        V=np.zeros(shape),
        W=np.zeros(shape),
        T=nondim_temperature,
    )

    sutherland_air(bf)

    def mu_nd(T_nd: np.ndarray) -> np.ndarray:
        return _sutherland_mu_dimensional(T_nd * t0) / _sutherland_mu_dimensional(t0)

    h = 3e-4
    first = (mu_nd(nondim_temperature + h) - mu_nd(nondim_temperature - h)) / (2.0 * h)
    second = (
        mu_nd(nondim_temperature + h)
        - 2.0 * mu_nd(nondim_temperature)
        + mu_nd(nondim_temperature - h)
    ) / h**2

    np.testing.assert_allclose(bf.dmudT, first, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(bf.d2mudT2, second, rtol=1e-5, atol=1e-7)
