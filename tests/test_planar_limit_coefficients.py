# SPDX-License-Identifier: GPL-3.0-only
"""Large-radius planar limit of the axisymmetric coefficients.

Take a cylinder of radius R0 + y with mode m = beta * R0 and a fixed base profile.
When R0 grows, the flow tends to the planar flow with spanwise wavenumber beta.
The axisymmetric coefficient matrices must then tend to the Cartesian ones.
The error must be of order 1 / R0.

The two coefficient sets do not have the same form.
The Cartesian set has a diagonal time-derivative matrix B (primitive variables).
The axisymmetric set has a lower-triangular B (conservative momentum and energy).
A pointwise matrix M = B_axi B_cart^-1 links the rows of the two sets.
Each test compares A_axi with M A_cart.
M does not remove the base-flow residual terms in A0.
For A0, each test uses a base flow that satisfies the steady equations.
That flow is a parallel shear flow with heat conduction and viscous heating.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from numpy.typing import NDArray

from linstabpy.operators.coefficients import (
    LNSECoefficients,
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
)

x, y = sp.symbols("x y")
RE, MA, PR, KAPPA, BETA = 500.0, 0.4, 0.7, 1.4, 0.8
R0_VALUES = (1e2, 1e3, 1e4, 1e5)
PTS_X = np.array([0.3, 1.1, 2.0, 2.9, 4.2])
PTS_Y = np.array([0.2, 0.5, 0.8, 1.0, 0.35])
FIELDS = ("U", "V", "W", "RHO", "T")
# Axisymmetric (radial, azimuthal, axial) velocity taken from the planar (V, W, U).
CYL_OF = {"U": "V", "V": "W", "W": "U"}
Profile = dict[str, sp.Expr]
Array = NDArray[np.float64]


def _generic_profile() -> Profile:
    # Smooth 2-D profile with every gradient present. It is not a steady solution.
    f: Profile = {
        "U": 1 + 0.3 * sp.cos(x) * sp.sin(y + 0.5),
        "V": 0.2 * sp.sin(x) * sp.cos(y),
        "W": 0.1 + 0.4 * sp.sin(y + 0.2) * sp.cos(0.5 * x),
        "RHO": 1 + 0.2 * sp.sin(x + y),
        "T": 1 + 0.1 * sp.cos(x - y),
    }
    f["MU"] = f["T"] ** 0.7
    f["dmudT"] = 0.7 * f["T"] ** -0.3
    f["d2mudT2"] = 0.7 * -0.3 * f["T"] ** -1.3
    return f


def _shear_profile() -> Profile:
    # Parallel shear flow that satisfies the steady compressible equations:
    # mu U' = a, mu W' = b, constant pressure, (k T')' = -mu (U'^2 + W'^2),
    # k = mu cp / Pr and mu = 1 + 0.3 y.
    a, b, c0, t0 = 0.5, 0.3, 0.5, 1.0
    mu = 1 + 0.3 * y
    cp_over_pr = 1.0 / ((KAPPA - 1.0) * MA**2 * PR)
    s = (a**2 + b**2) / 0.3
    ln_mu = sp.log(mu)
    temp = t0 + (c0 * ln_mu - s * ln_mu**2 / 2) / (0.3 * cp_over_pr)
    dmudt = sp.diff(mu, y) / sp.diff(temp, y)
    return {
        "U": 1.0 + a * ln_mu / 0.3,
        "V": sp.Integer(0),
        "W": 0.1 + b * ln_mu / 0.3,
        "RHO": 1 / temp,
        "T": temp,
        "MU": mu,
        "dmudT": dmudt,
        "d2mudT2": sp.diff(dmudt, y) / sp.diff(temp, y),
    }


def _ev(expr: sp.Expr) -> Array:
    value = sp.lambdify((x, y), expr, "numpy")(PTS_X, PTS_Y)
    return np.broadcast_to(np.asarray(value, dtype=float), PTS_X.shape).copy()


def _cartesian(f: Profile) -> LNSECoefficients:
    pressure = f["RHO"] * f["T"] / (MA**2 * KAPPA)
    kw: dict[str, Array | float] = {n: _ev(f[n]) for n in (*FIELDS, "MU", "dmudT", "d2mudT2")}
    kw.update(P=_ev(pressure), kx=BETA, Re=RE, Ma=MA, Pr=PR, kappa=KAPPA)
    for n in (*FIELDS, "P"):
        e = pressure if n == "P" else f[n]
        kw[f"d{n}dx"] = _ev(sp.diff(e, x))
        kw[f"d{n}dy"] = _ev(sp.diff(e, y))
    for n in ("U", "V", "W", "T"):
        kw[f"d2{n}dx2"] = _ev(sp.diff(f[n], x, 2))
        kw[f"d2{n}dy2"] = _ev(sp.diff(f[n], y, 2))
    for n in ("U", "V"):
        kw[f"d2{n}dxdy"] = _ev(sp.diff(f[n], x, y))
    kw["d2Wdxdy"] = np.empty(0)  # not used by the Cartesian formulas
    return compute_cartesian_coefficients(**kw)


def _axisymmetric(f: Profile, r0: float, *, parity: bool = False) -> LNSECoefficients:
    # Same profile on the radius r0 + y; parity switches to the LinStab2D operator.
    kw: dict[str, Array | float | bool] = {
        "R": r0 + PTS_Y,
        "m": BETA * r0,
        "Re": RE,
        "Ma": MA,
        "Pr": PR,
        "kappa": KAPPA,
        "RHO": _ev(f["RHO"]),
        "T": _ev(f["T"]),
        "MU": _ev(f["MU"]),
        "dmudT": _ev(f["dmudT"]),
        "d2mudT2": _ev(f["d2mudT2"]),
        "matlab_parity_dmudt": parity,
    }
    for n, p in CYL_OF.items():
        kw[n] = _ev(f[p])
        kw[f"d{n}dr"] = _ev(sp.diff(f[p], y))
        kw[f"d{n}dz"] = _ev(sp.diff(f[p], x))
        kw[f"d2{n}dr2"] = _ev(sp.diff(f[p], y, 2))
        kw[f"d2{n}dz2"] = _ev(sp.diff(f[p], x, 2))
    for n in ("RHO", "T"):
        kw[f"d{n}dr"] = _ev(sp.diff(f[n], y))
        kw[f"d{n}dz"] = _ev(sp.diff(f[n], x))
    kw["d2Tdr2"] = _ev(sp.diff(f["T"], y, 2))
    kw["d2Tdz2"] = _ev(sp.diff(f["T"], x, 2))
    kw["d2Udrz"] = _ev(sp.diff(f["V"], x, y))
    kw["d2Wdrz"] = _ev(sp.diff(f["U"], x, y))
    kw["d2Vdrz"] = np.empty(0)  # not used by the axisymmetric formulas
    return compute_cylindrical_coefficients(**kw)


def _mmul(m: NDArray[np.complex128], a: NDArray[np.complex128]) -> NDArray[np.complex128]:
    # Pointwise matrix product of two (5, 5, ngp) stacks.
    return np.einsum("ikp,kjp->ijp", m, a)


def _errors(
    f: Profile,
    names: tuple[str, ...],
    *,
    parity: bool = False,
    cols: slice = slice(None),
) -> dict[str, list[float]]:
    # Max |A_axi - M A_cart| over the points and the selected columns, for each R0.
    cart = _cartesian(f)
    b_inv = np.moveaxis(np.linalg.inv(np.moveaxis(cart.B, 2, 0)), 0, 2)
    out: dict[str, list[float]] = {n: [] for n in names}
    for r0 in R0_VALUES:
        axi = _axisymmetric(f, r0, parity=parity)
        m = _mmul(axi.B, b_inv)
        for n in names:
            diff = getattr(axi, n) - _mmul(m, getattr(cart, n))
            out[n].append(float(np.abs(diff[:, cols]).max()))
    return out


def _observed_order(errors: list[float]) -> float:
    return float(-np.polyfit(np.log(R0_VALUES), np.log(errors), 1)[0])


def _assert_first_order(name: str, errors: list[float]) -> None:
    if max(errors) < 1e-12:  # the two sets agree to rounding
        return
    assert errors[-1] < errors[0] / 500.0, (name, errors)
    assert _observed_order(errors) == pytest.approx(1.0, abs=0.1), (name, errors)


def _assert_no_decay(errors: list[float]) -> None:
    # The error reaches a constant: no change above 5 % between the last two radii.
    assert errors[-1] > 1e-3, errors
    assert errors[-1] == pytest.approx(errors[-2], rel=0.05), errors


def test_literal_comparison_differs_because_the_forms_differ() -> None:
    """B is diagonal in the Cartesian set and has entries below the diagonal in the other."""
    f = _generic_profile()
    cart, axi = _cartesian(f), _axisymmetric(f, 1e5)
    assert np.abs(axi.B - cart.B).max() > 1.0
    assert np.abs(np.triu(axi.B[:, :, 0], 1)).max() < 1e-14
    assert np.abs(np.tril(axi.B[:, :, 0], -1)).max() > 0.1


def test_derivative_coefficients_tend_to_the_planar_ones() -> None:
    """Ax, Ay, Axx, Ayy, Axy and B differ from the planar ones by O(1 / R0).

    The LinStab2D operator omits only zeroth-order terms, so both operators pass.
    """
    for parity in (False, True):
        errors = _errors(_generic_profile(), ("Ax", "Ay", "Axx", "Ayy", "Axy", "B"), parity=parity)
        for name, e in errors.items():
            _assert_first_order(name, e)


def test_a0_tends_to_the_planar_one_on_a_steady_base_flow() -> None:
    """All seven matrices differ from the planar ones by O(1 / R0)."""
    errors = _errors(_shear_profile(), ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"))
    for name, e in errors.items():
        _assert_first_order(name, e)


def test_matlab_parity_operator_does_not_converge_in_the_t_column() -> None:
    """The LinStab2D operator omits mu'(T) terms: A0 keeps an O(1) error in the T column."""
    f = _shear_profile()
    _assert_first_order("A0 default", _errors(f, ("A0",))["A0"])
    _assert_no_decay(_errors(f, ("A0",), parity=True, cols=slice(4, 5))["A0"])
    _assert_first_order(
        "A0 other columns", _errors(f, ("A0",), parity=True, cols=slice(0, 4))["A0"]
    )
