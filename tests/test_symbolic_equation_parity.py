#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Numeric parity check between SymPy templates and operator implementations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps
import sympy as sp

from linstabpy.mesh.mesh import _transform_second_derivatives
from linstabpy.operators.coefficients import (
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
)
from examples.martini.symbolic_compressible_nse import (
    cartesian_lnse_coefficients_from_nse,
    cartesian_operator_coefficients,
    cartesian_symbol_table,
    continuity_operator_coefficients,
    continuity_symbol_table,
    curvilinear_operator_coefficients,
    curvilinear_symbol_table,
    cylindrical_lnse_coefficients_from_nse,
    cylindrical_operator_coefficients,
    cylindrical_symbol_table,
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rng(seed: int):
    return np.random.default_rng(seed)


def _make_random_cartesian_fields(ngp: int, seed: int) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    out: dict[str, np.ndarray] = {}

    out["U"] = rng.uniform(0.2, 1.5, size=ngp)
    out["V"] = rng.uniform(-0.4, 0.4, size=ngp)
    out["W"] = rng.uniform(-0.3, 0.3, size=ngp)
    out["RHO"] = rng.uniform(0.8, 1.4, size=ngp)
    out["T"] = rng.uniform(0.9, 1.3, size=ngp)
    out["MU"] = rng.uniform(0.7, 1.1, size=ngp)

    out["dUdx"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dVdx"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dWdx"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dRHOdx"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dTdx"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dPdx"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dUdy"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dVdy"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dWdy"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dRHOdy"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dTdy"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dPdy"] = rng.uniform(-0.2, 0.2, size=ngp)

    out["d2Udx2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Vdx2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Wdx2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Udy2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Vdy2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Wdy2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Tdx2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Tdy2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Udxdy"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Vdxdy"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Wdxdy"] = rng.uniform(-0.2, 0.2, size=ngp)

    out["dmudT"] = rng.uniform(0.01, 0.06, size=ngp)
    out["d2mudT2"] = rng.uniform(-0.01, 0.01, size=ngp)

    return out


def _make_random_cylindrical_fields(ngp: int, seed: int) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    out: dict[str, np.ndarray] = {}

    out["U"] = rng.uniform(0.2, 1.5, size=ngp)
    out["V"] = rng.uniform(-0.4, 0.4, size=ngp)
    out["W"] = rng.uniform(-0.4, 0.4, size=ngp)
    out["R"] = rng.uniform(0.35, 2.0, size=ngp)
    out["RHO"] = rng.uniform(0.8, 1.4, size=ngp)
    out["T"] = rng.uniform(0.9, 1.3, size=ngp)
    out["MU"] = rng.uniform(0.7, 1.1, size=ngp)

    out["dUdr"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dVdr"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dWdr"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dRHOdr"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dTdr"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dUdz"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dVdz"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dWdz"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dRHOdz"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["dTdz"] = rng.uniform(-0.2, 0.2, size=ngp)

    out["d2Udr2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Vdr2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Wdr2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Tdr2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Udz2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Vdz2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Wdz2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Tdz2"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Udrz"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Vdrz"] = rng.uniform(-0.2, 0.2, size=ngp)
    out["d2Wdrz"] = rng.uniform(-0.2, 0.2, size=ngp)

    out["dmudT"] = rng.uniform(0.01, 0.06, size=ngp)
    out["d2mudT2"] = rng.uniform(-0.01, 0.01, size=ngp)

    return out


def _as_array(value: Any, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.ndim == 0:
        return np.full(n, arr, dtype=complex)
    return arr


@lru_cache(maxsize=1)
def _compiled_symbolic_cartesian() -> tuple[list[str], dict[str, Any]]:
    syms = cartesian_symbol_table()
    exprs = cartesian_operator_coefficients(syms)
    order = list(syms.keys())
    args = [syms[name] for name in order]
    funcs = {key: sp.lambdify(args, expr, modules=["numpy"]) for key, expr in exprs.items()}
    return order, funcs


@lru_cache(maxsize=1)
def _compiled_symbolic_cylindrical() -> tuple[list[str], dict[str, Any]]:
    syms = cylindrical_symbol_table()
    exprs = cylindrical_operator_coefficients(syms)
    order = list(syms.keys())
    args = [syms[name] for name in order]
    funcs = {key: sp.lambdify(args, expr, modules=["numpy"]) for key, expr in exprs.items()}
    return order, funcs


@lru_cache(maxsize=1)
def _compiled_symbolic_curvilinear() -> tuple[list[str], dict[str, Any]]:
    syms = curvilinear_symbol_table()
    exprs = curvilinear_operator_coefficients(syms)
    order = list(syms.keys())
    args = [syms[name] for name in order]
    funcs = {key: sp.lambdify(args, expr, modules=["numpy"]) for key, expr in exprs.items()}
    return order, funcs


def _eval_symbolic(
    values: dict[str, Any],
    order: list[str],
    funcs: dict[str, Any],
    n: int,
) -> dict[str, np.ndarray]:
    args = [values[name] for name in order]
    return {key: _as_array(fn(*args), n) for key, fn in funcs.items()}


def _matrix_entry_residuals(
    blocks: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> dict[str, Any]:
    residuals: dict[str, float] = {}
    max_abs_by_block: dict[str, float] = {}

    for block, arr in blocks.items():
        block_max = 0.0
        for i in range(5):
            for j in range(5):
                key = f"{block}_{i}{j}"
                resid = float(np.max(np.abs(arr[i, j, :] - expected[key])))
                residuals[key] = resid
                block_max = max(block_max, resid)
        max_abs_by_block[block] = block_max

    max_abs = max(residuals.values()) if residuals else 0.0
    return {
        "n_terms_checked": len(residuals),
        "max_abs_residual": max_abs,
        "max_abs_by_block": max_abs_by_block,
        "residuals": residuals,
    }


def _run_cartesian(ngp: int, seed: int, kx: float, tol: float) -> dict[str, Any]:
    Re, Ma, Pr, kappa = 1e4, 0.3, 0.7, 1.4
    fields = _make_random_cartesian_fields(ngp=ngp, seed=seed)
    P = fields["RHO"] * fields["T"] / (Ma**2 * kappa)

    coeffs = compute_cartesian_coefficients(
        U=fields["U"],
        V=fields["V"],
        W=fields["W"],
        RHO=fields["RHO"],
        T=fields["T"],
        MU=fields["MU"],
        P=P,
        kx=kx,
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        dUdx=fields["dUdx"],
        dVdx=fields["dVdx"],
        dWdx=fields["dWdx"],
        dRHOdx=fields["dRHOdx"],
        dTdx=fields["dTdx"],
        dPdx=fields["dPdx"],
        dUdy=fields["dUdy"],
        dVdy=fields["dVdy"],
        dWdy=fields["dWdy"],
        dRHOdy=fields["dRHOdy"],
        dTdy=fields["dTdy"],
        dPdy=fields["dPdy"],
        d2Udx2=fields["d2Udx2"],
        d2Vdx2=fields["d2Vdx2"],
        d2Wdx2=fields["d2Wdx2"],
        d2Udy2=fields["d2Udy2"],
        d2Vdy2=fields["d2Vdy2"],
        d2Wdy2=fields["d2Wdy2"],
        d2Tdx2=fields["d2Tdx2"],
        d2Tdy2=fields["d2Tdy2"],
        d2Udxdy=fields["d2Udxdy"],
        d2Vdxdy=fields["d2Vdxdy"],
        d2Wdxdy=fields["d2Wdxdy"],
        dmudT=fields["dmudT"],
        d2mudT2=fields["d2mudT2"],
    )

    order, funcs = _compiled_symbolic_cartesian()
    expected = _eval_symbolic(
        {
            "U": fields["U"],
            "V": fields["V"],
            "W": fields["W"],
            "RHO": fields["RHO"],
            "T": fields["T"],
            "MU": fields["MU"],
            "P": P,
            "kx": kx,
            "Re": Re,
            "Ma": Ma,
            "Pr": Pr,
            "kappa": kappa,
            "dUdx": fields["dUdx"],
            "dVdx": fields["dVdx"],
            "dWdx": fields["dWdx"],
            "dRHOdx": fields["dRHOdx"],
            "dTdx": fields["dTdx"],
            "dPdx": fields["dPdx"],
            "dUdy": fields["dUdy"],
            "dVdy": fields["dVdy"],
            "dWdy": fields["dWdy"],
            "dRHOdy": fields["dRHOdy"],
            "dTdy": fields["dTdy"],
            "dPdy": fields["dPdy"],
            "d2Udx2": fields["d2Udx2"],
            "d2Vdx2": fields["d2Vdx2"],
            "d2Wdx2": fields["d2Wdx2"],
            "d2Udy2": fields["d2Udy2"],
            "d2Vdy2": fields["d2Vdy2"],
            "d2Wdy2": fields["d2Wdy2"],
            "d2Tdx2": fields["d2Tdx2"],
            "d2Tdy2": fields["d2Tdy2"],
            "d2Udxdy": fields["d2Udxdy"],
            "d2Vdxdy": fields["d2Vdxdy"],
            "d2Wdxdy": fields["d2Wdxdy"],
            "dmudT": fields["dmudT"],
            "d2mudT2": fields["d2mudT2"],
        },
        order,
        funcs,
        ngp,
    )

    residuals = _matrix_entry_residuals(
        {
            "A0": coeffs.A0,
            "Ax": coeffs.Ax,
            "Ay": coeffs.Ay,
            "Axx": coeffs.Axx,
            "Ayy": coeffs.Ayy,
            "Axy": coeffs.Axy,
            "B": coeffs.B,
        },
        expected,
    )

    return {
        "scope": "cartesian_full_5x5",
        "ngp": ngp,
        "seed": seed,
        "kx": kx,
        "tolerance": tol,
        **residuals,
        "pass": bool(residuals["max_abs_residual"] <= tol),
    }


def _run_cylindrical(ngp: int, seed: int, m: float, tol: float) -> dict[str, Any]:
    Re, Ma, Pr, kappa = 1e4, 0.3, 0.7, 1.4
    fields = _make_random_cylindrical_fields(ngp=ngp, seed=seed)

    coeffs = compute_cylindrical_coefficients(
        U=fields["U"],
        V=fields["V"],
        W=fields["W"],
        R=fields["R"],
        RHO=fields["RHO"],
        T=fields["T"],
        MU=fields["MU"],
        m=m,
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        dUdr=fields["dUdr"],
        dVdr=fields["dVdr"],
        dWdr=fields["dWdr"],
        dRHOdr=fields["dRHOdr"],
        dTdr=fields["dTdr"],
        dUdz=fields["dUdz"],
        dVdz=fields["dVdz"],
        dWdz=fields["dWdz"],
        dRHOdz=fields["dRHOdz"],
        dTdz=fields["dTdz"],
        d2Udr2=fields["d2Udr2"],
        d2Vdr2=fields["d2Vdr2"],
        d2Wdr2=fields["d2Wdr2"],
        d2Tdr2=fields["d2Tdr2"],
        d2Udz2=fields["d2Udz2"],
        d2Vdz2=fields["d2Vdz2"],
        d2Wdz2=fields["d2Wdz2"],
        d2Tdz2=fields["d2Tdz2"],
        d2Udrz=fields["d2Udrz"],
        d2Vdrz=fields["d2Vdrz"],
        d2Wdrz=fields["d2Wdrz"],
        dmudT=fields["dmudT"],
        d2mudT2=fields["d2mudT2"],
    )

    order, funcs = _compiled_symbolic_cylindrical()
    expected = _eval_symbolic(
        {
            "U": fields["U"],
            "V": fields["V"],
            "W": fields["W"],
            "R": fields["R"],
            "RHO": fields["RHO"],
            "T": fields["T"],
            "MU": fields["MU"],
            "m": m,
            "Re": Re,
            "Ma": Ma,
            "Pr": Pr,
            "kappa": kappa,
            "dUdr": fields["dUdr"],
            "dVdr": fields["dVdr"],
            "dWdr": fields["dWdr"],
            "dRHOdr": fields["dRHOdr"],
            "dTdr": fields["dTdr"],
            "dUdz": fields["dUdz"],
            "dVdz": fields["dVdz"],
            "dWdz": fields["dWdz"],
            "dRHOdz": fields["dRHOdz"],
            "dTdz": fields["dTdz"],
            "d2Udr2": fields["d2Udr2"],
            "d2Vdr2": fields["d2Vdr2"],
            "d2Wdr2": fields["d2Wdr2"],
            "d2Tdr2": fields["d2Tdr2"],
            "d2Udz2": fields["d2Udz2"],
            "d2Vdz2": fields["d2Vdz2"],
            "d2Wdz2": fields["d2Wdz2"],
            "d2Tdz2": fields["d2Tdz2"],
            "d2Udrz": fields["d2Udrz"],
            "d2Vdrz": fields["d2Vdrz"],
            "d2Wdrz": fields["d2Wdrz"],
            "dmudT": fields["dmudT"],
            "d2mudT2": fields["d2mudT2"],
        },
        order,
        funcs,
        ngp,
    )

    residuals = _matrix_entry_residuals(
        {
            "A0": coeffs.A0,
            "Ax": coeffs.Ax,
            "Ay": coeffs.Ay,
            "Axx": coeffs.Axx,
            "Ayy": coeffs.Ayy,
            "Axy": coeffs.Axy,
            "B": coeffs.B,
        },
        expected,
    )

    return {
        "scope": "cylindrical_full_5x5",
        "ngp": ngp,
        "seed": seed,
        "m": m,
        "tolerance": tol,
        **residuals,
        "pass": bool(residuals["max_abs_residual"] <= tol),
    }


def _run_curvilinear(ngp: int, seed: int, tol: float) -> dict[str, Any]:
    rng = _rng(seed)
    n = int(max(ngp, 8))

    Dx = sps.csr_matrix(rng.normal(scale=0.4, size=(n, n)))
    Dy = sps.csr_matrix(rng.normal(scale=0.4, size=(n, n)))
    D2x = sps.csr_matrix(rng.normal(scale=0.25, size=(n, n)))
    D2y = sps.csr_matrix(rng.normal(scale=0.25, size=(n, n)))
    Dxy = sps.csr_matrix(rng.normal(scale=0.2, size=(n, n)))
    Dyx = sps.csr_matrix(rng.normal(scale=0.2, size=(n, n)))

    x = rng.normal(size=n)
    y = rng.normal(size=n)
    f = rng.normal(size=n)

    a = rng.uniform(0.4, 1.6, size=n)
    b = rng.uniform(-0.8, 0.8, size=n)
    c = rng.uniform(-0.8, 0.8, size=n)
    d = rng.uniform(0.4, 1.6, size=n)

    D2x_new_impl, Dxy_new_impl, Dyx_new_impl, D2y_new_impl = _transform_second_derivatives(
        a,
        b,
        c,
        d,
        D2x,
        D2y,
        Dxy,
        Dyx,
        Dx,
        Dy,
        x,
        y,
    )

    Dx_new_impl = sps.diags(a, format="csr") @ Dx + sps.diags(b, format="csr") @ Dy
    Dy_new_impl = sps.diags(c, format="csr") @ Dx + sps.diags(d, format="csr") @ Dy

    dJ11_xi = D2x @ x
    dJ12_xi = D2x @ y
    dJ21_xi = Dx @ (Dy @ x)
    dJ22_xi = Dx @ (Dy @ y)
    dJ11_eta = Dy @ x
    dJ12_eta = Dy @ y
    dJ21_eta = D2y @ x
    dJ22_eta = D2y @ y

    x2p = rng.uniform(0.4, 1.8, size=n)
    x2pp = rng.uniform(-0.3, 0.3, size=n)
    y2p = rng.uniform(0.4, 1.8, size=n)
    y2pp = rng.uniform(-0.3, 0.3, size=n)

    order, funcs = _compiled_symbolic_curvilinear()
    coeff = _eval_symbolic(
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "dJ11_xi": dJ11_xi,
            "dJ12_xi": dJ12_xi,
            "dJ21_xi": dJ21_xi,
            "dJ22_xi": dJ22_xi,
            "dJ11_eta": dJ11_eta,
            "dJ12_eta": dJ12_eta,
            "dJ21_eta": dJ21_eta,
            "dJ22_eta": dJ22_eta,
            "x2p": x2p,
            "x2pp": x2pp,
            "y2p": y2p,
            "y2pp": y2pp,
        },
        order,
        funcs,
        n,
    )

    f_xi = Dx @ f
    f_eta = Dy @ f
    f_xixi = D2x @ f
    f_xieta = Dxy @ f
    f_etaxi = Dyx @ f
    f_etaeta = D2y @ f

    def _full2d_expected(block: str) -> np.ndarray:
        return (
            coeff[f"full2d_{block}_c11"] * f_xixi
            + coeff[f"full2d_{block}_c12"] * f_xieta
            + coeff[f"full2d_{block}_c21"] * f_etaxi
            + coeff[f"full2d_{block}_c22"] * f_etaeta
            + coeff[f"full2d_{block}_qx"] * f_xi
            + coeff[f"full2d_{block}_qy"] * f_eta
        )

    residuals = {
        "curv_full2d_Dx": float(
            np.max(
                np.abs(
                    (Dx_new_impl @ f)
                    - (coeff["full2d_Dx_Dx"] * f_xi + coeff["full2d_Dx_Dy"] * f_eta)
                )
            )
        ),
        "curv_full2d_Dy": float(
            np.max(
                np.abs(
                    (Dy_new_impl @ f)
                    - (coeff["full2d_Dy_Dx"] * f_xi + coeff["full2d_Dy_Dy"] * f_eta)
                )
            )
        ),
        "curv_full2d_D2x": float(np.max(np.abs((D2x_new_impl @ f) - _full2d_expected("D2x")))),
        "curv_full2d_Dxy": float(np.max(np.abs((Dxy_new_impl @ f) - _full2d_expected("Dxy")))),
        "curv_full2d_Dyx": float(np.max(np.abs((Dyx_new_impl @ f) - _full2d_expected("Dyx")))),
        "curv_full2d_D2y": float(np.max(np.abs((D2y_new_impl @ f) - _full2d_expected("D2y")))),
    }

    Dx_x_impl = sps.diags(1.0 / x2p, format="csr") @ Dx
    D2x_x_impl = (
        sps.diags(1.0 / (x2p**2), format="csr") @ D2x
        - sps.diags(x2pp / (x2p**3), format="csr") @ Dx
    )
    Dy_y_impl = sps.diags(1.0 / y2p, format="csr") @ Dy
    D2y_y_impl = (
        sps.diags(1.0 / (y2p**2), format="csr") @ D2y
        - sps.diags(y2pp / (y2p**3), format="csr") @ Dy
    )

    residuals.update(
        {
            "curv_xonly_Dx": float(np.max(np.abs((Dx_x_impl @ f) - coeff["x_only_Dx_Dx"] * f_xi))),
            "curv_xonly_D2x": float(
                np.max(
                    np.abs(
                        (D2x_x_impl @ f)
                        - (coeff["x_only_D2x_D2x"] * f_xixi + coeff["x_only_D2x_Dx"] * f_xi)
                    )
                )
            ),
            "curv_yonly_Dy": float(np.max(np.abs((Dy_y_impl @ f) - coeff["y_only_Dy_Dy"] * f_eta))),
            "curv_yonly_D2y": float(
                np.max(
                    np.abs(
                        (D2y_y_impl @ f)
                        - (coeff["y_only_D2y_D2y"] * f_etaeta + coeff["y_only_D2y_Dy"] * f_eta)
                    )
                )
            ),
        }
    )

    max_abs_by_block = {
        "curv_full2d": max(
            residuals["curv_full2d_Dx"],
            residuals["curv_full2d_Dy"],
            residuals["curv_full2d_D2x"],
            residuals["curv_full2d_Dxy"],
            residuals["curv_full2d_Dyx"],
            residuals["curv_full2d_D2y"],
        ),
        "curv_xonly": max(residuals["curv_xonly_Dx"], residuals["curv_xonly_D2x"]),
        "curv_yonly": max(residuals["curv_yonly_Dy"], residuals["curv_yonly_D2y"]),
    }

    max_abs = max(residuals.values()) if residuals else 0.0
    return {
        "scope": "curvilinear_metric_transforms",
        "ngp": n,
        "seed": seed,
        "tolerance": tol,
        "n_terms_checked": len(residuals),
        "max_abs_residual": max_abs,
        "max_abs_by_block": max_abs_by_block,
        "residuals": residuals,
        "pass": bool(max_abs <= tol),
    }


def _cheb_diff_matrix(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev-Gauss-Lobatto first-derivative matrix (Trefethen ``cheb``).

    Exact for polynomials of degree <= n-1, so spectral differentiation of a
    low-degree polynomial field is reproduced to machine precision. Returns
    ``(D1, nodes)`` with ``nodes = cos(pi*k/(n-1))``.
    """
    if n == 1:
        return np.zeros((1, 1)), np.array([1.0])
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))
    cvec = np.ones(n)
    cvec[0] = 2.0
    cvec[n - 1] = 2.0
    cvec = cvec * (-1.0) ** k
    X = np.tile(x, (n, 1)).T
    dX = X - X.T
    D = np.outer(cvec, 1.0 / cvec) / (dX + np.eye(n))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def _run_curvilinear_full2d_oracle(n: int = 8, tol: float = 1e-9) -> dict[str, Any]:
    """Validate the production full-2D metric transform against an INDEPENDENT
    analytic oracle.

    Unlike ``_run_curvilinear`` (which checks the production transform against
    the symbolic ``curvilinear_operator_coefficients`` -- a second transcription
    of the same ``dJi = -Ji.dJ.Ji`` algebra, hence self-referential), this oracle
    never uses that identity. It picks an analytic map ``(X(xi,eta), Y(xi,eta))``
    and a polynomial field ``P(xi,eta)``, computes the true physical derivatives
    of the composite mechanically with SymPy, and asserts the production transform
    reproduces them on a Chebyshev grid where spectral differentiation of the
    low-degree polynomials is exact. A dropped/garbled q-term in production would
    diverge from the SymPy ground truth and fail here.
    """
    # --- exact computational-space tensor-product operators (CGL spectral) ---
    D1, nodes = _cheb_diff_matrix(n)
    D2_1d = D1 @ D1
    eye = np.eye(n)
    Dx = sps.csr_matrix(np.kron(D1, eye))  # d/dxi
    Dy = sps.csr_matrix(np.kron(eye, D1))  # d/deta
    D2x = sps.csr_matrix(np.kron(D2_1d, eye))
    D2y = sps.csr_matrix(np.kron(eye, D2_1d))
    Dxy = (Dx @ Dy).tocsr()  # d2/dxi deta
    Dyx = (Dy @ Dx).tocsr()

    XI, ETA = np.meshgrid(nodes, nodes, indexing="ij")
    xi_flat = XI.reshape(-1)
    eta_flat = ETA.reshape(-1)

    # --- analytic map + polynomial field (the independent oracle) ---
    xi, eta = sp.symbols("xi eta", real=True)
    X = xi + sp.Rational(15, 100) * xi * eta + sp.Rational(5, 100) * eta**2  # deg 2
    Y = eta + sp.Rational(10, 100) * xi**2 - sp.Rational(8, 100) * xi * eta  # deg 2
    P = (
        1
        + sp.Rational(1, 2) * xi
        - sp.Rational(3, 10) * eta
        + sp.Rational(4, 10) * xi**2
        - sp.Rational(1, 5) * xi * eta
        + sp.Rational(7, 100) * eta**2
        + sp.Rational(1, 10) * xi**3
        - sp.Rational(6, 100) * xi**2 * eta
        + sp.Rational(3, 100) * eta**3
    )  # deg 3 field

    # forward Jacobian J = [[X_xi, Y_xi], [X_eta, Y_eta]]; Ji = J^-1 (inverse map)
    J = sp.Matrix([[sp.diff(X, xi), sp.diff(Y, xi)], [sp.diff(X, eta), sp.diff(Y, eta)]])
    Ji = J.inv()
    a_s, b_s, c_s, d_s = Ji[0, 0], Ji[0, 1], Ji[1, 0], Ji[1, 1]

    # physical-derivative operators built straight from the map (no dJi identity)
    def dX(g: sp.Expr) -> sp.Expr:
        return a_s * sp.diff(g, xi) + b_s * sp.diff(g, eta)

    def dY(g: sp.Expr) -> sp.Expr:
        return c_s * sp.diff(g, xi) + d_s * sp.diff(g, eta)

    FX = dX(P)
    FY = dY(P)
    FXX = dX(FX)
    FXY = dX(FY)
    FYX = dY(FX)
    FYY = dY(FY)

    def _sample(expr: sp.Expr) -> np.ndarray:
        fn = sp.lambdify((xi, eta), expr, "numpy")
        val = fn(xi_flat, eta_flat)
        return np.broadcast_to(np.asarray(val, dtype=float), xi_flat.shape).astype(float)

    a = _sample(a_s)
    b = _sample(b_s)
    c = _sample(c_s)
    d = _sample(d_s)
    xvec = _sample(X)
    yvec = _sample(Y)
    fvec = _sample(P)
    fx, fy = _sample(FX), _sample(FY)
    fxx, fxy, fyx, fyy = _sample(FXX), _sample(FXY), _sample(FYX), _sample(FYY)

    # --- drive production and compare to the analytic ground truth ---
    Dx_new = sps.diags(a, format="csr") @ Dx + sps.diags(b, format="csr") @ Dy
    Dy_new = sps.diags(c, format="csr") @ Dx + sps.diags(d, format="csr") @ Dy
    D2x_new, Dxy_new, Dyx_new, D2y_new = _transform_second_derivatives(
        a, b, c, d, D2x, D2y, Dxy, Dyx, Dx, Dy, xvec, yvec
    )

    residuals = {
        "full2d_Dx": float(np.max(np.abs((Dx_new @ fvec) - fx))),
        "full2d_Dy": float(np.max(np.abs((Dy_new @ fvec) - fy))),
        "full2d_D2x": float(np.max(np.abs((D2x_new @ fvec) - fxx))),
        "full2d_Dxy": float(np.max(np.abs((Dxy_new @ fvec) - fxy))),
        "full2d_Dyx": float(np.max(np.abs((Dyx_new @ fvec) - fyx))),
        "full2d_D2y": float(np.max(np.abs((D2y_new @ fvec) - fyy))),
    }
    max_abs = max(residuals.values())
    return {
        "scope": "curvilinear_full2d_independent_oracle",
        "ngp": n * n,
        "tolerance": tol,
        "n_terms_checked": len(residuals),
        "max_abs_residual": max_abs,
        "residuals": residuals,
        "pass": bool(max_abs <= tol),
    }


def _merge_reports(reports: dict[str, dict[str, Any]], tol: float) -> dict[str, Any]:
    combined_residuals: dict[str, float] = {}
    combined_block_max: dict[str, float] = {}

    for model_name, report in reports.items():
        for key, value in report["residuals"].items():
            combined_residuals[f"{model_name}:{key}"] = value
        for block, value in report["max_abs_by_block"].items():
            combined_block_max[block] = max(combined_block_max.get(block, 0.0), value)

    max_abs = max(combined_residuals.values()) if combined_residuals else 0.0
    n_terms = sum(report["n_terms_checked"] for report in reports.values())

    if len(reports) == 1:
        scope = next(iter(reports.values()))["scope"]
    elif set(reports.keys()) == {"cartesian", "cylindrical"}:
        scope = "full_5x5_cartesian_and_cylindrical"
    else:
        scope = "full_symbolic_coverage"

    return {
        "scope": scope,
        "tolerance": tol,
        "n_terms_checked": n_terms,
        "max_abs_residual": max_abs,
        "max_abs_by_block": combined_block_max,
        "residuals": combined_residuals,
        "models": reports,
        "pass": bool(max_abs <= tol and all(r["pass"] for r in reports.values())),
    }


def run_parity(
    ngp: int,
    seed: int,
    kx: float,
    m: float,
    tol: float,
    mode: str,
) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}

    if mode in ("cartesian", "both", "all"):
        reports["cartesian"] = _run_cartesian(ngp=ngp, seed=seed, kx=kx, tol=tol)
    if mode in ("cylindrical", "both", "all"):
        reports["cylindrical"] = _run_cylindrical(ngp=ngp, seed=seed + 17, m=m, tol=tol)
    if mode in ("curvilinear", "all"):
        reports["curvilinear"] = _run_curvilinear(ngp=ngp, seed=seed + 31, tol=tol)

    merged = _merge_reports(reports, tol=tol)
    return {
        "generated_at_utc": _now_utc(),
        "ngp": ngp,
        "seed": seed,
        "kx": kx,
        "m": m,
        "mode": mode,
        **merged,
    }


# ---------------------------------------------------------------------------
# pytest entry points
#
# This module began as a standalone CLI (see ``main`` below), so the symbolic
# guard it implements was never collected by pytest. It silently went dark when
# the P1 coefficient fusion removed the named ``c_*_ij`` assignments its
# reference parser depends on. The tests below keep the guard permanently in the
# suite. ``_coeff_source_path`` now parses the frozen pre-fusion reference, so
# the cartesian/cylindrical checks assert that the live fused production
# coefficients still match the validated reference algebra (symbolically), and
# ``test_fused_coefficients_match_frozen_reference_numerically`` states that
# equivalence directly.
# ---------------------------------------------------------------------------

_FROZEN_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "martini"
    / "coefficients_reference_base.py"
)


def _load_frozen_reference():
    import importlib.util

    spec = importlib.util.spec_from_file_location("coefficients_reference_base", _FROZEN_REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cartesian_coefficients_match_frozen_reference_symbolically():
    report = _run_cartesian(ngp=512, seed=20260531, kx=0.73, tol=1e-9)
    assert report["pass"], report["max_abs_by_block"]


def test_cylindrical_coefficients_match_frozen_reference_symbolically():
    report = _run_cylindrical(ngp=512, seed=20260532, m=2.0, tol=1e-9)
    assert report["pass"], report["max_abs_by_block"]


def test_curvilinear_metric_transforms_match_analytic_chain_rule():
    # This harness draws (a, b, c, d) independently of the (x, y) mapping that
    # feeds the dJ terms, so only the x-only and y-only deformation branches form
    # a self-consistent metric; those match the analytic chain rule to machine
    # precision and are asserted here. The full-2D block is NOT asserted in this
    # test: with decoupled random inputs its metric is physically inconsistent and
    # the full-2D identity cannot hold (residual ~1e4). The full-2D transform is
    # independently validated in
    # test_curvilinear_full2d_matches_independent_analytic_oracle, which uses a
    # consistent analytic map and a SymPy ground truth that does not share the
    # production dJi = -Ji.dJ.Ji algebra.
    report = _run_curvilinear(ngp=512, seed=20260533, tol=1e-9)
    blocks = report["max_abs_by_block"]
    assert blocks["curv_xonly"] < 1e-9, blocks
    assert blocks["curv_yonly"] < 1e-9, blocks


def test_curvilinear_full2d_matches_independent_analytic_oracle():
    # Independent oracle for the full-2D metric transform: an analytic map and a
    # polynomial field whose true physical derivatives are computed mechanically
    # by SymPy (never via production's dJi = -Ji.dJ.Ji q-term algebra), checked on
    # a Chebyshev grid where spectral differentiation of the polynomials is exact.
    # A dropped or garbled q-term in _transform_second_derivatives would diverge
    # from this ground truth -- the self-referential blind spot that hid the
    # cylindrical A0_44 bug does not apply here.
    # Bead ...2.6 acceptance is <1e-10; production reaches ~4.3e-13 here, so 1e-10
    # leaves ~2 orders of margin while matching the stated oracle tolerance.
    report = _run_curvilinear_full2d_oracle(n=8, tol=1e-10)
    assert report["pass"], report["residuals"]


def test_fused_coefficients_match_frozen_reference_numerically():
    """The fused production coefficients reproduce the frozen pre-fusion
    reference to ~machine precision on random fields (proves P1 changed no
    algebra, only operation ordering)."""
    ref = _load_frozen_reference()
    Re, Ma, Pr, kappa = 1e4, 0.3, 0.7, 1.4
    fields = _make_random_cartesian_fields(ngp=512, seed=20260534)
    P = fields["RHO"] * fields["T"] / (Ma**2 * kappa)
    kwargs = dict(P=P, kx=0.73, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, **fields)

    fused = compute_cartesian_coefficients(**kwargs)
    reference = ref.compute_cartesian_coefficients(**kwargs)

    max_abs = 0.0
    for block in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
        diff = float(np.max(np.abs(getattr(fused, block) - getattr(reference, block))))
        max_abs = max(max_abs, diff)
    assert max_abs < 1e-10, f"fused vs frozen reference max|Δ| = {max_abs:.3e}"


def test_cartesian_coefficients_match_independent_nse_derivation():
    """Independent first-principles derivation of all 175 cartesian LNSE
    coefficients must agree with the production compute_cartesian_coefficients
    to machine precision.  If any entry disagrees the test FAILS and prints the
    exact block, index, symbolic expression, and numeric values.

    Also verifies: the continuity row of the independent derivation reproduces
    continuity_operator_coefficients exactly (sympy.simplify == 0 for all 10
    nonzero anchor entries).
    """
    import sympy as sp  # noqa: PLC0415 (local re-import for clarity)

    # ---- 1. Continuity anchor check (symbolic) --------------------------
    cart_syms = cartesian_symbol_table()
    cont_syms = continuity_symbol_table()
    derived = cartesian_lnse_coefficients_from_nse(cart_syms)
    anchor = continuity_operator_coefficients(cont_syms)

    # Build substitution map: continuity_symbol_table uses 'rho'; rest share names.
    subs = {cont_syms["rho"]: cart_syms["RHO"]}
    for name in ("U", "V", "W", "kx", "dUdx", "dVdy", "dRHOdx", "dRHOdy"):
        subs[cont_syms[name]] = cart_syms[name]

    anchor_entries = {
        "A0_00": anchor["A0_00"].subs(subs),
        "A0_01": anchor["A0_01"].subs(subs),
        "A0_02": anchor["A0_02"].subs(subs),
        "A0_03": anchor["A0_03"].subs(subs),
        "A0_04": sp.Integer(0),
        "Ax_00": anchor["Ax_00"].subs(subs),
        "Ax_01": anchor["Ax_01"].subs(subs),
        "Ay_00": anchor["Ay_00"].subs(subs),
        "Ay_02": anchor["Ay_02"].subs(subs),
        "B_00": anchor["B_00"],
    }
    continuity_anchor_ok = True
    for key, anchor_expr in anchor_entries.items():
        diff = sp.simplify(derived[key] - anchor_expr)
        if diff != 0:
            continuity_anchor_ok = False
            print(f"  ANCHOR MISMATCH {key}: simplify(derived - anchor) = {diff}")
    assert continuity_anchor_ok, "Continuity anchor mismatch: derivation framework is wrong"

    # ---- 2. Numeric parity vs production code ---------------------------
    Re, Ma, Pr, kappa = 1e4, 0.3, 0.7, 1.4
    ngp = 256
    fields = _make_random_cartesian_fields(ngp=ngp, seed=20260531)
    P_arr = fields["RHO"] * fields["T"] / (Ma**2 * kappa)

    prod = compute_cartesian_coefficients(
        P=P_arr, kx=0.73, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, **fields
    )
    prod_blocks = {
        "A0": prod.A0,
        "Ax": prod.Ax,
        "Ay": prod.Ay,
        "Axx": prod.Axx,
        "Ayy": prod.Ayy,
        "Axy": prod.Axy,
        "B": prod.B,
    }

    sym_order = list(cart_syms.keys())
    sym_args = [cart_syms[n] for n in sym_order]
    field_vals = {
        **fields,
        "P": P_arr,
        "kx": 0.73,
        "Re": Re,
        "Ma": Ma,
        "Pr": Pr,
        "kappa": kappa,
    }
    arg_vals = [field_vals[n] for n in sym_order]

    row_labels = ["continuity", "x-mom", "y-mom", "z-mom", "energy"]
    tol = 1e-10
    all_pass = True
    row_max = {label: 0.0 for label in row_labels}

    print("\nPer-block residuals (independent NSE derivation vs production):")
    for block in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
        block_max = 0.0
        for i in range(5):
            for j in range(5):
                key = f"{block}_{i}{j}"
                expr = derived[key]
                fn = sp.lambdify(sym_args, expr, modules=["numpy"])
                sym_val = np.asarray(fn(*arg_vals), dtype=complex)
                if sym_val.ndim == 0:
                    sym_val = np.full(ngp, sym_val)
                code_val = prod_blocks[block][i, j, :]
                resid = float(np.max(np.abs(code_val - sym_val)))
                block_max = max(block_max, resid)
                row_max[row_labels[i]] = max(row_max[row_labels[i]], resid)
                if resid >= tol:
                    all_pass = False
                    print(
                        f"  DISAGREEMENT {block}[{i},{j}] ({row_labels[i]} row): max|Δ|={resid:.3e}"
                    )
                    print(f"    symbolic: {expr}")
                    print(f"    code[0]={code_val[0]:.6e}, derivation[0]={sym_val[0]:.6e}")
        print(f"  {block}: max|Δ|={block_max:.3e}")

    print("\nPer-row max residual:")
    for label in row_labels:
        print(f"  {label:12s}: max|Δ|={row_max[label]:.3e}")

    assert all_pass, (
        "Independent NSE derivation disagrees with production code (see printed entries above)"
    )


def test_cylindrical_coefficients_match_independent_nse_derivation():
    """Independent first-principles cylindrical NSE derivation must match
    the production compute_cylindrical_coefficients to machine precision.

    This guard is NOT self-referential: the derivation in
    cylindrical_lnse_coefficients_from_nse is a conservative cylindrical
    NSE linearization derived mechanically from first principles, independent
    of the production code path. It would catch a MATLAB->Python translation
    bug (e.g. the historical A0_55 sign error in Front E) that a frozen-
    reference round-trip cannot detect.

    Convention: frozen-base-viscosity -- Martini's LNSE operator excludes
    the perturbation-viscosity x base-stress/heat-flux divergence terms
    (all O(dmudT/Re)). The _momentum_temperature_base_stress_corrections and
    _energy_temperature_base_flux_correction helpers apply this convention
    consistently so the independent derivation matches the production operator.

    Parameters: Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, m in {0.0, 1.0, 2.0},
    seed=42, ngp=64.
    """
    Re, Ma, Pr, kappa = 1e4, 0.3, 0.7, 1.4
    ngp = 64
    seed = 42
    tol = 1e-10

    syms = cylindrical_symbol_table()
    derived = cylindrical_lnse_coefficients_from_nse(syms)

    sym_order = list(syms.keys())
    sym_args = [syms[n] for n in sym_order]
    lambdified = {
        key: sp.lambdify(sym_args, expr, modules=["numpy"]) for key, expr in derived.items()
    }

    overall_max = 0.0
    worst_block = ""
    worst_i = -1
    worst_j = -1

    for m_val in (0.0, 1.0, 2.0):
        fields = _make_random_cylindrical_fields(ngp=ngp, seed=seed)
        field_vals = {
            **fields,
            "m": m_val,
            "Re": Re,
            "Ma": Ma,
            "Pr": Pr,
            "kappa": kappa,
        }
        arg_vals = [field_vals[n] for n in sym_order]

        prod = compute_cylindrical_coefficients(
            U=fields["U"],
            V=fields["V"],
            W=fields["W"],
            R=fields["R"],
            RHO=fields["RHO"],
            T=fields["T"],
            MU=fields["MU"],
            m=m_val,
            Re=Re,
            Ma=Ma,
            Pr=Pr,
            kappa=kappa,
            dUdr=fields["dUdr"],
            dVdr=fields["dVdr"],
            dWdr=fields["dWdr"],
            dRHOdr=fields["dRHOdr"],
            dTdr=fields["dTdr"],
            dUdz=fields["dUdz"],
            dVdz=fields["dVdz"],
            dWdz=fields["dWdz"],
            dRHOdz=fields["dRHOdz"],
            dTdz=fields["dTdz"],
            d2Udr2=fields["d2Udr2"],
            d2Vdr2=fields["d2Vdr2"],
            d2Wdr2=fields["d2Wdr2"],
            d2Tdr2=fields["d2Tdr2"],
            d2Udz2=fields["d2Udz2"],
            d2Vdz2=fields["d2Vdz2"],
            d2Wdz2=fields["d2Wdz2"],
            d2Tdz2=fields["d2Tdz2"],
            d2Udrz=fields["d2Udrz"],
            d2Vdrz=fields["d2Vdrz"],
            d2Wdrz=fields["d2Wdrz"],
            dmudT=fields["dmudT"],
            d2mudT2=fields["d2mudT2"],
        )
        prod_blocks = {
            "A0": prod.A0,
            "Ax": prod.Ax,
            "Ay": prod.Ay,
            "Axx": prod.Axx,
            "Ayy": prod.Ayy,
            "Axy": prod.Axy,
            "B": prod.B,
        }

        for block in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
            for i in range(5):
                for j in range(5):
                    key = f"{block}_{i}{j}"
                    fn = lambdified[key]
                    sym_val = np.asarray(fn(*arg_vals), dtype=complex)
                    if sym_val.ndim == 0:
                        sym_val = np.full(ngp, sym_val)
                    code_val = prod_blocks[block][i, j, :]
                    resid = float(np.max(np.abs(code_val - sym_val)))
                    if resid > overall_max:
                        overall_max = resid
                        worst_block = block
                        worst_i = i
                        worst_j = j

    assert overall_max < tol, (
        f"Independent cylindrical NSE derivation disagrees with production: "
        f"max|delta|={overall_max:.3e} at block={worst_block}[{worst_i},{worst_j}] "
        f"(tol={tol:.1e})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngp", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kx", type=float, default=0.8)
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument(
        "--mode",
        choices=("cartesian", "cylindrical", "curvilinear", "both", "all"),
        default="all",
    )
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/martini/operator_symbolic_parity.json"),
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    report = run_parity(
        ngp=args.ngp,
        seed=args.seed,
        kx=args.kx,
        m=args.m,
        tol=args.tol,
        mode=args.mode,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "symbolic parity:",
        f"mode={report['mode']}",
        f"pass={report['pass']}",
        f"n_terms={report['n_terms_checked']}",
        f"max_abs_residual={report['max_abs_residual']:.3e}",
        f"output={args.output}",
    )
    if not report["pass"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
