#!/usr/bin/env python3
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

from linstab2d.mesh.mesh import _transform_second_derivatives
from linstab2d.operators.coefficients import (
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
)

try:
    from symbolic_compressible_nse import (
        cartesian_operator_coefficients,
        cartesian_symbol_table,
        curvilinear_operator_coefficients,
        curvilinear_symbol_table,
        cylindrical_operator_coefficients,
        cylindrical_symbol_table,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation style
    from tools.parity.symbolic_compressible_nse import (
        cartesian_operator_coefficients,
        cartesian_symbol_table,
        curvilinear_operator_coefficients,
        curvilinear_symbol_table,
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
            "curv_xonly_Dx": float(
                np.max(np.abs((Dx_x_impl @ f) - coeff["x_only_Dx_Dx"] * f_xi))
            ),
            "curv_xonly_D2x": float(
                np.max(
                    np.abs(
                        (D2x_x_impl @ f)
                        - (
                            coeff["x_only_D2x_D2x"] * f_xixi
                            + coeff["x_only_D2x_Dx"] * f_xi
                        )
                    )
                )
            ),
            "curv_yonly_Dy": float(
                np.max(np.abs((Dy_y_impl @ f) - coeff["y_only_Dy_Dy"] * f_eta))
            ),
            "curv_yonly_D2y": float(
                np.max(
                    np.abs(
                        (D2y_y_impl @ f)
                        - (
                            coeff["y_only_D2y_D2y"] * f_etaeta
                            + coeff["y_only_D2y_Dy"] * f_eta
                        )
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
        default=Path("validation/martini2024/output/operator_symbolic_parity.json"),
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
