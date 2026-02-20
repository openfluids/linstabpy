#!/usr/bin/env python3
"""Numeric parity check between SymPy continuity template and operator code."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from linstab2d.operators.coefficients import compute_cartesian_coefficients

try:
    from symbolic_compressible_nse import (
        continuity_operator_coefficients,
        continuity_symbol_table,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation style
    from tools.parity.symbolic_compressible_nse import (
        continuity_operator_coefficients,
        continuity_symbol_table,
    )


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rng(seed: int):
    return np.random.default_rng(seed)


def _make_random_fields(ngp: int, seed: int) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    out: dict[str, np.ndarray] = {}

    # Keep physical fields bounded away from zero where needed.
    out["U"] = rng.uniform(0.2, 1.5, size=ngp)
    out["V"] = rng.uniform(-0.4, 0.4, size=ngp)
    out["W"] = rng.uniform(-0.3, 0.3, size=ngp)
    out["RHO"] = rng.uniform(0.8, 1.4, size=ngp)
    out["T"] = rng.uniform(0.9, 1.3, size=ngp)
    out["MU"] = rng.uniform(0.7, 1.1, size=ngp)

    # First derivatives
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

    # Second derivatives
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

    # Viscosity derivatives
    out["dmudT"] = rng.uniform(0.01, 0.06, size=ngp)
    out["d2mudT2"] = rng.uniform(-0.01, 0.01, size=ngp)

    return out


def _build_expected_continuity(fields: dict[str, np.ndarray], kx: float) -> dict[str, np.ndarray]:
    syms = continuity_symbol_table()
    exprs = continuity_operator_coefficients(syms)
    order = list(syms.keys())

    fn = {
        key: sp.lambdify(
            [syms[name] for name in order],
            expr,
            modules=["numpy"],
        )
        for key, expr in exprs.items()
    }

    values = [
        fields["RHO"],
        fields["U"],
        fields["V"],
        fields["W"],
        np.full_like(fields["U"], fill_value=kx, dtype=float),
        fields["dUdx"],
        fields["dVdy"],
        fields["dRHOdx"],
        fields["dRHOdy"],
    ]

    return {key: np.asarray(fn_key(*values), dtype=complex) for key, fn_key in fn.items()}


def run_parity(ngp: int, seed: int, kx: float, tol: float) -> dict[str, Any]:
    Re, Ma, Pr, kappa = 1e4, 0.3, 0.7, 1.4
    fields = _make_random_fields(ngp=ngp, seed=seed)
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

    expected = _build_expected_continuity(fields, kx=kx)
    residuals: dict[str, float] = {}

    residuals["A0_00"] = float(np.max(np.abs(coeffs.A0[0, 0, :] - expected["A0_00"])))
    residuals["A0_01"] = float(np.max(np.abs(coeffs.A0[0, 1, :] - expected["A0_01"])))
    residuals["A0_02"] = float(np.max(np.abs(coeffs.A0[0, 2, :] - expected["A0_02"])))
    residuals["A0_03"] = float(np.max(np.abs(coeffs.A0[0, 3, :] - expected["A0_03"])))
    residuals["A0_04"] = float(np.max(np.abs(coeffs.A0[0, 4, :] - expected["A0_04"])))
    residuals["Ax_00"] = float(np.max(np.abs(coeffs.Ax[0, 0, :] - expected["Ax_00"])))
    residuals["Ax_01"] = float(np.max(np.abs(coeffs.Ax[0, 1, :] - expected["Ax_01"])))
    residuals["Ay_00"] = float(np.max(np.abs(coeffs.Ay[0, 0, :] - expected["Ay_00"])))
    residuals["Ay_02"] = float(np.max(np.abs(coeffs.Ay[0, 2, :] - expected["Ay_02"])))
    residuals["B_00"] = float(np.max(np.abs(coeffs.B[0, 0, :] - expected["B_00"])))

    # Structural zero checks in continuity row.
    residuals["Axx_row0"] = float(np.max(np.abs(coeffs.Axx[0, :, :])))
    residuals["Ayy_row0"] = float(np.max(np.abs(coeffs.Ayy[0, :, :])))
    residuals["Axy_row0"] = float(np.max(np.abs(coeffs.Axy[0, :, :])))

    max_abs = max(residuals.values()) if residuals else 0.0
    return {
        "generated_at_utc": _now_utc(),
        "scope": "cartesian_continuity_row",
        "ngp": ngp,
        "seed": seed,
        "kx": kx,
        "tolerance": tol,
        "max_abs_residual": max_abs,
        "residuals": residuals,
        "pass": bool(max_abs <= tol),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngp", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kx", type=float, default=0.8)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/operator_symbolic_parity.json"),
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    report = run_parity(ngp=args.ngp, seed=args.seed, kx=args.kx, tol=args.tol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "symbolic parity:",
        f"pass={report['pass']}",
        f"max_abs_residual={report['max_abs_residual']:.3e}",
        f"output={args.output}",
    )
    if not report["pass"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
