#!/usr/bin/env python3
"""SymPy symbolic templates for linearized compressible-NSE operator parity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sympy as sp


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def continuity_symbol_table() -> dict[str, sp.Symbol]:
    """Return canonical symbols used by the continuity operator template."""
    names = [
        "rho",
        "U",
        "V",
        "W",
        "kx",
        "dUdx",
        "dVdy",
        "dRHOdx",
        "dRHOdy",
    ]
    return {name: sp.Symbol(name, complex=True) for name in names}


def continuity_operator_coefficients(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    """Return symbolic coefficients for Cartesian continuity row (physical order).

    Physical perturbation order is ``[rho, u, v, w, T]``.
    Expressions match the sign convention used by
    ``compute_cartesian_coefficients`` after internal coordinate mapping.
    """
    rho = symbols["rho"]
    U = symbols["U"]
    V = symbols["V"]
    W = symbols["W"]
    kx = symbols["kx"]
    dUdx = symbols["dUdx"]
    dVdy = symbols["dVdy"]
    dRHOdx = symbols["dRHOdx"]
    dRHOdy = symbols["dRHOdy"]

    I = sp.I
    return {
        "A0_00": -(I * kx * W + dUdx + dVdy),
        "A0_01": -dRHOdx,
        "A0_02": -dRHOdy,
        "A0_03": -(I * kx * rho),
        "A0_04": sp.Integer(0),
        "Ax_00": -U,
        "Ax_01": -rho,
        "Ay_00": -V,
        "Ay_02": -rho,
        "B_00": I,
    }


def build_symbolic_report() -> dict[str, Any]:
    symbols = continuity_symbol_table()
    coeffs = continuity_operator_coefficients(symbols)
    return {
        "generated_at_utc": _now_utc(),
        "scope": "cartesian_continuity_row",
        "symbol_order": list(symbols.keys()),
        "expressions": {k: str(sp.simplify(v)) for k, v in coeffs.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/operator_symbolic_templates.json"),
        help="Output JSON with symbolic expressions",
    )
    args = parser.parse_args()

    report = build_symbolic_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"wrote symbolic template report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
