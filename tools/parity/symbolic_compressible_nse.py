#!/usr/bin/env python3
"""SymPy templates for compressible-LNSE and curvilinear metric parity."""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import sympy as sp

_ASSIGN_RE = re.compile(r"c_(A0|Ar|Az|Arr|Azz|Arz|B)_([1-5])([1-5])$")
_BLOCK_ORDER = ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cartesian_symbol_table() -> dict[str, sp.Symbol]:
    names = [
        "U",
        "V",
        "W",
        "RHO",
        "T",
        "MU",
        "P",
        "kx",
        "Re",
        "Ma",
        "Pr",
        "kappa",
        "dUdx",
        "dVdx",
        "dWdx",
        "dRHOdx",
        "dTdx",
        "dPdx",
        "dUdy",
        "dVdy",
        "dWdy",
        "dRHOdy",
        "dTdy",
        "dPdy",
        "d2Udx2",
        "d2Vdx2",
        "d2Wdx2",
        "d2Udy2",
        "d2Vdy2",
        "d2Wdy2",
        "d2Tdx2",
        "d2Tdy2",
        "d2Udxdy",
        "d2Vdxdy",
        "d2Wdxdy",
        "dmudT",
        "d2mudT2",
    ]
    return {name: sp.Symbol(name, complex=True) for name in names}


def cylindrical_symbol_table() -> dict[str, sp.Symbol]:
    names = [
        "U",
        "V",
        "W",
        "R",
        "RHO",
        "T",
        "MU",
        "m",
        "Re",
        "Ma",
        "Pr",
        "kappa",
        "dUdr",
        "dVdr",
        "dWdr",
        "dRHOdr",
        "dTdr",
        "dUdz",
        "dVdz",
        "dWdz",
        "dRHOdz",
        "dTdz",
        "d2Udr2",
        "d2Vdr2",
        "d2Wdr2",
        "d2Tdr2",
        "d2Udz2",
        "d2Vdz2",
        "d2Wdz2",
        "d2Tdz2",
        "d2Udrz",
        "d2Vdrz",
        "d2Wdrz",
        "dmudT",
        "d2mudT2",
    ]
    return {name: sp.Symbol(name, complex=True) for name in names}


def curvilinear_symbol_table() -> dict[str, sp.Symbol]:
    names = [
        "a",
        "b",
        "c",
        "d",
        "dJ11_xi",
        "dJ12_xi",
        "dJ21_xi",
        "dJ22_xi",
        "dJ11_eta",
        "dJ12_eta",
        "dJ21_eta",
        "dJ22_eta",
        "x2p",
        "x2pp",
        "y2p",
        "y2pp",
    ]
    return {name: sp.Symbol(name, complex=True) for name in names}


def continuity_symbol_table() -> dict[str, sp.Symbol]:
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


def _coeff_source_path() -> Path:
    return Path(__file__).resolve().parents[2] / "linstab2d" / "operators" / "coefficients.py"


@lru_cache(maxsize=4)
def _assignment_expressions(function_name: str) -> dict[tuple[str, int, int], str]:
    source = _coeff_source_path().read_text(encoding="utf-8")
    tree = ast.parse(source)

    fn_node: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            fn_node = node
            break
    if fn_node is None:
        raise RuntimeError(f"Could not locate {function_name} in coefficients.py")

    out: dict[tuple[str, int, int], str] = {}
    for stmt in fn_node.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        m = _ASSIGN_RE.match(target.id)
        if m is None:
            continue
        block = m.group(1)
        i = int(m.group(2)) - 1
        j = int(m.group(3)) - 1
        out[(block, i, j)] = ast.unparse(stmt.value)

    expected_terms = 7 * 5 * 5
    if len(out) != expected_terms:
        raise RuntimeError(
            f"Expected {expected_terms} assignments in {function_name}, found {len(out)}."
        )
    return out


def _cartesian_eval_namespace(symbols: dict[str, sp.Symbol]) -> dict[str, Any]:
    U = symbols["U"]
    V = symbols["V"]
    W = symbols["W"]
    RHO = symbols["RHO"]
    T = symbols["T"]
    MU = symbols["MU"]
    P = symbols["P"]
    kx = symbols["kx"]

    ns: dict[str, Any] = {
        "sp": sp,
        "U": U,
        "V": V,
        "W": W,
        "RHO": RHO,
        "T": T,
        "MU": MU,
        "P": P,
        "kx": kx,
        "Re": symbols["Re"],
        "Ma": symbols["Ma"],
        "Pr": symbols["Pr"],
        "kappa": symbols["kappa"],
        "Uc": -W,
        "Vc": V,
        "Wc": U,
        "kz": -kx,
        "dUdr": -symbols["dWdy"],
        "dUdz": -symbols["dWdx"],
        "d2Udr2": -symbols["d2Wdy2"],
        "d2Udz2": -symbols["d2Wdx2"],
        "d2Udrz": -symbols["d2Wdxdy"],
        "dVdr": symbols["dVdy"],
        "dVdz": symbols["dVdx"],
        "d2Vdr2": symbols["d2Vdy2"],
        "d2Vdz2": symbols["d2Vdx2"],
        "d2Vdrz": symbols["d2Vdxdy"],
        "dWdr": symbols["dUdy"],
        "dWdz": symbols["dUdx"],
        "d2Wdr2": symbols["d2Udy2"],
        "d2Wdz2": symbols["d2Udx2"],
        "d2Wdrz": symbols["d2Udxdy"],
        "dRHOdr": symbols["dRHOdy"],
        "dRHOdz": symbols["dRHOdx"],
        "dTdr": symbols["dTdy"],
        "dTdz": symbols["dTdx"],
        "d2Tdr2": symbols["d2Tdy2"],
        "d2Tdz2": symbols["d2Tdx2"],
        "dPdr": symbols["dPdy"],
        "dPdz": symbols["dPdx"],
        "dMUdT": symbols["dmudT"],
        "d2MUdT2": symbols["d2mudT2"],
        "Z": sp.Integer(0),
        "I": sp.Integer(1),
    }
    return ns


def _cylindrical_eval_namespace(symbols: dict[str, sp.Symbol]) -> dict[str, Any]:
    kappa = symbols["kappa"]
    Re = symbols["Re"]
    Pr = symbols["Pr"]
    Ma = symbols["Ma"]

    c1 = (kappa - 1) * Re * Pr * Ma**2
    c2 = kappa * Ma**2
    cv = 1 / (kappa * (kappa - 1) * Ma**2)

    ns: dict[str, Any] = {
        "sp": sp,
        "c1": c1,
        "c2": c2,
        "cv": cv,
        "Z": sp.Integer(0),
        "I": sp.Integer(1),
    }
    ns.update(symbols)
    return ns


def _evaluate_coeff_blocks(
    function_name: str,
    symbols: dict[str, sp.Symbol],
    namespace_builder: Callable[[dict[str, sp.Symbol]], dict[str, Any]],
) -> dict[str, dict[tuple[int, int], sp.Expr]]:
    raw = _assignment_expressions(function_name)
    ns = namespace_builder(symbols)
    blocks = {name: {} for name in ("A0", "Ar", "Az", "Arr", "Azz", "Arz", "B")}

    for (block, i, j), expr_src in raw.items():
        expr = sp.sympify(eval(expr_src, {"__builtins__": {}}, ns))
        blocks[block][(i, j)] = expr

    return blocks


def _map_coeff_to_physical(
    blocks: dict[str, dict[tuple[int, int], sp.Expr]],
    *,
    sign_flip_non_b: bool,
    c2p_idx: list[int],
    c2p_sgn: list[int],
) -> dict[str, dict[tuple[int, int], sp.Expr]]:
    signed: dict[str, dict[tuple[int, int], sp.Expr]] = {}
    for name, entries in blocks.items():
        if name == "B" or not sign_flip_non_b:
            signed[name] = dict(entries)
        else:
            signed[name] = {(i, j): -expr for (i, j), expr in entries.items()}

    permuted: dict[str, dict[tuple[int, int], sp.Expr]] = {}
    for name, entries in signed.items():
        mapped: dict[tuple[int, int], sp.Expr] = {}
        for ic in range(5):
            ip = c2p_idx[ic]
            for jc in range(5):
                jp = c2p_idx[jc]
                mapped[(ip, jp)] = c2p_sgn[ic] * c2p_sgn[jc] * entries[(ic, jc)]
        permuted[name] = mapped

    return {
        "A0": permuted["A0"],
        "Ax": permuted["Az"],
        "Ay": permuted["Ar"],
        "Axx": permuted["Azz"],
        "Ayy": permuted["Arr"],
        "Axy": permuted["Arz"],
        "B": permuted["B"],
    }


def _flatten_blocks(blocks: dict[str, dict[tuple[int, int], sp.Expr]]) -> dict[str, sp.Expr]:
    out: dict[str, sp.Expr] = {}
    for block in _BLOCK_ORDER:
        for i in range(5):
            for j in range(5):
                out[f"{block}_{i}{j}"] = blocks[block][(i, j)]
    return out


def cartesian_operator_coefficients(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    coeff_blocks = _evaluate_coeff_blocks(
        "compute_cartesian_coefficients",
        symbols,
        _cartesian_eval_namespace,
    )
    phys_blocks = _map_coeff_to_physical(
        coeff_blocks,
        sign_flip_non_b=True,
        c2p_idx=[0, 3, 2, 1, 4],
        c2p_sgn=[1, -1, 1, 1, 1],
    )
    return _flatten_blocks(phys_blocks)


def cylindrical_operator_coefficients(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    coeff_blocks = _evaluate_coeff_blocks(
        "compute_cylindrical_coefficients",
        symbols,
        _cylindrical_eval_namespace,
    )
    phys_blocks = _map_coeff_to_physical(
        coeff_blocks,
        sign_flip_non_b=False,
        c2p_idx=[0, 2, 3, 1, 4],
        c2p_sgn=[1, 1, 1, 1, 1],
    )
    return _flatten_blocks(phys_blocks)


def curvilinear_operator_coefficients(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    """Return symbolic chain-rule coefficients for curvilinear transforms."""
    a = symbols["a"]
    b = symbols["b"]
    c = symbols["c"]
    d = symbols["d"]

    dJ_xi = sp.Matrix(
        [
            [symbols["dJ11_xi"], symbols["dJ12_xi"]],
            [symbols["dJ21_xi"], symbols["dJ22_xi"]],
        ]
    )
    dJ_eta = sp.Matrix(
        [
            [symbols["dJ11_eta"], symbols["dJ12_eta"]],
            [symbols["dJ21_eta"], symbols["dJ22_eta"]],
        ]
    )
    Ji = sp.Matrix([[a, b], [c, d]])

    dJi_xi = -Ji * dJ_xi * Ji
    dJi_eta = -Ji * dJ_eta * Ji

    q11_x = a * dJi_xi[0, 0] + b * dJi_eta[0, 0]
    q11_y = a * dJi_xi[0, 1] + b * dJi_eta[0, 1]
    q12_x = c * dJi_xi[0, 0] + d * dJi_eta[0, 0]
    q12_y = c * dJi_xi[0, 1] + d * dJi_eta[0, 1]
    q21_x = a * dJi_xi[1, 0] + b * dJi_eta[1, 0]
    q21_y = a * dJi_xi[1, 1] + b * dJi_eta[1, 1]
    q22_x = c * dJi_xi[1, 0] + d * dJi_eta[1, 0]
    q22_y = c * dJi_xi[1, 1] + d * dJi_eta[1, 1]

    out: dict[str, sp.Expr] = {
        # Full-2D first derivatives
        "full2d_Dx_Dx": a,
        "full2d_Dx_Dy": b,
        "full2d_Dy_Dx": c,
        "full2d_Dy_Dy": d,
        # Full-2D second derivatives: [D2x, Dxy, Dyx, D2y, Dx, Dy] coefficients
        "full2d_D2x_c11": a * a,
        "full2d_D2x_c12": a * b,
        "full2d_D2x_c21": b * a,
        "full2d_D2x_c22": b * b,
        "full2d_D2x_qx": q11_x,
        "full2d_D2x_qy": q11_y,
        "full2d_Dxy_c11": a * c,
        "full2d_Dxy_c12": c * b,
        "full2d_Dxy_c21": d * a,
        "full2d_Dxy_c22": d * b,
        "full2d_Dxy_qx": q12_x,
        "full2d_Dxy_qy": q12_y,
        "full2d_Dyx_c11": c * a,
        "full2d_Dyx_c12": c * d,
        "full2d_Dyx_c21": d * a,
        "full2d_Dyx_c22": d * d,
        "full2d_Dyx_qx": q21_x,
        "full2d_Dyx_qy": q21_y,
        "full2d_D2y_c11": c * c,
        "full2d_D2y_c12": c * d,
        "full2d_D2y_c21": d * c,
        "full2d_D2y_c22": d * d,
        "full2d_D2y_qx": q22_x,
        "full2d_D2y_qy": q22_y,
    }

    x2p = symbols["x2p"]
    x2pp = symbols["x2pp"]
    y2p = symbols["y2p"]
    y2pp = symbols["y2pp"]

    out.update(
        {
            # X-only branch
            "x_only_Dx_Dx": 1 / x2p,
            "x_only_D2x_D2x": 1 / x2p**2,
            "x_only_D2x_Dx": -(x2pp / x2p**3),
            # Y-only branch
            "y_only_Dy_Dy": 1 / y2p,
            "y_only_D2y_D2y": 1 / y2p**2,
            "y_only_D2y_Dy": -(y2pp / y2p**3),
        }
    )

    return out


def _find_unresolved(
    exprs: dict[str, sp.Expr],
    allowed_symbols: dict[str, sp.Symbol],
) -> dict[str, list[str]]:
    allowed = set(allowed_symbols.values())
    unresolved: dict[str, list[str]] = {}
    for key, expr in exprs.items():
        unknown = sorted(str(sym) for sym in expr.free_symbols if sym not in allowed)
        if unknown:
            unresolved[key] = unknown
    return unresolved


def _has_nonfinite(expr: sp.Expr) -> bool:
    return bool(expr.has(sp.oo) or expr.has(-sp.oo) or expr.has(sp.zoo) or expr.has(sp.nan))


def build_symbolic_report(mode: str = "all") -> dict[str, Any]:
    tasks: list[
        tuple[
            str,
            Callable[[], dict[str, sp.Symbol]],
            Callable[[dict[str, sp.Symbol]], dict[str, sp.Expr]],
        ]
    ] = []

    if mode in ("cartesian", "both", "all"):
        tasks.append(("cartesian_full_5x5", cartesian_symbol_table, cartesian_operator_coefficients))
    if mode in ("cylindrical", "both", "all"):
        tasks.append(("cylindrical_full_5x5", cylindrical_symbol_table, cylindrical_operator_coefficients))
    if mode in ("curvilinear", "all"):
        tasks.append(("curvilinear_metric_terms", curvilinear_symbol_table, curvilinear_operator_coefficients))

    model_reports: dict[str, Any] = {}
    combined_unresolved: dict[str, list[str]] = {}
    combined_nonfinite: list[str] = []
    total_terms = 0

    for scope, sym_builder, coeff_builder in tasks:
        syms = sym_builder()
        coeffs = coeff_builder(syms)
        unresolved = _find_unresolved(coeffs, syms)
        nonfinite = [k for k, expr in coeffs.items() if _has_nonfinite(expr)]
        total_terms += len(coeffs)

        model_reports[scope] = {
            "n_terms": len(coeffs),
            "n_unresolved_terms": len(unresolved),
            "n_nonfinite_terms": len(nonfinite),
            "pass": bool(not unresolved and not nonfinite),
        }

        for key, value in unresolved.items():
            combined_unresolved[f"{scope}:{key}"] = value
        combined_nonfinite.extend(f"{scope}:{key}" for key in nonfinite)

    passed = bool(not combined_unresolved and not combined_nonfinite)
    return {
        "generated_at_utc": _now_utc(),
        "mode": mode,
        "scope": "full_symbolic_coverage" if mode in ("all", "both") else mode,
        "n_terms_total": total_terms,
        "models": model_reports,
        "n_unresolved_terms": len(combined_unresolved),
        "n_nonfinite_terms": len(combined_nonfinite),
        "unresolved": combined_unresolved,
        "nonfinite": combined_nonfinite,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cartesian", "cylindrical", "curvilinear", "both", "all"),
        default="all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/parity/output/operator_symbolic_templates.json"),
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    report = build_symbolic_report(mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "symbolic templates:",
        f"mode={report['mode']}",
        f"pass={report['pass']}",
        f"n_terms={report['n_terms_total']}",
        f"unresolved={report['n_unresolved_terms']}",
        f"nonfinite={report['n_nonfinite_terms']}",
        f"output={args.output}",
    )
    if not report["pass"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
