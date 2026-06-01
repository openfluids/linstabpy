# SPDX-License-Identifier: GPL-3.0-only
"""Genericized MATLAB-source parity audit for linstabpy coefficient functions.

Exposes ``audit(cyl_m_path, cart_m_path)`` which parses two MATLAB ``.m``
coefficient files and symbolically compares every (block, i, j) entry against
the live ``compute_cartesian_coefficients`` /
``compute_cylindrical_coefficients`` in ``coefficients.py``.

All parsing and comparison logic is carried over verbatim from the original
``_matlab_translation_audit.py`` scratch script.
"""

from __future__ import annotations

import random
import re
import signal
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import sympy as sp


ROOT = Path(__file__).resolve().parents[4]
PYTHON_COEFFS = ROOT / "src/linstabpy/operators/coefficients.py"

BLOCKS = ("A0", "Ar", "Az", "Arz", "Arr", "Azz", "B")
TOTAL = len(BLOCKS) * 5 * 5


@dataclass
class AuditResult:
    clean: list[str]
    machine_eps: list[str]
    structural: dict[str, sp.Expr]
    parse_failures: dict[str, str]


def _symbols(names: str) -> dict[str, sp.Symbol]:
    return {name: sp.Symbol(name) for name in names.replace(",", " ").split()}


def _base_namespace(*, cylindrical: bool) -> dict[str, object]:
    if cylindrical:
        names = """
            R U V W RHO T MU m Re Ma Pr kappa
            dUdr dVdr dWdr dRHOdr dTdr
            dUdz dVdz dWdz dRHOdz dTdz
            d2Udr2 d2Vdr2 d2Wdr2 d2Tdr2
            d2Udz2 d2Vdz2 d2Wdz2 d2Tdz2
            d2Udrz d2Vdrz d2Wdrz
            dmudT d2mudT2
        """
    else:
        names = """
            U V W RHO T MU P kz Re Ma Pr kappa
            dUdr dVdr dWdr dRHOdr dTdr dPdr
            dUdz dVdz dWdz dRHOdz dTdz dPdz
            d2Udr2 d2Vdr2 d2Wdr2 d2Tdr2
            d2Udz2 d2Vdz2 d2Wdz2 d2Tdz2
            d2Udrz d2Vdrz d2Wdrz
            dmudT d2mudT2 dMUdT d2MUdT2
        """

    ns: dict[str, object] = {"sp": sp, "Z": sp.Integer(0), "I": sp.Integer(1)}
    ns.update(_symbols(names))
    ns["c1"] = (ns["kappa"] - 1) * ns["Re"] * ns["Pr"] * ns["Ma"] ** 2
    ns["c2"] = ns["kappa"] * ns["Ma"] ** 2
    ns["cv"] = 1 / (ns["kappa"] * (ns["kappa"] - 1) * ns["Ma"] ** 2)
    ns["C"] = sp.Integer(0)
    ns["dMUdT"] = ns["dmudT"]
    ns["d2MUdT2"] = ns["d2mudT2"]
    ns["Uc"] = ns["U"]
    ns["Vc"] = ns["V"]
    ns["Wc"] = ns["W"]
    return ns


def _matlab_to_python(expr: str) -> str:
    out = expr.strip()
    out = out.replace(".*", "*").replace("./", "/").replace(".^", "**")
    out = re.sub(r"\b1i\b", "sp.I", out)
    for src, dst in {
        "0.1e1": "1",
        "0.2e1": "2",
        "0.3e1": "3",
        "0.4e1": "4",
        "0.6e1": "6",
    }.items():
        out = out.replace(src, dst)
    return out


def _python_expr(expr: str) -> str:
    return re.sub(r"\b1j\b", "sp.I", textwrap.dedent(expr).strip())


def _eval_expr(expr: str, ns: dict[str, object]) -> sp.Expr:
    return sp.nsimplify(eval(f"({expr})", {"__builtins__": {}}, ns), rational=True)


@contextmanager
def _time_limit(seconds: int):
    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"symbolic simplification exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _simplify_diff(lhs: sp.Expr, rhs: sp.Expr) -> sp.Expr:
    raw = lhs - rhs
    if raw == 0:
        return sp.Integer(0)
    quick = sp.cancel(sp.together(raw))
    if quick == 0:
        return sp.Integer(0)
    with _time_limit(8):
        return sp.factor(sp.simplify(quick))


def _parse_matlab(path: Path, *, cylindrical: bool) -> tuple[dict[str, sp.Expr], dict[str, str]]:
    text = path.read_text()
    ns = _base_namespace(cylindrical=cylindrical)
    parsed: dict[str, sp.Expr] = {}
    failures: dict[str, str] = {}

    pattern = re.compile(r"\b(A0|Ar|Az|Arz|Arr|Azz|B)_(\d)(\d)\s*=\s*(.*?);", re.S)
    for match in pattern.finditer(text):
        block, row, col, raw = match.groups()
        key = f"{block}_{row}{col}"
        try:
            parsed[key] = _eval_expr(_matlab_to_python(raw), ns)
        except Exception as exc:  # noqa: BLE001 - this is an audit report.
            failures[key] = f"{type(exc).__name__}: {exc}"

    return parsed, failures


def _function_body(text: str, name: str) -> str:
    start = text.index(f"def {name}(")
    next_def = text.find("\ndef ", start + 1)
    return text[start:] if next_def == -1 else text[start:next_def]


def _split_call_args(arg_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    in_string: str | None = None
    escape = False
    for idx, char in enumerate(arg_text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(arg_text[start:idx].strip())
            start = idx + 1
    args.append(arg_text[start:].strip())
    return args


def _iter_set_coeff_calls(text: str) -> list[str]:
    calls: list[str] = []
    pos = 0
    token = "_set_coeff("
    while True:
        start = text.find(token, pos)
        if start == -1:
            return calls
        idx = start + len(token)
        depth = 1
        in_string: str | None = None
        escape = False
        while idx < len(text):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == in_string:
                    in_string = None
            elif char in {"'", '"'}:
                in_string = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    calls.append(text[start + len(token) : idx])
                    pos = idx + 1
                    break
            idx += 1


def _parse_python_function(
    source: str, function_name: str, *, cylindrical: bool
) -> tuple[dict[str, sp.Expr], dict[str, str]]:
    body = _function_body(source, function_name)
    ns = _base_namespace(cylindrical=cylindrical)
    parsed: dict[str, sp.Expr] = {}
    failures: dict[str, str] = {}

    for call in _iter_set_coeff_calls(body):
        key = f"<call {len(parsed) + len(failures) + 1}>"
        try:
            args = _split_call_args(call)
            block = args[0].strip().strip("\"'")
            row = int(args[1])
            col = int(args[2])
            key = f"{block}_{row}{col}"
            parsed[key] = _eval_expr(_python_expr(args[3]), ns)
        except Exception as exc:  # noqa: BLE001 - this is an audit report.
            failures[key] = f"{type(exc).__name__}: {exc}"

    return parsed, failures


def _cartesian_pressure_subs(ns: dict[str, object]) -> dict[sp.Symbol, sp.Expr]:
    return {
        ns["P"]: ns["RHO"] * ns["T"] / (ns["kappa"] * ns["Ma"] ** 2),
        ns["dPdr"]: (ns["T"] * ns["dRHOdr"] + ns["RHO"] * ns["dTdr"])
        / (ns["kappa"] * ns["Ma"] ** 2),
        ns["dPdz"]: (ns["T"] * ns["dRHOdz"] + ns["RHO"] * ns["dTdz"])
        / (ns["kappa"] * ns["Ma"] ** 2),
    }


def _numeric_probe(diff: sp.Expr, ns: dict[str, object]) -> bool:
    rng = random.Random(8675309)
    values: dict[sp.Symbol, float] = {}
    fixed = {"kappa": 1.4, "Pr": 0.72, "Ma": 0.8, "Re": 1000.0}
    for value in ns.values():
        if isinstance(value, sp.Symbol):
            values[value] = fixed.get(str(value), rng.uniform(1.0, 5.0))
    try:
        probed = complex(diff.evalf(30, subs=values))
    except Exception:
        return False
    return abs(probed) < 1e-8


def _audit(
    *,
    matlab_path: Path,
    python_source: str,
    python_function: str,
    cylindrical: bool,
) -> AuditResult:
    ns = _base_namespace(cylindrical=cylindrical)
    matlab, matlab_failures = _parse_matlab(matlab_path, cylindrical=cylindrical)
    python, python_failures = _parse_python_function(
        python_source, python_function, cylindrical=cylindrical
    )
    pressure_subs = {} if cylindrical else _cartesian_pressure_subs(ns)

    clean: list[str] = []
    machine_eps: list[str] = []
    structural: dict[str, sp.Expr] = {}
    failures = {**matlab_failures, **python_failures}

    for block in BLOCKS:
        for row in range(1, 6):
            for col in range(1, 6):
                key = f"{block}_{row}{col}"
                if key not in matlab:
                    failures.setdefault(key, "missing MATLAB assignment")
                    continue
                if key not in python:
                    failures.setdefault(key, "missing Python _set_coeff call")
                    continue

                lhs = matlab[key].subs(pressure_subs)
                rhs = python[key].subs(pressure_subs)
                try:
                    diff = _simplify_diff(lhs, rhs)
                    if diff == 0:
                        clean.append(key)
                    elif _numeric_probe(diff, ns):
                        machine_eps.append(key)
                    else:
                        structural[key] = diff
                except Exception as exc:  # noqa: BLE001 - continue the audit.
                    failures.setdefault(key, f"{type(exc).__name__}: {exc}")

    return AuditResult(clean, machine_eps, structural, failures)


def audit(
    cyl_m_path: Path,
    cart_m_path: Path,
    *,
    tol: float = 1e-9,  # noqa: ARG001 - reserved; comparison uses symbolic zero-test
) -> dict[str, AuditResult]:
    """Parse the two .m files and compare against live coefficients.py.

    Parameters
    ----------
    cyl_m_path:
        Path to the MATLAB cylindrical coefficient file
        (e.g. ``martini_examples/getCoeffsLaminar.m``).
    cart_m_path:
        Path to the MATLAB cartesian coefficient file
        (e.g. ``martini_examples/getCoeffsLaminar_Cartesian.m``).
    tol:
        Reserved for future use.  The actual comparison uses symbolic
        zero-testing, not a floating-point tolerance.

    Returns
    -------
    dict with keys ``"cylindrical"`` and ``"cartesian"``, each an
    :class:`AuditResult`.
    """
    python_source = PYTHON_COEFFS.read_text()
    return {
        "cylindrical": _audit(
            matlab_path=cyl_m_path,
            python_source=python_source,
            python_function="compute_cylindrical_coefficients",
            cylindrical=True,
        ),
        "cartesian": _audit(
            matlab_path=cart_m_path,
            python_source=python_source,
            python_function="compute_cartesian_coefficients",
            cylindrical=False,
        ),
    }
