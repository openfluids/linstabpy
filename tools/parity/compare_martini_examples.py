#!/usr/bin/env python3
"""Compare extracted Martini MATLAB bundles against shipped Python examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linstab2d.examples._helpers import load_jsonc
from tools.parity.matlab_hdf5 import compound_to_complex, sanitize_key

DEFAULT_CONFIG = ROOT / "tools" / "parity" / "martini_cases.jsonc"
DEFAULT_NPZ_DIR = ROOT / "tools" / "parity" / "npz"
DEFAULT_REPORT = ROOT / "tools" / "parity" / "reports" / "martini_parity_latest.json"
DEFAULT_TABLE = ROOT / "tools" / "parity" / "reports" / "martini_parity_table.csv"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle = False) as data:
        return {key: data[key] for key in data.files}


def _npz_get(data: dict[str, np.ndarray], logical_key: str) -> np.ndarray:
    key = sanitize_key(logical_key)
    if key not in data:
        raise KeyError(f"Missing key '{logical_key}' ({key})")
    return data[key]


def _bool_scalar(values: np.ndarray) -> bool:
    return bool(np.asarray(values).reshape(-1)[0])


def _float_scalar(values: np.ndarray) -> float:
    return float(np.asarray(values).reshape(-1)[0])


def _complex_vector(values: np.ndarray) -> np.ndarray:
    return np.asarray(compound_to_complex(values), dtype=complex).reshape(-1)


def _best_complex_scale(reference: np.ndarray, candidate: np.ndarray) -> complex:
    denom = np.vdot(candidate, candidate)
    if denom == 0:
        return complex(1.0)
    return np.vdot(candidate, reference) / denom


def _rel_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(reference)), 1e-30)
    return float(np.linalg.norm(reference - candidate) / denom)


def _mode_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(compound_to_complex(values), dtype=complex)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mode matrix, got shape {arr.shape}")
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 256:
        return arr.T
    return arr


def _spectrum_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, float, float]:
    cost = np.abs(reference[:, None] - candidate[None, :])
    ridx, cidx = linear_sum_assignment(cost)
    abs_err = cost[ridx, cidx]
    rel_err = abs_err / np.maximum(np.abs(reference[ridx]), 1e-30)
    return cidx, float(np.max(abs_err)), float(np.max(rel_err))


def _top_by_imag(values: np.ndarray, n_compare: int) -> np.ndarray:
    order = np.argsort(np.imag(values))[::-1]
    return order[: min(len(order), n_compare)]


def _closest_to_target(values: np.ndarray, n_compare: int, target: complex | None) -> np.ndarray:
    if target is None:
        return _top_by_imag(values, n_compare)
    order = np.argsort(np.abs(values - target))
    return order[: min(len(order), n_compare)]


def compare_resolvent_case(
    *,
    gains_py: np.ndarray,
    responses_py: np.ndarray,
    forcings_py: np.ndarray,
    gains_mat: np.ndarray,
    responses_mat: np.ndarray,
    forcings_mat: np.ndarray,
    n_compare: int,
) -> dict[str, Any]:
    gains_py = np.asarray(gains_py, dtype=float).reshape(-1)
    gains_mat = np.asarray(gains_mat, dtype=float).reshape(-1)
    responses_py = _mode_matrix(responses_py)
    forcings_py = _mode_matrix(forcings_py)
    responses_mat = _mode_matrix(responses_mat)
    forcings_mat = _mode_matrix(forcings_mat)

    n = int(
        min(
            max(1, n_compare),
            gains_py.size,
            gains_mat.size,
            responses_py.shape[1],
            forcings_py.shape[1],
            responses_mat.shape[1],
            forcings_mat.shape[1],
        )
    )

    gains_py = gains_py[:n]
    gains_mat = gains_mat[:n]
    responses_py = responses_py[:, :n]
    forcings_py = forcings_py[:, :n]
    responses_mat = responses_mat[:, :n]
    forcings_mat = forcings_mat[:, :n]

    gain_cost = np.abs(gains_mat[:, None] - gains_py[None, :])
    ridx, cidx = linear_sum_assignment(gain_cost)
    gain_abs = gain_cost[ridx, cidx]
    gain_rel = gain_abs / np.maximum(np.abs(gains_mat[ridx]), 1e-30)

    per_mode: list[dict[str, float | int]] = []
    for mat_i, py_i in sorted(zip(ridx, cidx), key=lambda pair: int(pair[0])):
        f_ref = forcings_mat[:, mat_i]
        r_ref = responses_mat[:, mat_i]
        f_py = forcings_py[:, py_i] * _best_complex_scale(f_ref, forcings_py[:, py_i])
        r_py = responses_py[:, py_i] * _best_complex_scale(r_ref, responses_py[:, py_i])
        per_mode.append(
            {
                "matlab_mode_index_1based": int(mat_i) + 1,
                "python_mode_index_1based": int(py_i) + 1,
                "forcing_rel_l2": _rel_l2(f_ref, f_py),
                "response_rel_l2": _rel_l2(r_ref, r_py),
            }
        )

    return {
        "n_compare": n,
        "gains": {
            "max_abs_error": float(np.max(gain_abs)),
            "max_rel_error": float(np.max(gain_rel)),
        },
        "modes": {
            "max_forcing_rel_l2": float(max(row["forcing_rel_l2"] for row in per_mode)),
            "max_response_rel_l2": float(max(row["response_rel_l2"] for row in per_mode)),
            "per_mode": per_mode,
        },
    }


def compare_curvilinear_protocol_case(
    *,
    python_npz: dict[str, np.ndarray],
    matlab_results: dict[str, np.ndarray],
    matlab_logs: dict[str, np.ndarray],
) -> dict[str, Any]:
    df = np.asarray(_npz_get(matlab_results, "results_TestCurv/df"), dtype=float)
    df_an = np.asarray(_npz_get(matlab_results, "results_TestCurv/df_an"), dtype=float)

    matlab_linf = float(np.max(np.abs(df - df_an)))
    matlab_rel = float(np.linalg.norm(df - df_an) / max(np.linalg.norm(df_an), 1e-30))
    python_pass = _bool_scalar(python_npz["overall_pass"])

    branch_names = [str(name) for name in python_npz["branch_names"]]
    branch_rows = []
    for i, name in enumerate(branch_names):
        branch_rows.append(
            {
                "branch": name,
                "pass": bool(np.asarray(python_npz["pass_flags"], dtype=bool)[i]),
                "dfdx_inf": float(np.asarray(python_npz["dfdx_inf"], dtype=float)[i]),
                "dfdy_inf": float(np.asarray(python_npz["dfdy_inf"], dtype=float)[i]),
                "d2xy_inf": float(np.asarray(python_npz["d2xy_inf"], dtype=float)[i]),
                "d2yx_inf": float(np.asarray(python_npz["d2yx_inf"], dtype=float)[i]),
            }
        )

    timing_total = float(np.asarray(_npz_get(matlab_logs, "timing_total"), dtype=float).reshape(-1)[0])
    return {
        "python_protocol_pass": python_pass,
        "matlab_df_linf": matlab_linf,
        "matlab_df_rel_l2": matlab_rel,
        "matlab_timing_total_seconds": timing_total,
        "python_branches": branch_rows,
    }


def _compare_cavity_case(
    *,
    python_npz: dict[str, np.ndarray],
    matlab_direct: dict[str, np.ndarray],
    matlab_adjoint: dict[str, np.ndarray],
    n_compare: int,
    omega_target: complex | None,
) -> dict[str, Any]:
    omega_py = _complex_vector(python_npz["omega"])
    direct_modes_py = _mode_matrix(python_npz["direct_modes"])
    omega_adj_py_direct = np.conj(-1j * _complex_vector(python_npz["omega_adj"]))
    adjoint_modes_py = _mode_matrix(python_npz["adjoint_modes"])

    omega_mat = _complex_vector(_npz_get(matlab_direct, "results_Cavity_direct/omega"))
    direct_modes_mat = _mode_matrix(_npz_get(matlab_direct, "results_Cavity_direct/V"))
    omega_adj_mat_direct = np.conj(_complex_vector(_npz_get(matlab_adjoint, "results_Cavity_adjoint/omegaAdj")))
    adjoint_modes_mat = _mode_matrix(_npz_get(matlab_adjoint, "results_Cavity_adjoint/VAdj"))

    direct_sel_py = _closest_to_target(omega_py, n_compare, omega_target)
    direct_sel_mat = _closest_to_target(omega_mat, n_compare, omega_target)
    direct_match_idx, direct_max_abs, direct_max_rel = _spectrum_metrics(
        omega_mat[direct_sel_mat],
        omega_py[direct_sel_py],
    )

    direct_rows = []
    for mat_i, py_local in enumerate(direct_match_idx):
        py_i = int(direct_sel_py[py_local])
        ref = direct_modes_mat[:, int(direct_sel_mat[mat_i])]
        cand = direct_modes_py[:, py_i]
        cand = cand * _best_complex_scale(ref, cand)
        direct_rows.append(
            {
                "matlab_mode_index_1based": int(direct_sel_mat[mat_i]) + 1,
                "python_mode_index_1based": py_i + 1,
                "rel_l2": _rel_l2(ref, cand),
            }
        )

    adj_sel_py = _closest_to_target(omega_adj_py_direct, n_compare, omega_target)
    adj_sel_mat = _closest_to_target(omega_adj_mat_direct, n_compare, omega_target)
    adj_match_idx, adj_max_abs, adj_max_rel = _spectrum_metrics(
        omega_adj_mat_direct[adj_sel_mat],
        omega_adj_py_direct[adj_sel_py],
    )

    adj_rows = []
    for mat_i, py_local in enumerate(adj_match_idx):
        py_i = int(adj_sel_py[py_local])
        ref = adjoint_modes_mat[:, int(adj_sel_mat[mat_i])]
        cand = adjoint_modes_py[:, py_i]
        cand = cand * _best_complex_scale(ref, cand)
        adj_rows.append(
            {
                "matlab_mode_index_1based": int(adj_sel_mat[mat_i]) + 1,
                "python_mode_index_1based": py_i + 1,
                "rel_l2": _rel_l2(ref, cand),
            }
        )

    return {
        "direct": {
            "n_compare": len(direct_rows),
            "spectrum_max_abs_error": direct_max_abs,
            "spectrum_max_rel_error": direct_max_rel,
            "mode_max_rel_l2": float(max(row["rel_l2"] for row in direct_rows)),
            "per_mode": direct_rows,
        },
        "adjoint": {
            "n_compare": len(adj_rows),
            "spectrum_max_abs_error": adj_max_abs,
            "spectrum_max_rel_error": adj_max_rel,
            "mode_max_rel_l2": float(max(row["rel_l2"] for row in adj_rows)),
            "per_mode": adj_rows,
        },
    }


def _compare_case(case: dict[str, Any], npz_dir: Path) -> dict[str, Any]:
    status = "HARD_FAIL"
    note = str(case.get("note", ""))
    comparison_kind = str(case.get("comparison_kind", ""))
    base = {
        "case_id": str(case.get("case_id", "unknown")),
        "label": str(case.get("label", "")),
        "comparison_kind": comparison_kind,
        "comparison_scope": str(case.get("comparison_scope", "")),
        "status": status,
        "note": note,
    }

    if comparison_kind == "unavailable":
        base["status"] = "UNAVAILABLE"
        return base

    python_npz_path = ROOT / str(case["python_npz"])
    if not python_npz_path.exists():
        base["note"] = f"Missing Python artifact: {python_npz_path}"
        return base
    python_npz = _load_npz_dict(python_npz_path)
    base["python_npz"] = str(python_npz_path.relative_to(ROOT))

    try:
        if comparison_kind == "curvilinear_protocol":
            matlab_results_path = npz_dir / f"{case['matlab_results_stem']}.npz"
            matlab_logs_path = npz_dir / f"{case['matlab_logs_stem']}.npz"
            if not matlab_results_path.exists() or not matlab_logs_path.exists():
                raise FileNotFoundError(f"Missing MATLAB NPZ bundle(s) for {base['case_id']}")
            report = compare_curvilinear_protocol_case(
                python_npz=python_npz,
                matlab_results=_load_npz_dict(matlab_results_path),
                matlab_logs=_load_npz_dict(matlab_logs_path),
            )
            base["matlab_npz"] = [
                str(matlab_results_path.relative_to(ROOT)),
                str(matlab_logs_path.relative_to(ROOT)),
            ]
            base["status"] = "PASS" if report["python_protocol_pass"] else "SOFT_FAIL"
            base["metrics"] = report
            return base

        if comparison_kind == "resolvent":
            matlab_results_path = npz_dir / f"{case['matlab_results_stem']}.npz"
            if not matlab_results_path.exists():
                raise FileNotFoundError(f"Missing MATLAB NPZ bundle: {matlab_results_path}")
            matlab_npz = _load_npz_dict(matlab_results_path)
            report = compare_resolvent_case(
                gains_py=python_npz["gains"],
                responses_py=python_npz["responses"],
                forcings_py=python_npz["forcings"],
                gains_mat=_npz_get(matlab_npz, str(case["matlab_gains_key"])),
                responses_mat=_npz_get(matlab_npz, str(case["matlab_responses_key"])),
                forcings_mat=_npz_get(matlab_npz, str(case["matlab_forcings_key"])),
                n_compare=int(case.get("n_compare", 3)),
            )
            tol = dict(case.get("tolerances", {}))
            passed = (
                report["gains"]["max_abs_error"] <= float(tol.get("gain_abs", np.inf))
                and report["gains"]["max_rel_error"] <= float(tol.get("gain_rel", np.inf))
                and report["modes"]["max_forcing_rel_l2"] <= float(tol.get("mode_rel", np.inf))
                and report["modes"]["max_response_rel_l2"] <= float(tol.get("mode_rel", np.inf))
            )
            base["matlab_npz"] = [str(matlab_results_path.relative_to(ROOT))]
            base["status"] = "PASS" if passed else "SOFT_FAIL"
            base["metrics"] = report
            return base

        if comparison_kind == "cavity":
            direct_path = npz_dir / f"{case['matlab_direct_stem']}.npz"
            adjoint_path = npz_dir / f"{case['matlab_adjoint_stem']}.npz"
            if not direct_path.exists() or not adjoint_path.exists():
                raise FileNotFoundError(f"Missing MATLAB NPZ bundle(s) for {base['case_id']}")
            report = _compare_cavity_case(
                python_npz=python_npz,
                matlab_direct=_load_npz_dict(direct_path),
                matlab_adjoint=_load_npz_dict(adjoint_path),
                n_compare=int(case.get("n_compare", 6)),
                omega_target=complex(str(case["omega_target"])) if "omega_target" in case else None,
            )
            tol = dict(case.get("tolerances", {}))
            passed = (
                report["direct"]["spectrum_max_abs_error"] <= float(tol.get("spectrum_abs", np.inf))
                and report["direct"]["spectrum_max_rel_error"] <= float(tol.get("spectrum_rel", np.inf))
                and report["direct"]["mode_max_rel_l2"] <= float(tol.get("mode_rel", np.inf))
                and report["adjoint"]["spectrum_max_abs_error"] <= float(tol.get("spectrum_abs", np.inf))
                and report["adjoint"]["spectrum_max_rel_error"] <= float(tol.get("spectrum_rel", np.inf))
                and report["adjoint"]["mode_max_rel_l2"] <= float(tol.get("mode_rel", np.inf))
            )
            base["matlab_npz"] = [str(direct_path.relative_to(ROOT)), str(adjoint_path.relative_to(ROOT))]
            base["status"] = "PASS" if passed else "SOFT_FAIL"
            base["metrics"] = report
            return base

        base["note"] = f"Unsupported comparison_kind={comparison_kind!r}"
        return base
    except Exception as exc:
        base["note"] = str(exc)
        return base


def _csv_row(case: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(case.get("metrics", {}))
    row = {
        "case_id": case.get("case_id", ""),
        "label": case.get("label", ""),
        "comparison_scope": case.get("comparison_scope", ""),
        "status": case.get("status", ""),
        "gain_max_abs": "",
        "gain_max_rel": "",
        "forcing_max_rel_l2": "",
        "response_max_rel_l2": "",
        "direct_spec_max_abs": "",
        "direct_spec_max_rel": "",
        "direct_mode_max_rel_l2": "",
        "adjoint_spec_max_abs": "",
        "adjoint_spec_max_rel": "",
        "adjoint_mode_max_rel_l2": "",
        "matlab_df_linf": "",
        "matlab_df_rel_l2": "",
        "python_protocol_pass": "",
        "note": case.get("note", ""),
    }

    if "gains" in metrics:
        row["gain_max_abs"] = metrics["gains"]["max_abs_error"]
        row["gain_max_rel"] = metrics["gains"]["max_rel_error"]
        row["forcing_max_rel_l2"] = metrics["modes"]["max_forcing_rel_l2"]
        row["response_max_rel_l2"] = metrics["modes"]["max_response_rel_l2"]
    if "direct" in metrics:
        row["direct_spec_max_abs"] = metrics["direct"]["spectrum_max_abs_error"]
        row["direct_spec_max_rel"] = metrics["direct"]["spectrum_max_rel_error"]
        row["direct_mode_max_rel_l2"] = metrics["direct"]["mode_max_rel_l2"]
        row["adjoint_spec_max_abs"] = metrics["adjoint"]["spectrum_max_abs_error"]
        row["adjoint_spec_max_rel"] = metrics["adjoint"]["spectrum_max_rel_error"]
        row["adjoint_mode_max_rel_l2"] = metrics["adjoint"]["mode_max_rel_l2"]
    if "matlab_df_linf" in metrics:
        row["matlab_df_linf"] = metrics["matlab_df_linf"]
        row["matlab_df_rel_l2"] = metrics["matlab_df_rel_l2"]
        row["python_protocol_pass"] = metrics["python_protocol_pass"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()

    cfg = load_jsonc(args.config)
    npz_dir = args.npz_dir.resolve()
    report_path = args.report.resolve()
    table_path = args.table.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [_compare_case(case, npz_dir) for case in cfg.get("cases", [])]
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["status"]] = counts.get(case["status"], 0) + 1

    report = {
        "generated_at_utc": _now_utc(),
        "config": str(args.config.resolve()),
        "npz_dir": str(npz_dir),
        "counts": counts,
        "cases": cases,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [_csv_row(case) for case in cases]
    fieldnames = list(rows[0].keys()) if rows else []
    with table_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Martini parity comparison:",
        f"counts={counts}",
        f"report={report_path}",
        f"table={table_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
