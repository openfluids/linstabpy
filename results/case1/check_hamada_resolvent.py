#!/usr/bin/env python3
"""Compare Hamada Python resolvent outputs against MATLAB exports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._helpers import load_jsonc


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cfg_get(cfg: dict, path: str, default):
    node = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _as_complex(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.fields and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return np.asarray(arr["real"]) + 1j * np.asarray(arr["imag"])
    return np.asarray(arr, dtype=complex)


def _best_complex_scale(reference: np.ndarray, candidate: np.ndarray) -> complex:
    denom = np.vdot(candidate, candidate)
    if denom == 0:
        return complex(1.0)
    return np.vdot(candidate, reference) / denom


def _rel_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(reference)), 1e-30)
    return float(np.linalg.norm(reference - candidate) / denom)


def run(args: argparse.Namespace) -> dict:
    py = np.load(args.python_npz, allow_pickle = False)
    gains_py = np.asarray(py["gains"], dtype=float).reshape(-1)
    responses_py = np.asarray(py["responses"], dtype=complex)
    forcings_py = np.asarray(py["forcings"], dtype=complex)

    with h5py.File(args.matlab_resolvent, "r") as handle:
        gains_mat = np.asarray(handle["S"], dtype=float).reshape(-1)
        # MATLAB export shape is (n_modes, ndof) for U/V.
        responses_mat = _as_complex(np.asarray(handle["U"])).T
        forcings_mat = _as_complex(np.asarray(handle["V"])).T

    n_compare = int(min(max(1, args.n_compare), gains_py.size, gains_mat.size))
    gains_py = gains_py[:n_compare]
    gains_mat = gains_mat[:n_compare]
    responses_py = responses_py[:, :n_compare]
    forcings_py = forcings_py[:, :n_compare]
    responses_mat = responses_mat[:, :n_compare]
    forcings_mat = forcings_mat[:, :n_compare]

    gain_abs = np.abs(gains_py - gains_mat)
    gain_rel = gain_abs / np.maximum(np.abs(gains_mat), 1e-30)

    mode_rows: list[dict] = []
    for i in range(n_compare):
        f_ref = forcings_mat[:, i]
        r_ref = responses_mat[:, i]
        f_py = forcings_py[:, i] * _best_complex_scale(f_ref, forcings_py[:, i])
        r_py = responses_py[:, i] * _best_complex_scale(r_ref, responses_py[:, i])

        mode_rows.append(
            {
                "mode_index_1based": i + 1,
                "forcing_rel_l2": _rel_l2(f_ref, f_py),
                "response_rel_l2": _rel_l2(r_ref, r_py),
            }
        )

    forcing_rel_max = float(max(row["forcing_rel_l2"] for row in mode_rows))
    response_rel_max = float(max(row["response_rel_l2"] for row in mode_rows))
    gain_abs_max = float(np.max(gain_abs))
    gain_rel_max = float(np.max(gain_rel))

    passed = (
        gain_abs_max <= float(args.tol_gain_abs)
        and gain_rel_max <= float(args.tol_gain_rel)
        and forcing_rel_max <= float(args.tol_mode_rel)
        and response_rel_max <= float(args.tol_mode_rel)
    )

    return {
        "generated_at_utc": _now_utc(),
        "python_npz": str(args.python_npz),
        "matlab_resolvent": str(args.matlab_resolvent),
        "n_compare": n_compare,
        "tolerances": {
            "gain_abs": float(args.tol_gain_abs),
            "gain_rel": float(args.tol_gain_rel),
            "mode_rel": float(args.tol_mode_rel),
        },
        "gains": {
            "max_abs_error": gain_abs_max,
            "max_rel_error": gain_rel_max,
        },
        "modes": {
            "max_forcing_rel_l2": forcing_rel_max,
            "max_response_rel_l2": response_rel_max,
            "per_mode": mode_rows,
        },
        "pass": bool(passed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("example_6_hamada/parity_config.jsonc"),
        help="Optional parity JSONC config.",
    )
    parser.add_argument(
        "--python-npz",
        type=Path,
        default=Path("example_6_hamada.npz"),
    )
    parser.add_argument(
        "--matlab-resolvent",
        type=Path,
        default=Path("example_6_hamada/MATLAB_exports_resolvent/res_svd_final.mat"),
    )
    parser.add_argument("--n-compare", type=int, default=10)
    parser.add_argument("--tol-gain-abs", type=float, default=1e-8)
    parser.add_argument("--tol-gain-rel", type=float, default=1e-6)
    parser.add_argument("--tol-mode-rel", type=float, default=1e-5)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("output/hamada_resolvent_parity.json"),
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    if args.config.exists():
        cfg = load_jsonc(args.config)
        args.matlab_resolvent = Path(
            _cfg_get(
                cfg,
                "paths.matlab_exports_resolvent_dir",
                str(args.matlab_resolvent.parent),
            )
        ) / "res_svd_final.mat"

    report = run(args)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "Hamada resolvent parity:",
        f"n_compare={report['n_compare']}",
        f"gain_max_abs={report['gains']['max_abs_error']:.6e}",
        f"gain_max_rel={report['gains']['max_rel_error']:.6e}",
        f"forcing_max_rel={report['modes']['max_forcing_rel_l2']:.6e}",
        f"response_max_rel={report['modes']['max_response_rel_l2']:.6e}",
        f"pass={report['pass']}",
    )

    if (not report["pass"]) and (not args.allow_fail):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
