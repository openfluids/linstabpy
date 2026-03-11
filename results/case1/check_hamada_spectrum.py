#!/usr/bin/env python3
"""Compare Hamada Python spectrum against MATLAB reference spectrum."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _diag_or_flat(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return np.diag(arr).astype(complex)
    return arr.reshape(-1).astype(complex)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-npz",
        type=Path,
        default=Path("example_6_hamada.npz"),
        help="Python output NPZ with 'omega' key",
    )
    parser.add_argument(
        "--matlab-spectrum",
        type=Path,
        default=Path("example_6_hamada/MATLAB_exports/spectrum.mat"),
        help="MATLAB spectrum .mat file containing 'omega'",
    )
    parser.add_argument("--n-compare", type=int, default=100, help="Number of modes to compare")
    parser.add_argument("--tol-abs", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--tol-rel", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("output/hamada_spectrum_parity.json"),
        help="Where to write parity summary JSON",
    )
    args = parser.parse_args()

    py = np.load(args.python_npz, allow_pickle = False)
    if "omega" not in py:
        raise KeyError(f"{args.python_npz} has no 'omega' key")
    omega_py_all = np.asarray(py["omega"], dtype=complex).reshape(-1)
    if omega_py_all.size == 0:
        raise ValueError("Python spectrum is empty; run stability in example_6_hamada first")

    omega_target = complex(np.asarray(py.get("omega_target", 0.0)).reshape(-1)[0])
    mat = loadmat(args.matlab_spectrum)
    if "omega" not in mat:
        raise KeyError(f"{args.matlab_spectrum} has no 'omega' key")
    omega_mat_all = _diag_or_flat(np.asarray(mat["omega"]))

    n = int(min(max(1, args.n_compare), omega_py_all.size, omega_mat_all.size))
    py_order = np.argsort(np.abs(omega_py_all - omega_target))
    mat_order = np.argsort(np.abs(omega_mat_all - omega_target))
    omega_py = omega_py_all[py_order][:n]
    omega_mat = omega_mat_all[mat_order][:n]

    cost = np.abs(omega_py[:, None] - omega_mat[None, :])
    ridx, cidx = linear_sum_assignment(cost)
    matched_abs = cost[ridx, cidx]
    matched_mat = omega_mat[cidx]

    rel = matched_abs / np.maximum(np.abs(matched_mat), 1e-14)
    max_abs = float(np.max(matched_abs))
    rms_abs = float(np.sqrt(np.mean(np.square(matched_abs))))
    max_rel = float(np.max(rel))

    passed = (max_abs <= float(args.tol_abs)) and (max_rel <= float(args.tol_rel))
    summary = {
        "generated_at_utc": _now_utc(),
        "python_npz": str(args.python_npz),
        "matlab_spectrum": str(args.matlab_spectrum),
        "n_compare": n,
        "omega_target": {"real": float(np.real(omega_target)), "imag": float(np.imag(omega_target))},
        "tol_abs": float(args.tol_abs),
        "tol_rel": float(args.tol_rel),
        "max_abs_error": max_abs,
        "rms_abs_error": rms_abs,
        "max_rel_error": max_rel,
        "pass": passed,
    }

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "Hamada spectrum parity:",
        f"n_compare={n}",
        f"max_abs_error={max_abs:.6e}",
        f"max_rel_error={max_rel:.6e}",
        f"pass={passed}",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
