"""Parity checks for Hamada resolvent gains and leading modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


ROOT = Path(__file__).resolve().parents[1]
PY_NPZ = ROOT / "results/case1/case1.npz"
MAT_RES = ROOT / "results/case1/MATLAB_exports_resolvent/res_svd_final.mat"


def _require_inputs() -> None:
    missing = [p for p in (PY_NPZ, MAT_RES) if not p.exists()]
    if missing:
        msg = ", ".join(str(p) for p in missing)
        pytest.skip(f"Hamada resolvent parity inputs missing: {msg}. Run example_6_hamada first.")


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


def test_hamada_resolvent_gains_and_modes_match_matlab() -> None:
    _require_inputs()

    py = np.load(PY_NPZ, allow_pickle = False)
    gains_py = np.asarray(py["gains"], dtype=float).reshape(-1)
    responses_py = np.asarray(py["responses"], dtype=complex)
    forcings_py = np.asarray(py["forcings"], dtype=complex)

    with h5py.File(MAT_RES, "r") as handle:
        gains_mat = np.asarray(handle["S"], dtype=float).reshape(-1)
        responses_mat = _as_complex(np.asarray(handle["U"])).T
        forcings_mat = _as_complex(np.asarray(handle["V"])).T

    n = min(gains_py.size, gains_mat.size, responses_py.shape[1], forcings_py.shape[1])
    assert n >= 3, "Need at least 3 resolvent modes for parity check"

    gains_py = gains_py[:n]
    gains_mat = gains_mat[:n]
    responses_py = responses_py[:, :n]
    forcings_py = forcings_py[:, :n]
    responses_mat = responses_mat[:, :n]
    forcings_mat = forcings_mat[:, :n]

    gain_abs = np.abs(gains_py - gains_mat)
    gain_rel = gain_abs / np.maximum(np.abs(gains_mat), 1e-30)
    assert float(np.max(gain_abs)) < 1e-8
    assert float(np.max(gain_rel)) < 1e-6

    # Mode vectors are phase-indeterminate; align each mode independently.
    forcing_rel = []
    response_rel = []
    for i in range(n):
        f_ref = forcings_mat[:, i]
        r_ref = responses_mat[:, i]
        f_py = forcings_py[:, i] * _best_complex_scale(f_ref, forcings_py[:, i])
        r_py = responses_py[:, i] * _best_complex_scale(r_ref, responses_py[:, i])
        forcing_rel.append(_rel_l2(f_ref, f_py))
        response_rel.append(_rel_l2(r_ref, r_py))

    assert max(forcing_rel) < 1e-5
    assert max(response_rel) < 1e-5
