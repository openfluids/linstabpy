"""Parity checks for Case 1 case spectra and mode fields."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent.parent
PY_NPZ = ROOT / "results/case1/case1.npz"
MAT_SPEC = ROOT / "results/case1/MATLAB_results/spectrum.mat"
MAT_MODE7 = ROOT / "results/case1/MATLAB_results/eigenvector_7.mat"


def _require_inputs() -> None:
    missing = [p for p in (PY_NPZ, MAT_SPEC, MAT_MODE7) if not p.exists()]
    if missing:
        msg = ", ".join(str(p) for p in missing)
        pytest.skip(f"Case 1 parity inputs missing: {msg}. Run case1 first.")


def _diag_or_flat(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return np.diag(arr).astype(complex)
    return arr.reshape(-1).astype(complex)


def _hungarian_errors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cost = np.abs(a[:, None] - b[None, :])
    r, c = linear_sum_assignment(cost)
    return cost[r, c]


def _phase_align(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    num = np.vdot(cand, ref)
    if num == 0:
        return cand
    return cand * np.exp(-1j * np.angle(num))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a)
    if denom == 0.0:
        return float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denom)


def test_case1_spectrum_parity() -> None:
    _require_inputs()
    py = np.load(PY_NPZ, allow_pickle = False)
    omega_py = np.asarray(py["omega"], dtype=complex).reshape(-1)
    omega_target = complex(np.asarray(py["omega_target"]).reshape(-1)[0])
    omega_mat = _diag_or_flat(loadmat(MAT_SPEC)["omega"])

    n_compare = int(os.getenv("LINSTAB2D_CASE1_SPEC_N", "100"))
    n = min(max(1, n_compare), omega_py.size, omega_mat.size)

    py_sel = omega_py[np.argsort(np.abs(omega_py - omega_target))[:n]]
    mat_sel = omega_mat[np.argsort(np.abs(omega_mat - omega_target))[:n]]

    abs_err = _hungarian_errors(py_sel, mat_sel)
    max_abs = float(np.max(abs_err))
    rms_abs = float(np.sqrt(np.mean(abs_err**2)))

    tol_max_abs = float(os.getenv("LINSTAB2D_CASE1_SPEC_TOL_MAX_ABS", "1.0"))
    tol_rms_abs = float(os.getenv("LINSTAB2D_CASE1_SPEC_TOL_RMS_ABS", "0.5"))

    assert max_abs <= tol_max_abs and rms_abs <= tol_rms_abs, (
        f"Spectrum mismatch: n={n}, max_abs={max_abs:.6e}, rms_abs={rms_abs:.6e}, "
        f"tols=({tol_max_abs:.6e}, {tol_rms_abs:.6e})"
    )


def test_case1_mode7_u_field_parity() -> None:
    _require_inputs()
    py = np.load(PY_NPZ, allow_pickle = False)
    omega_py = np.asarray(py["omega"], dtype=complex).reshape(-1)
    direct = np.asarray(py["direct_modes"], dtype=complex)
    idx_u = np.asarray(py["idx_u_j"], dtype=int)
    omega_mat = _diag_or_flat(loadmat(MAT_SPEC)["omega"])
    u_mat = np.asarray(loadmat(MAT_MODE7)["u"], dtype=complex).reshape(-1)

    mode_idx_mat = int(os.getenv("LINSTAB2D_CASE1_MODE_INDEX", "7")) - 1
    mode_idx_mat = max(0, min(mode_idx_mat, omega_mat.size - 1))
    omega_ref = omega_mat[mode_idx_mat]
    mode_idx_py = int(np.argmin(np.abs(omega_py - omega_ref)))

    u_py = direct[:, mode_idx_py][idx_u].reshape(-1)
    u_py = _phase_align(u_mat, u_py)

    u_mat_n = u_mat / max(np.linalg.norm(u_mat), 1e-30)
    u_py_n = u_py / max(np.linalg.norm(u_py), 1e-30)

    rel_err_complex = _rel_l2(u_mat_n, u_py_n)
    rel_err_real = _rel_l2(np.real(u_mat_n), np.real(u_py_n))
    corr_real = float(np.corrcoef(np.real(u_mat_n), np.real(u_py_n))[0, 1])

    tol_rel_complex = float(os.getenv("LINSTAB2D_CASE1_MODE_TOL_REL_COMPLEX", "0.8"))
    tol_rel_real = float(os.getenv("LINSTAB2D_CASE1_MODE_TOL_REL_REAL", "0.8"))
    tol_corr_real = float(os.getenv("LINSTAB2D_CASE1_MODE_TOL_CORR_REAL", "0.5"))

    assert (
        rel_err_complex <= tol_rel_complex
        and rel_err_real <= tol_rel_real
        and corr_real >= tol_corr_real
    ), (
        f"Mode-u mismatch: matlab_mode={mode_idx_mat+1}, python_mode={mode_idx_py+1}, "
        f"|Δω|={abs(omega_py[mode_idx_py]-omega_ref):.6e}, "
        f"rel_complex={rel_err_complex:.6e}, rel_real={rel_err_real:.6e}, corr_real={corr_real:.6f}, "
        f"tols=(rel_complex<={tol_rel_complex:.6e}, rel_real<={tol_rel_real:.6e}, corr_real>={tol_corr_real:.6f})"
    )
