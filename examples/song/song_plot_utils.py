# SPDX-License-Identifier: GPL-3.0-only
"""Shared Song reference I/O and LinStabPy comparison helpers (θ = 32°)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment

from examples.song.reference_paths import LENGTH_SCALE, SPECTRUM_MATCH_EPS_DEFAULT

_PRIMITIVES: tuple[tuple[int, str, str], ...] = (
    (0, "rho", r"\rho"),
    (1, "u", "u"),
    (2, "v", "v"),
    (3, "w", "w"),
    (4, "t", "T"),
)


def load_song_omega(path: Path, scale: float = LENGTH_SCALE) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected two columns in {path}")
    return np.asarray(scale * (arr[:, 0] + 1.0j * arr[:, 1]), dtype=complex).reshape(-1)


def load_song_leading_mode_npz(
    path: Path,
    *,
    scale: float = LENGTH_SCALE,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load Song lead eigenmode from the shipped ``reference/eigenmode`` NPZ.

    On-disk arrays are ``(nx, ny)`` in the Song export layout.
    Transpose to the LinStabPy mesh convention ``(ny, nx)`` before plotting
    or interpolation.
    """
    with np.load(path, allow_pickle=False) as z:
        x_scaled = np.asarray(z["x"], dtype=float).T / scale
        y_scaled = np.asarray(z["y"], dtype=float).T / scale
        primitives: dict[str, np.ndarray] = {
            "rho": np.asarray(z["rho"]).T,
            "u": np.asarray(z["u"]).T,
            "v": np.asarray(z["v"]).T,
            "w": np.asarray(z["w"]).T,
            "t": np.asarray(z["t"]).T,
        }
    return x_scaled, y_scaled, primitives


def song_ref_on_mesh(
    X_py: np.ndarray,
    Y_py: np.ndarray,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    F_ref: np.ndarray,
) -> np.ndarray:
    y_axis = Y_ref[:, 0]
    x_axis = X_ref[0, :]
    y_inc, x_inc = y_axis, x_axis
    F_work = np.asarray(F_ref)
    if y_axis[0] > y_axis[-1]:
        y_inc = y_axis[::-1]
        F_work = F_work[::-1, :]
    if x_axis[0] > x_axis[-1]:
        x_inc = x_axis[::-1]
        F_work = F_work[:, ::-1]
    pts = np.column_stack([Y_py.ravel(), X_py.ravel()])
    if np.iscomplexobj(F_work):
        itp_r = RegularGridInterpolator(
            (y_inc, x_inc), np.real(F_work), bounds_error=False, fill_value=np.nan
        )
        itp_i = RegularGridInterpolator(
            (y_inc, x_inc), np.imag(F_work), bounds_error=False, fill_value=np.nan
        )
        return (itp_r(pts) + 1.0j * itp_i(pts)).reshape(X_py.shape)
    itp = RegularGridInterpolator(
        (y_inc, x_inc), np.asarray(F_work, dtype=float), bounds_error=False, fill_value=np.nan
    )
    return itp(pts).reshape(X_py.shape)


def per_primitive_phase_align(
    prim_ref_on: dict[str, np.ndarray],
    prim_py_raw: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, complex]]:
    aligned: dict[str, np.ndarray] = {}
    phases: dict[str, complex] = {}
    for _, key, _ in _PRIMITIVES:
        ref = prim_ref_on[key].astype(np.complex128)
        py = np.asarray(prim_py_raw[key], dtype=complex)
        denom = np.vdot(py, py)
        alpha = np.vdot(py, ref) / denom if abs(denom) > 0.0 else 1.0 + 0.0j
        phases[key] = alpha
        aligned[key] = py * alpha
    return aligned, phases


def component_primitive(
    matrix: np.ndarray, ngp: int, component: int, mode_index: int, shape: tuple[int, int]
) -> np.ndarray:
    start = component * ngp
    stop = (component + 1) * ngp
    return np.asarray(matrix[start:stop, mode_index], dtype=complex).reshape(shape, order="F")


def max_omega_i_index(omega: np.ndarray) -> int:
    om = np.asarray(omega, dtype=complex).reshape(-1)
    return int(np.argmax(np.imag(om)))


def nearest_eigenvalue_index(omega: np.ndarray, target: complex) -> int:
    om = np.asarray(omega, dtype=complex).reshape(-1)
    return int(np.argmin(np.abs(om - target)))


def normalized_real(field: np.ndarray) -> np.ndarray:
    r = np.real(np.asarray(field, dtype=complex))
    peak = float(np.nanmax(np.abs(r)))
    if peak <= 0.0:
        return np.zeros_like(r)
    return r / peak


@dataclass(frozen=True)
class SpectrumClassification:
    matched_pairs: tuple[tuple[int, int, float], ...]
    ref_only: tuple[int, ...]
    py_only: tuple[int, ...]
    assignment_mean: float
    max_nearest: float


def classify_spectrum_matches(
    omega_ref: np.ndarray,
    omega_py: np.ndarray,
    *,
    match_eps: float = SPECTRUM_MATCH_EPS_DEFAULT,
) -> SpectrumClassification:
    ref = np.asarray(omega_ref, dtype=complex).reshape(-1)
    py = np.asarray(omega_py, dtype=complex).reshape(-1)
    cost = np.abs(ref[:, None] - py[None, :])
    ridx, cidx = linear_sum_assignment(cost)

    matched_pairs: list[tuple[int, int, float]] = []
    matched_ref: set[int] = set()
    matched_py: set[int] = set()
    for i, j in zip(ridx.tolist(), cidx.tolist()):
        dist = float(cost[i, j])
        if dist <= match_eps:
            matched_pairs.append((int(i), int(j), dist))
            matched_ref.add(int(i))
            matched_py.add(int(j))

    ref_only = tuple(i for i in range(ref.size) if i not in matched_ref)
    py_only = tuple(j for j in range(py.size) if j not in matched_py)
    assignment_mean = float(np.mean(cost[ridx, cidx]))
    nearest_ref = np.min(cost, axis=1)
    nearest_py = np.min(cost, axis=0)
    max_nearest = float(max(np.max(nearest_ref), np.max(nearest_py)))
    return SpectrumClassification(
        matched_pairs=tuple(matched_pairs),
        ref_only=ref_only,
        py_only=py_only,
        assignment_mean=assignment_mean,
        max_nearest=max_nearest,
    )


def hungarian_assignment(
    omega_ref: np.ndarray,
    omega_py: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cost = np.abs(
        np.asarray(omega_ref).reshape(-1)[:, None] - np.asarray(omega_py).reshape(-1)[None, :]
    )
    ridx, cidx = linear_sum_assignment(cost)
    return ridx, cidx, cost[ridx, cidx]
