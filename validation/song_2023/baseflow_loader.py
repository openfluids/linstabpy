from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from linstab2d.baseflow import BaseFlow
from linstab2d.baseflow import sutherland_air


def _binomial5_filter(arr: np.ndarray, axis: int) -> np.ndarray:
    """Apply a simple 5-point binomial smoothing filter along one axis."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array for filtering, got shape {arr.shape}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    pad = [(0, 0), (0, 0)]
    pad[axis] = (2, 2)
    padded = np.pad(arr, pad, mode="edge")
    if axis == 0:
        return (
            padded[:-4, :]
            + 4.0 * padded[1:-3, :]
            + 6.0 * padded[2:-2, :]
            + 4.0 * padded[3:-1, :]
            + padded[4:, :]
        ) / 16.0
    return (
        padded[:, :-4]
        + 4.0 * padded[:, 1:-3]
        + 6.0 * padded[:, 2:-2]
        + 4.0 * padded[:, 3:-1]
        + padded[:, 4:]
    ) / 16.0


def _as_four_numbers(values: Any, *, cast, name: str) -> tuple[Any, Any, Any, Any]:
    if values is None:
        values = [0, 0, 0, 0]
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{name} must be a list of four values [top, right, bottom, left]")
    return tuple(cast(v) for v in values)


def _generate_boundary_sponge(
    shape: tuple[int, int],
    *,
    points: tuple[int, int, int, int],
    amplitudes: tuple[float, float, float, float],
) -> np.ndarray:
    """Generate a simple boundary sponge using quadratic ramps.

    The four entries follow the Song helper ordering:
    [top, right, bottom, left].
    """
    ny, nx = shape
    sponge = np.zeros(shape, dtype=float)

    def _ramp(count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=float)
        return (np.arange(1, count + 1, dtype=float) / float(count)) ** 2

    top_n, right_n, bottom_n, left_n = points
    top_amp, right_amp, bottom_amp, left_amp = amplitudes

    if top_n > 0:
        ramp = top_amp * _ramp(top_n)
        sponge[-top_n:, :] += ramp[:, None]
    if bottom_n > 0:
        ramp = bottom_amp * _ramp(bottom_n)[::-1]
        sponge[:bottom_n, :] += ramp[:, None]
    if right_n > 0:
        ramp = right_amp * _ramp(right_n)
        sponge[:, -right_n:] += ramp[None, :]
    if left_n > 0:
        ramp = left_amp * _ramp(left_n)[::-1]
        sponge[:, :left_n] += ramp[None, :]

    return sponge


def _apply_preprocessing(
    X: np.ndarray,
    Y: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    sponge: np.ndarray,
    preprocess_cfg: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply optional Song baseflow preprocessing.

    This deliberately starts with the least invasive controls:
    - sponge scaling and simple boundary-sponge synthesis,
    - repeated explicit smoothing passes on selected fields.

    Shock-specific preprocessing is not implemented yet; enabling it raises a
    clear error instead of silently doing nothing.
    """
    info: dict[str, Any] = {
        "enabled": False,
        "applied": [],
        "filter": {"enabled": False, "passes": 0, "fields": []},
        "sponge": {"scale": 1.0, "boundary": {"enabled": False}},
    }

    if preprocess_cfg is None:
        info["summary"] = "disabled"
        return X, Y, rho, u, v, w, T, sponge, info
    if not isinstance(preprocess_cfg, dict):
        raise TypeError("baseflow_preprocess must be a JSON object when provided")

    enabled = bool(preprocess_cfg.get("enabled", False))
    info["enabled"] = enabled
    if not enabled:
        info["summary"] = "disabled"
        return X, Y, rho, u, v, w, T, sponge, info

    shock_cfg = preprocess_cfg.get("shock", {})
    if isinstance(shock_cfg, dict) and bool(shock_cfg.get("enabled", False)):
        raise NotImplementedError(
            "Song shock preprocessing is not implemented yet; use sponge/filter options first"
        )

    sponge_cfg = preprocess_cfg.get("sponge", {})
    if sponge_cfg is None:
        sponge_cfg = {}
    if not isinstance(sponge_cfg, dict):
        raise TypeError("baseflow_preprocess.sponge must be a JSON object when provided")

    sponge_scale = float(sponge_cfg.get("scale", 1.0))
    info["sponge"]["scale"] = sponge_scale
    if sponge_scale != 1.0:
        sponge = sponge * sponge_scale
        info["applied"].append(f"sponge_scale={sponge_scale:.6g}")

    boundary_cfg = sponge_cfg.get("boundary", {})
    if boundary_cfg is None:
        boundary_cfg = {}
    if not isinstance(boundary_cfg, dict):
        raise TypeError("baseflow_preprocess.sponge.boundary must be a JSON object when provided")

    boundary_enabled = bool(boundary_cfg.get("enabled", False))
    info["sponge"]["boundary"] = {"enabled": boundary_enabled}
    if boundary_enabled:
        points = _as_four_numbers(boundary_cfg.get("points"), cast=int, name="boundary sponge points")
        amplitudes = _as_four_numbers(
            boundary_cfg.get("amplitudes"), cast=float, name="boundary sponge amplitudes"
        )
        mode = str(boundary_cfg.get("mode", "add")).strip().lower()
        generated = _generate_boundary_sponge(sponge.shape, points=points, amplitudes=amplitudes)
        if mode == "replace":
            sponge = generated
        elif mode == "max":
            sponge = np.maximum(sponge, generated)
        elif mode == "add":
            sponge = sponge + generated
        else:
            raise ValueError("boundary sponge mode must be one of: add, max, replace")
        info["sponge"]["boundary"] = {
            "enabled": True,
            "mode": mode,
            "points": list(points),
            "amplitudes": list(amplitudes),
            "generated_max": float(np.max(generated)) if generated.size else 0.0,
        }
        info["applied"].append(
            "boundary_sponge="
            f"mode:{mode},points:{list(points)},amplitudes:{[float(a) for a in amplitudes]}"
        )

    filter_cfg = preprocess_cfg.get("filter", {})
    if filter_cfg is None:
        filter_cfg = {}
    if not isinstance(filter_cfg, dict):
        raise TypeError("baseflow_preprocess.filter must be a JSON object when provided")

    filter_enabled = bool(filter_cfg.get("enabled", False))
    passes = int(filter_cfg.get("passes", 0))
    fields = [str(name) for name in filter_cfg.get("fields", ["rho", "u", "v", "T"])]
    directions_raw = filter_cfg.get("directions", ["x", "y"])
    if not isinstance(directions_raw, (list, tuple)) or not directions_raw:
        raise ValueError("baseflow_preprocess.filter.directions must be a non-empty list")
    direction_map = {"x": 1, "y": 0}
    directions: list[str] = []
    axes: list[int] = []
    for item in directions_raw:
        name = str(item).strip().lower()
        if name not in direction_map:
            raise ValueError("filter directions must contain only 'x' and/or 'y'")
        if name not in directions:
            directions.append(name)
            axes.append(direction_map[name])

    eta_min = int(filter_cfg.get("eta_min_index", 0))
    eta_max_raw = filter_cfg.get("eta_max_index")
    eta_max = rho.shape[0] if eta_max_raw is None else int(eta_max_raw)
    eta_min = max(0, min(rho.shape[0], eta_min))
    eta_max = max(eta_min, min(rho.shape[0], eta_max))

    info["filter"] = {
        "enabled": filter_enabled,
        "passes": passes,
        "fields": fields,
        "directions": directions,
        "eta_min_index": eta_min,
        "eta_max_index": None if eta_max_raw is None else eta_max,
    }

    if filter_enabled and passes > 0 and eta_min < eta_max:
        field_map = {"rho": rho, "u": u, "v": v, "w": w, "T": T}
        invalid = [name for name in fields if name not in field_map]
        if invalid:
            raise ValueError(f"Unsupported filter field names: {invalid}")

        change_norms: dict[str, float] = {}
        region = (slice(eta_min, eta_max), slice(None))
        for name in fields:
            original = np.array(field_map[name], copy=True)
            filtered_region = np.array(field_map[name][region], copy=True)
            for _ in range(passes):
                for axis in axes:
                    filtered_region = _binomial5_filter(filtered_region, axis=axis)
            field_map[name][region] = filtered_region
            denom = float(np.linalg.norm(original))
            delta = float(np.linalg.norm(field_map[name] - original))
            change_norms[name] = 0.0 if denom == 0.0 else delta / denom

        info["filter"]["relative_l2_change"] = change_norms
        info["applied"].append(
            "explicit_filter="
            f"passes:{passes},fields:{fields},directions:{directions},eta:[{eta_min},{eta_max})"
        )

    info["sponge"]["max"] = float(np.max(sponge)) if sponge.size else 0.0
    info["summary"] = "; ".join(info["applied"]) if info["applied"] else "enabled(no-op)"
    return X, Y, rho, u, v, w, T, sponge, info


def load_my_case(
    case_dir: str | Path,
    *,
    Re: float,
    Ma: float,
    Pr: float,
    kappa: float,
    T_0: float,
    stride_ksi: int = 1,
    stride_eta: int = 1,
    transpose_input: bool = False,
    zero_spanwise: bool = False,
    preprocess_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, BaseFlow, np.ndarray, dict[str, Any]]:
    """
    Generic loader for user-supplied baseflows stored as .npy files.

    For the Song (2023) case, we expect all .npy fields to live in the
    ``baseflow`` subdirectory inside ``case_dir``:

        case_dir/
            baseflow/
                x.npy, y.npy
                rho.npy, u.npy, v.npy, w.npy, T.npy
                sponge.npy

    All arrays must be 2-D with the same shape.
    """
    root = Path(case_dir) / "baseflow"

    def _load(name: str) -> np.ndarray:
        path = root / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing field file: {path}")
        arr = np.asarray(np.load(path), dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{path} must be 2-D, got shape {arr.shape}")
        return arr

    X = _load("x")
    Y = _load("y")
    rho = _load("rho")
    u = _load("u")
    v = _load("v")
    w = _load("w")
    T = _load("T")
    sponge = _load("sponge")

    if transpose_input:
        X, Y, rho, u, v, w, T, sponge = [a.T for a in (X, Y, rho, u, v, w, T, sponge)]

    if stride_ksi < 1 or stride_eta < 1:
        raise ValueError("stride_ksi and stride_eta must be >= 1")

    if stride_ksi > 1 or stride_eta > 1:
        X = X[::stride_ksi, ::stride_eta]
        Y = Y[::stride_ksi, ::stride_eta]
        rho = rho[::stride_ksi, ::stride_eta]
        u = u[::stride_ksi, ::stride_eta]
        v = v[::stride_ksi, ::stride_eta]
        w = w[::stride_ksi, ::stride_eta]
        T = T[::stride_ksi, ::stride_eta]
        sponge = sponge[::stride_ksi, ::stride_eta]

    X, Y, rho, u, v, w, T, sponge, preprocess_info = _apply_preprocessing(
        X, Y, rho, u, v, w, T, sponge, preprocess_cfg
    )

    if zero_spanwise:
        w = np.zeros_like(u)

    shape = X.shape
    for name, arr in {
        "Y": Y,
        "rho": rho,
        "u": u,
        "v": v,
        "w": w,
        "T": T,
        "sponge": sponge,
    }.items():
        if arr.shape != shape:
            raise ValueError(f"Field '{name}' shape {arr.shape} does not match x shape {shape}")

    baseflow = BaseFlow(
        Re=float(Re),
        Ma=float(Ma),
        Pr=float(Pr),
        kappa=float(kappa),
        T_0=float(T_0),
        RHO=rho,
        U=u,
        V=v,
        W=w,
        T=T,
    )

    # Populate viscosity and derived transport coefficients
    sutherland_air(baseflow)

    return X, Y, baseflow, sponge, preprocess_info
