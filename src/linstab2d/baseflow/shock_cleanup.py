"""Optional shock-aware mean-flow cleanup helpers.

This module is intentionally self-contained so experimental shock cleanup can
be enabled or removed without touching the default Case 1 / Case 2 workflow.
All behavior is opt-in through the ``enabled`` flag in the cleanup config.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

_VALID_FIELDS = {"rho", "u", "v", "w", "T"}
_VALID_DIRECTIONS = {"x": 1, "y": 0}

_DEFAULT_CFG: dict[str, Any] = {
    "enabled": False,
    "sensor": {
        "enabled": False,
        "kind": "compression_ducros",
        "compression_threshold": -1.0,
        "ducros_threshold": 0.95,
        "eps": 1.0e-14,
        "buffer_cells": 3,
        "blend_sigma": 0.0,
    },
    "filter": {
        "enabled": False,
        "passes": 1,
        "directions": ["x", "y"],
        "fields": ["rho", "u", "v", "T"],
    },
    "sponge": {
        "enabled": False,
        "amplitude": 0.0,
        "mode": "add",
        "smooth_sigma": 0.0,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return _deep_merge(_DEFAULT_CFG, {})
    if not isinstance(cfg, dict):
        raise TypeError("shock_cleanup config must be a JSON object when provided")
    return _deep_merge(_DEFAULT_CFG, cfg)


def _normalize_directions(values: Any) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("shock_cleanup.filter.directions must be a non-empty list")
    axes: list[int] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip().lower()
        if name not in _VALID_DIRECTIONS:
            raise ValueError("shock_cleanup.filter.directions must contain only 'x' and/or 'y'")
        if name in seen:
            continue
        seen.add(name)
        axes.append(_VALID_DIRECTIONS[name])
    return axes


def _normalize_fields(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("shock_cleanup.filter.fields must be a non-empty list")
    fields: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value)
        if name not in _VALID_FIELDS:
            raise ValueError(
                "shock_cleanup.filter.fields must contain only "
                + ", ".join(sorted(_VALID_FIELDS))
            )
        if name in seen:
            continue
        seen.add(name)
        fields.append(name)
    return fields


def _binomial5_filter(arr: np.ndarray, axis: int) -> np.ndarray:
    """Apply a compact 5-point smoothing kernel along one axis."""
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D field, got shape {arr.shape}")
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


def _computational_gradients(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = field.shape
    eta = np.linspace(0.0, 1.0, ny, dtype=float)
    xi = np.linspace(0.0, 1.0, nx, dtype=float)
    edge_order = 2 if ny >= 3 and nx >= 3 else 1
    d_deta, d_dxi = np.gradient(field, eta, xi, edge_order=edge_order)
    return np.asarray(d_dxi, dtype=float), np.asarray(d_deta, dtype=float)


def _physical_derivatives(
    field: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    jacobian_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx_dxi, dx_deta = _computational_gradients(X)
    dy_dxi, dy_deta = _computational_gradients(Y)
    df_dxi, df_deta = _computational_gradients(field)

    jac = dx_dxi * dy_deta - dx_deta * dy_dxi
    sign = np.where(jac >= 0.0, 1.0, -1.0)
    safe_jac = np.where(np.abs(jac) >= jacobian_eps, jac, sign * jacobian_eps)

    df_dx = (df_dxi * dy_deta - df_deta * dy_dxi) / safe_jac
    df_dy = (-df_dxi * dx_deta + df_deta * dx_dxi) / safe_jac
    return df_dx, df_dy


def _compression_ducros_sensor(
    X: np.ndarray,
    Y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    compression_threshold: float,
    ducros_threshold: float,
    eps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    du_dx, du_dy = _physical_derivatives(u, X, Y, jacobian_eps=max(eps, 1.0e-12))
    dv_dx, dv_dy = _physical_derivatives(v, X, Y, jacobian_eps=max(eps, 1.0e-12))

    div = du_dx + dv_dy
    vort = dv_dx - du_dy
    ducros = div * div / (div * div + vort * vort + eps)
    mask = (div <= compression_threshold) & (ducros >= ducros_threshold)
    return mask, {
        "divergence": div,
        "vorticity": vort,
        "ducros_ratio": ducros,
    }


def _compression_only_sensor(
    X: np.ndarray,
    Y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    compression_threshold: float,
    eps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    du_dx, _ = _physical_derivatives(u, X, Y, jacobian_eps=max(eps, 1.0e-12))
    _, dv_dy = _physical_derivatives(v, X, Y, jacobian_eps=max(eps, 1.0e-12))
    div = du_dx + dv_dy
    mask = div <= compression_threshold
    return mask, {
        "divergence": div,
        "vorticity": np.zeros_like(div),
        "ducros_ratio": np.ones_like(div, dtype=float),
    }


def build_shock_sensor(
    X: np.ndarray,
    Y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a shock mask and supporting diagnostics."""
    kind = str(cfg.get("kind", "compression_ducros")).strip().lower()
    compression_threshold = float(cfg.get("compression_threshold", -1.0))
    ducros_threshold = float(cfg.get("ducros_threshold", 0.95))
    eps = float(cfg.get("eps", 1.0e-14))

    if kind == "compression_ducros":
        mask, raw = _compression_ducros_sensor(
            X,
            Y,
            u,
            v,
            compression_threshold=compression_threshold,
            ducros_threshold=ducros_threshold,
            eps=eps,
        )
    elif kind == "compression_only":
        mask, raw = _compression_only_sensor(
            X,
            Y,
            u,
            v,
            compression_threshold=compression_threshold,
            eps=eps,
        )
    elif kind == "modified_ducros":
        raise NotImplementedError(
            "shock_cleanup sensor kind 'modified_ducros' is reserved for the group-supplied formula"
        )
    else:
        raise ValueError(
            "shock_cleanup.sensor.kind must be one of "
            "'compression_ducros', 'compression_only', or 'modified_ducros'"
        )

    buffer_cells = int(cfg.get("buffer_cells", 0))
    if buffer_cells < 0:
        raise ValueError("shock_cleanup.sensor.buffer_cells must be >= 0")
    buffered_mask = binary_dilation(mask, iterations=buffer_cells) if buffer_cells > 0 else mask.copy()

    blend_sigma = float(cfg.get("blend_sigma", 0.0))
    if blend_sigma < 0.0:
        raise ValueError("shock_cleanup.sensor.blend_sigma must be >= 0")
    preserve_weight = buffered_mask.astype(float)
    if blend_sigma > 0.0 and np.any(buffered_mask):
        preserve_weight = gaussian_filter(buffered_mask.astype(float), sigma=blend_sigma, mode="nearest")
        peak = float(np.max(preserve_weight))
        if peak > 0.0:
            preserve_weight /= peak
        preserve_weight = np.clip(preserve_weight, 0.0, 1.0)
        preserve_weight[buffered_mask] = 1.0

    diagnostics: dict[str, Any] = {
        "kind": kind,
        "compression_threshold": compression_threshold,
        "ducros_threshold": ducros_threshold,
        "buffer_cells": buffer_cells,
        "blend_sigma": blend_sigma,
        "mask": mask,
        "buffered_mask": buffered_mask,
        "preserve_weight": preserve_weight,
        "mask_fraction": float(np.mean(mask)),
        "buffered_mask_fraction": float(np.mean(buffered_mask)),
        "divergence": raw["divergence"],
        "vorticity": raw["vorticity"],
        "ducros_ratio": raw["ducros_ratio"],
    }
    return mask, diagnostics


def _apply_masked_filter(
    field: np.ndarray,
    *,
    preserve_weight: np.ndarray,
    passes: int,
    axes: list[int],
) -> np.ndarray:
    if passes <= 0:
        return np.array(field, copy=True)

    original = np.asarray(field, dtype=float)
    filtered = np.array(original, copy=True)
    for _ in range(passes):
        for axis in axes:
            candidate = _binomial5_filter(filtered, axis=axis)
            filtered = preserve_weight * original + (1.0 - preserve_weight) * candidate
    return filtered


def _generate_shock_sponge(
    buffered_mask: np.ndarray,
    *,
    amplitude: float,
    smooth_sigma: float,
) -> np.ndarray:
    if amplitude == 0.0 or not np.any(buffered_mask):
        return np.zeros_like(buffered_mask, dtype=float)
    support = buffered_mask.astype(float)
    if smooth_sigma > 0.0:
        support = gaussian_filter(support, sigma=smooth_sigma, mode="nearest")
        peak = float(np.max(support))
        if peak > 0.0:
            support /= peak
    support = np.clip(support, 0.0, None)
    return float(amplitude) * support


def apply_optional_shock_cleanup(
    X: np.ndarray,
    Y: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    sponge: np.ndarray,
    cfg: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply an optional shock-preserving cleanup to the mean flow.

    Returns the possibly modified ``rho, u, v, w, T, sponge`` arrays plus a
    diagnostic dictionary. When disabled, all arrays are returned unchanged.
    """
    config = _normalize_cfg(cfg)
    info: dict[str, Any] = {
        "enabled": bool(config.get("enabled", False)),
        "summary": "disabled",
        "sensor": {"enabled": False},
        "filter": {"enabled": False},
        "sponge": {"enabled": False},
    }

    rho_out = np.array(rho, copy=True)
    u_out = np.array(u, copy=True)
    v_out = np.array(v, copy=True)
    w_out = np.array(w, copy=True)
    T_out = np.array(T, copy=True)
    sponge_out = np.array(sponge, copy=True)

    if not info["enabled"]:
        return rho_out, u_out, v_out, w_out, T_out, sponge_out, info

    sensor_cfg = config.get("sensor", {})
    filter_cfg = config.get("filter", {})
    sponge_cfg = config.get("sponge", {})

    sensor_enabled = bool(sensor_cfg.get("enabled", False))
    filter_enabled = bool(filter_cfg.get("enabled", False))
    shock_sponge_enabled = bool(sponge_cfg.get("enabled", False))
    if (filter_enabled or shock_sponge_enabled) and not sensor_enabled:
        raise ValueError(
            "shock_cleanup.sensor.enabled must be true when filter or sponge cleanup is enabled"
        )

    applied: list[str] = []
    sensor_info: dict[str, Any] = {"enabled": sensor_enabled}
    if sensor_enabled:
        _, sensor_info = build_shock_sensor(X, Y, u_out, v_out, sensor_cfg)
        sensor_info["enabled"] = True
        applied.append(
            "sensor="
            f"{sensor_info['kind']},mask={sensor_info['mask_fraction']:.4f},"
            f"buffered={sensor_info['buffered_mask_fraction']:.4f}"
        )
    info["sensor"] = sensor_info

    preserve_weight = np.asarray(sensor_info.get("preserve_weight", np.zeros_like(X, dtype=float)), dtype=float)
    buffered_mask = np.asarray(sensor_info.get("buffered_mask", np.zeros_like(X, dtype=bool)), dtype=bool)

    field_map = {
        "rho": rho_out,
        "u": u_out,
        "v": v_out,
        "w": w_out,
        "T": T_out,
    }

    filter_info: dict[str, Any] = {"enabled": filter_enabled}
    if filter_enabled:
        fields = _normalize_fields(filter_cfg.get("fields", ["rho", "u", "v", "T"]))
        passes = int(filter_cfg.get("passes", 1))
        if passes < 0:
            raise ValueError("shock_cleanup.filter.passes must be >= 0")
        axes = _normalize_directions(filter_cfg.get("directions", ["x", "y"]))
        relative_l2_change: dict[str, float] = {}
        max_abs_change: dict[str, float] = {}

        for name in fields:
            original = np.array(field_map[name], copy=True)
            field_map[name][...] = _apply_masked_filter(
                original,
                preserve_weight=preserve_weight,
                passes=passes,
                axes=axes,
            )
            delta = field_map[name] - original
            denom = float(np.linalg.norm(original))
            relative_l2_change[name] = 0.0 if denom == 0.0 else float(np.linalg.norm(delta) / denom)
            max_abs_change[name] = float(np.max(np.abs(delta))) if delta.size else 0.0

        filter_info = {
            "enabled": True,
            "fields": fields,
            "passes": passes,
            "directions": [name for name, axis in _VALID_DIRECTIONS.items() if axis in axes],
            "relative_l2_change": relative_l2_change,
            "max_abs_change": max_abs_change,
        }
        applied.append(
            "filter="
            f"passes:{passes},fields:{fields},dirs:{filter_info['directions']}"
        )
    info["filter"] = filter_info

    shock_sponge = np.zeros_like(sponge_out, dtype=float)
    sponge_info: dict[str, Any] = {"enabled": shock_sponge_enabled}
    if shock_sponge_enabled:
        amplitude = float(sponge_cfg.get("amplitude", 0.0))
        if amplitude < 0.0:
            raise ValueError("shock_cleanup.sponge.amplitude must be >= 0")
        mode = str(sponge_cfg.get("mode", "add")).strip().lower()
        if mode not in {"add", "max"}:
            raise ValueError("shock_cleanup.sponge.mode must be either 'add' or 'max'")
        smooth_sigma = float(sponge_cfg.get("smooth_sigma", 0.0))
        if smooth_sigma < 0.0:
            raise ValueError("shock_cleanup.sponge.smooth_sigma must be >= 0")

        shock_sponge = _generate_shock_sponge(
            buffered_mask,
            amplitude=amplitude,
            smooth_sigma=smooth_sigma,
        )
        if mode == "add":
            sponge_out += shock_sponge
        else:
            sponge_out = np.maximum(sponge_out, shock_sponge)
        sponge_info = {
            "enabled": True,
            "mode": mode,
            "amplitude": amplitude,
            "smooth_sigma": smooth_sigma,
            "max": float(np.max(shock_sponge)) if shock_sponge.size else 0.0,
            "fraction_nonzero": float(np.mean(shock_sponge > 0.0)),
            "shock_sponge": shock_sponge,
        }
        applied.append(
            "shock_sponge="
            f"mode:{mode},amp:{amplitude:.6g},max:{sponge_info['max']:.6g}"
        )
    else:
        sponge_info["shock_sponge"] = shock_sponge
    info["sponge"] = sponge_info

    info["summary"] = "; ".join(applied) if applied else "enabled(no-op)"
    info["mask"] = np.asarray(sensor_info.get("mask", np.zeros_like(X, dtype=bool)), dtype=bool)
    info["buffered_mask"] = buffered_mask
    info["preserve_weight"] = preserve_weight
    info["shock_sponge"] = shock_sponge
    info["filtered_fraction"] = float(np.mean(1.0 - preserve_weight))
    return rho_out, u_out, v_out, w_out, T_out, sponge_out, info
