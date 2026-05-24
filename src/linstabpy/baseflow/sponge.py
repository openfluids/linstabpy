"""Native structured sponge generators for Case 1-style workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

EDGE_NAMES = ("top", "right", "bottom", "left")

# Direction ordering follows the legacy Case 1 workflow arrays:
# [top, right, bottom, left]
# where "top"/"bottom" act on the second array axis and "right"/"left"
# act on the first array axis.
CASE1_SPONGE_PRESETS: dict[str, dict[str, Any]] = {
    "case1_stability": {
        "type": "localized",
        "pts_dir": (0, 40, 65, 0),
        "amp_dir": (1000.0, 1000.0, 1000.0, 1000.0),
    },
    "case1_resolvent": {
        "type": "localized",
        "pts_dir": (0, 50, 80, 0),
        "amp_dir": (1000.0, 1000.0, 1000.0, 1000.0),
    },
    "case1_smooth_soft": {
        "type": "smooth_localized",
        "width_frac_dir": (0.0, 0.28, 0.28, 0.0),
        "amp_dir": (0.0, 0.5, 0.5, 0.0),
        "profile": "smootherstep",
    },
    "case1_smooth_medium": {
        "type": "smooth_localized",
        "width_frac_dir": (0.0, 0.22, 0.22, 0.0),
        "amp_dir": (0.0, 2.0, 2.0, 0.0),
        "profile": "smootherstep",
    },
    "case1_smooth_quarter": {
        "type": "smooth_localized",
        "width_frac_dir": (0.0, 0.28, 0.28, 0.0),
        "amp_dir": (0.0, 0.25, 0.25, 0.0),
        "profile": "smootherstep",
    },
    "case1_smooth_half": {
        "type": "smooth_localized",
        "width_frac_dir": (0.0, 0.28, 0.28, 0.0),
        "amp_dir": (0.0, 0.5, 0.5, 0.0),
        "profile": "smootherstep",
    },
    "case1_smooth_unit": {
        "type": "smooth_localized",
        "width_frac_dir": (0.0, 0.28, 0.28, 0.0),
        "amp_dir": (0.0, 1.0, 1.0, 0.0),
        "profile": "smootherstep",
    },
    "case1_smooth_double": {
        "type": "smooth_localized",
        "width_frac_dir": (0.0, 0.28, 0.28, 0.0),
        "amp_dir": (0.0, 2.0, 2.0, 0.0),
        "profile": "smootherstep",
    },
}


def _as_dir_counts(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=int).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{name} must contain exactly 4 entries [top, right, bottom, left]")
    if np.any(arr < 0):
        raise ValueError(f"{name} entries must be non-negative")
    return arr


def _as_dir_amplitudes(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{name} must contain exactly 4 entries [top, right, bottom, left]")
    return arr


def _as_dir_widths(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{name} must contain exactly 4 entries [top, right, bottom, left]")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} entries must be non-negative")
    return arr


def _as_dir_fractions(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{name} must contain exactly 4 entries [top, right, bottom, left]")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} entries must be non-negative")
    return arr


def _constant_sponge(shape: tuple[int, int], pts_dir: np.ndarray, amp_dir: np.ndarray) -> np.ndarray:
    sponge = np.zeros(shape, dtype=float)
    if pts_dir[0] != 0:
        sponge[:, -pts_dir[0] :] = amp_dir[0]
    if pts_dir[1] != 0:
        sponge[-pts_dir[1] :, :] = amp_dir[1]
    if pts_dir[2] != 0:
        sponge[:, : pts_dir[2]] = amp_dir[2]
    if pts_dir[3] != 0:
        sponge[: pts_dir[3], :] = amp_dir[3]
    return sponge


def _parabolic_sponge(shape: tuple[int, int], pts_dir: np.ndarray, amp_dir: np.ndarray) -> np.ndarray:
    sponge = np.zeros(shape, dtype=float)
    if pts_dir[0] != 0:
        width = pts_dir[0]
        for i in range(-width, 0):
            sponge[:, i] = amp_dir[0] * (width + 1 + i) ** 2 / width**2
    if pts_dir[1] != 0:
        width = pts_dir[1]
        for i in range(-width, 0):
            sponge[i, :] = amp_dir[1] * (width + 1 + i) ** 2 / width**2
    if pts_dir[2] != 0:
        width = pts_dir[2]
        for i in range(0, width + 1):
            sponge[:, i] = amp_dir[2] * (width - i) ** 2 / width**2
    if pts_dir[3] != 0:
        width = pts_dir[3]
        for i in range(0, width + 1):
            sponge[i, :] = amp_dir[3] * (width - i) ** 2 / width**2
    return sponge


def _localized_sponge(shape: tuple[int, int], pts_dir: np.ndarray, amp_dir: np.ndarray) -> np.ndarray:
    sponge = np.ones(shape, dtype=float)
    if pts_dir[0] != 0:
        sponge[:, :-pts_dir[0]] *= 0.0
    if pts_dir[1] != 0:
        sponge[:-pts_dir[1], :] *= 0.0
    if pts_dir[2] != 0:
        sponge[:, pts_dir[2] :] *= 0.0
    if pts_dir[3] != 0:
        sponge[pts_dir[3] :, :] *= 0.0
    sponge *= amp_dir[0]
    return sponge


def _cumulative_distances(X: np.ndarray, Y: np.ndarray, *, axis: int) -> np.ndarray:
    ds = np.sqrt(np.diff(X, axis=axis) ** 2 + np.diff(Y, axis=axis) ** 2)
    distances = np.zeros_like(X, dtype=float)
    if axis == 0:
        distances[1:, :] = np.cumsum(ds, axis=0)
    elif axis == 1:
        distances[:, 1:] = np.cumsum(ds, axis=1)
    else:
        raise ValueError(f"Unsupported axis={axis}; expected 0 or 1")
    return distances


def _boundary_distance_fields(X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have matching shapes, got {X.shape} and {Y.shape}")

    xi = _cumulative_distances(X, Y, axis=0)
    eta = _cumulative_distances(X, Y, axis=1)
    total_xi = xi[-1:, :]
    total_eta = eta[:, -1:]
    return {
        "left": xi,
        "right": total_xi - xi,
        "bottom": eta,
        "top": total_eta - eta,
    }


def _boundary_reference_lengths(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    distances = _boundary_distance_fields(X, Y)
    return np.array(
        [
            float(np.median(distances["top"][:, 0])),
            float(np.median(distances["right"][0, :])),
            float(np.median(distances["bottom"][:, -1])),
            float(np.median(distances["left"][-1, :])),
        ],
        dtype=float,
    )


def _smooth_profile(
    distance: np.ndarray,
    width: np.ndarray,
    *,
    profile: str,
    power: float,
    steepness: float,
) -> np.ndarray:
    ramp = np.zeros_like(distance, dtype=float)
    active = width > 0.0
    if not np.any(active):
        return ramp

    t = np.zeros_like(distance, dtype=float)
    t[active] = 1.0 - distance[active] / width[active]
    np.clip(t, 0.0, 1.0, out=t)

    profile_norm = str(profile).strip().lower()
    if profile_norm == "smoothstep":
        ramp = t * t * (3.0 - 2.0 * t)
    elif profile_norm == "smootherstep":
        ramp = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    elif profile_norm == "cosine":
        ramp = 0.5 * (1.0 - np.cos(np.pi * t))
    elif profile_norm == "tanh":
        if steepness <= 0.0:
            raise ValueError(f"steepness must be > 0 for tanh profile, got {steepness}")
        ramp = np.tanh(steepness * t) / np.tanh(steepness)
    else:
        raise ValueError(
            f"Unsupported smooth sponge profile {profile!r}; "
            "expected smoothstep|smootherstep|cosine|tanh"
        )

    if power <= 0.0:
        raise ValueError(f"power must be > 0, got {power}")
    if power != 1.0:
        ramp = ramp**power
    return ramp


def _smooth_boundary_sponge(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    width_dir: np.ndarray,
    amp_dir: np.ndarray,
    profile: str,
    power: float,
    steepness: float,
    blend: str,
) -> np.ndarray:
    distances = _boundary_distance_fields(X, Y)
    sponge = np.zeros_like(X, dtype=float)
    blend_norm = str(blend).strip().lower()

    for edge, width, amplitude in zip(EDGE_NAMES, width_dir, amp_dir, strict=True):
        if width <= 0.0 or amplitude == 0.0:
            continue
        width_field = np.full_like(X, float(width), dtype=float)
        ramp = float(amplitude) * _smooth_profile(
            distances[edge],
            width_field,
            profile=profile,
            power=power,
            steepness=steepness,
        )
        if blend_norm == "max":
            sponge = np.maximum(sponge, ramp)
        elif blend_norm in {"add", "sum", "additive"}:
            sponge += ramp
        elif blend_norm in {"rss", "quadrature"}:
            sponge = np.sqrt(sponge * sponge + ramp * ramp)
        else:
            raise ValueError(
                f"Unsupported smooth sponge blend {blend!r}; expected max|additive|quadrature"
            )
    return sponge


def _smooth_localized_sponge(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    width_dir: np.ndarray,
    amp_dir: np.ndarray,
    profile: str,
    power: float,
    steepness: float,
) -> np.ndarray:
    distances = _boundary_distance_fields(X, Y)
    active_edges = [
        edge for edge, width, amplitude in zip(EDGE_NAMES, width_dir, amp_dir, strict=True)
        if width > 0.0 and amplitude != 0.0
    ]
    if not active_edges:
        return np.zeros_like(X, dtype=float)

    ramp_product = np.ones_like(X, dtype=float)
    active_amplitudes: list[float] = []
    for edge, width, amplitude in zip(EDGE_NAMES, width_dir, amp_dir, strict=True):
        if width <= 0.0 or amplitude == 0.0:
            continue
        width_field = np.full_like(X, float(width), dtype=float)
        ramp = _smooth_profile(
            distances[edge],
            width_field,
            profile=profile,
            power=power,
            steepness=steepness,
        )
        ramp_product *= ramp
        active_amplitudes.append(float(amplitude))

    if not active_amplitudes:
        return np.zeros_like(X, dtype=float)
    amplitude = max(active_amplitudes)
    return amplitude * ramp_product


def generate_case1_sponge(
    shape: tuple[int, int],
    *,
    sponge_type: str = "localized",
    pts_dir: Any = (0, 0, 0, 0),
    amp_dir: Any = (0.0, 0.0, 0.0, 0.0),
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
    width_dir: Any = (0.0, 0.0, 0.0, 0.0),
    width_frac_dir: Any | None = None,
    profile: str = "smootherstep",
    power: float = 1.0,
    steepness: float = 4.0,
    blend: str = "max",
) -> np.ndarray:
    """Generate a native Case 1 sponge field on a structured grid.

    The implementation is a direct translation of the legacy Case 1
    ``workflow/sponge_generator.py`` helper.
    """
    pts = _as_dir_counts(pts_dir, name="pts_dir")
    amps = _as_dir_amplitudes(amp_dir, name="amp_dir")
    sponge_type_norm = str(sponge_type).strip().lower()
    if sponge_type_norm == "constant":
        return _constant_sponge(shape, pts, amps)
    if sponge_type_norm == "parabolic":
        return _parabolic_sponge(shape, pts, amps)
    if sponge_type_norm == "localized":
        return _localized_sponge(shape, pts, amps)
    if sponge_type_norm in {"smooth", "smooth_boundary", "smooth_localized"}:
        if X is None or Y is None:
            raise ValueError(f"{sponge_type_norm} sponge requires X and Y coordinate arrays")
        X_arr = np.asarray(X, dtype=float)
        Y_arr = np.asarray(Y, dtype=float)
        if X_arr.shape != shape or Y_arr.shape != shape:
            raise ValueError(
                f"X/Y shapes must match sponge shape {shape}, got {X_arr.shape} and {Y_arr.shape}"
            )
        widths = _as_dir_widths(width_dir, name="width_dir")
        if width_frac_dir is not None:
            width_fracs = _as_dir_fractions(width_frac_dir, name="width_frac_dir")
            widths = widths + width_fracs * _boundary_reference_lengths(X_arr, Y_arr)
        if sponge_type_norm in {"smooth", "smooth_boundary"}:
            return _smooth_boundary_sponge(
                X_arr,
                Y_arr,
                width_dir=widths,
                amp_dir=amps,
                profile=profile,
                power=float(power),
                steepness=float(steepness),
                blend=blend,
            )
        return _smooth_localized_sponge(
            X_arr,
            Y_arr,
            width_dir=widths,
            amp_dir=amps,
            profile=profile,
            power=float(power),
            steepness=float(steepness),
        )
    raise ValueError(
        f"Unsupported sponge_type={sponge_type!r}; "
        "expected constant|parabolic|localized|smooth_boundary|smooth_localized"
    )


def case1_sponge_from_config(
    shape: tuple[int, int],
    cfg: dict[str, Any] | None,
    *,
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
) -> np.ndarray:
    """Build a native Case 1 sponge from config or return zeros when disabled."""
    if not cfg:
        return np.zeros(shape, dtype=float)
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return np.zeros(shape, dtype=float)

    preset = cfg.get("preset")
    resolved: dict[str, Any] = {}
    if preset is not None:
        preset_key = str(preset).strip().lower()
        if preset_key not in CASE1_SPONGE_PRESETS:
            raise ValueError(
                f"Unsupported sponge preset {preset!r}; "
                f"expected one of {sorted(CASE1_SPONGE_PRESETS)}"
            )
        resolved.update(CASE1_SPONGE_PRESETS[preset_key])
    resolved.update({k: v for k, v in cfg.items() if k != "enabled"})

    return generate_case1_sponge(
        shape,
        sponge_type=resolved.get("type", "localized"),
        pts_dir=resolved.get("pts_dir", (0, 0, 0, 0)),
        amp_dir=resolved.get("amp_dir", (0.0, 0.0, 0.0, 0.0)),
        X=X,
        Y=Y,
        width_dir=resolved.get("width_dir", (0.0, 0.0, 0.0, 0.0)),
        width_frac_dir=resolved.get("width_frac_dir"),
        profile=resolved.get("profile", "smootherstep"),
        power=float(resolved.get("power", 1.0)),
        steepness=float(resolved.get("steepness", 4.0)),
        blend=resolved.get("blend", "max"),
    )
