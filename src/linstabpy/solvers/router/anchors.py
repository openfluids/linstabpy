# SPDX-License-Identifier: GPL-3.0-only
"""Adaptive pre-sweep anchor placement for stable-regime resolvent sweeps."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from linstabpy.solvers.router.band_marking import BandMarks
from linstabpy.utils._solver_cfg import as_bool as _as_bool
from linstabpy.utils._solver_cfg import cfg_get

Array = np.ndarray
Cfg = Mapping[str, Any] | object | None

COARSE_TRIGGER = "coarse"
EVP_DISTANCE_TRIGGER = "evp_distance"
FGMRES_ITERATIONS_TRIGGER = "fgmres_iterations"
FGMRES_RESIDUAL_TRIGGER = "fgmres_residual"


@dataclass(frozen=True)
class IterProbeResult:
    """Measured FGMRES work at one sample frequency."""

    iterations: int = 0
    residual: float = 0.0


@dataclass(frozen=True)
class AnchorTriggerRecord:
    """Provenance for one final anchor returned by the adaptive chooser."""

    omega: complex
    sample_omega: complex
    triggers: tuple[str, ...]
    inserted: bool
    evp_distance: float = float("nan")
    evp_radius: float = float("nan")
    fgmres_iterations: int = -1
    fgmres_residual: float = float("nan")
    nearest_anchor_before: complex | None = None
    nearest_anchor_distance_before: float = float("nan")

    @property
    def trigger(self) -> str:
        """Backward-compatible single-string view of ``triggers``."""
        return "+".join(self.triggers)


IterProbe = Callable[[complex, Array, Cfg], IterProbeResult | None]


def choose_anchors_adaptive(
    omega_range: Sequence[complex] | Array,
    band_marks: BandMarks | None,
    cfg: Cfg,
) -> tuple[Array, tuple[AnchorTriggerRecord, ...]]:
    """Choose a deterministic adaptive anchor set before a stable sweep.

    ``cfg`` may define the canonical keys at top level or inside an
    ``adaptive_anchors`` subsection.

    Keys include ``coarse_anchor_count``,
    ``coarse_anchors``, ``sample_omegas``, ``evp_distance_radius``,
    ``fgmres_iter_threshold``, ``fgmres_residual_threshold``,
    ``min_insert_spacing``, ``duplicate_tol``, ``max_insertions``,
    ``enable_evp_distance``, ``enable_fgmres_probe``, ``iter_probe``,
    ``fgmres_iterations``, and ``fgmres_residuals``.
    """
    omega_grid = _sort_unique_complex(_as_1d_complex(omega_range, "omega_range"))
    records = [
        AnchorTriggerRecord(
            omega=complex(anchor),
            sample_omega=complex(anchor),
            triggers=(COARSE_TRIGGER,),
            inserted=False,
        )
        for anchor in _initial_anchors(omega_grid, cfg)
    ]
    anchors = np.asarray([record.omega for record in records], dtype=complex)

    samples = _sample_omegas(omega_grid, cfg)
    evp_radius = _evp_radius(band_marks, cfg)
    iter_threshold = _threshold(
        _cfg_get(cfg, "fgmres_iter_threshold"), default=40.0, name="fgmres_iter_threshold"
    )
    residual_threshold = _threshold(
        _cfg_get(cfg, "fgmres_residual_threshold"),
        default=1.0e-6,
        name="fgmres_residual_threshold",
    )
    min_insert_spacing = _nonnegative_float(
        _cfg_get(cfg, "min_insert_spacing", 0.0), "min_insert_spacing"
    )
    duplicate_tol = _nonnegative_float(_cfg_get(cfg, "duplicate_tol", 1.0e-12), "duplicate_tol")
    max_insertions_raw = _cfg_get(cfg, "max_insertions")
    max_insertions = (
        None
        if max_insertions_raw is None
        else _nonnegative_int(max_insertions_raw, "max_insertions")
    )
    enable_evp = _as_bool(_cfg_get(cfg, "enable_evp_distance", True))
    enable_fgmres = _as_bool(_cfg_get(cfg, "enable_fgmres_probe", True))
    probe_fn = _cfg_get(cfg, "iter_probe")
    insertions = 0

    for sample in samples:
        sample = complex(sample)
        evp_distance = _band_distance_at(sample, band_marks)
        probe = (
            _probe_result(sample, anchors, omega_grid, cfg, probe_fn)
            if enable_fgmres
            else IterProbeResult()
        )
        triggers: list[str] = []
        if (
            enable_evp
            and np.isfinite(evp_radius)
            and evp_radius > 0.0
            and evp_distance < evp_radius
        ):
            triggers.append(EVP_DISTANCE_TRIGGER)
        if enable_fgmres and probe.iterations > iter_threshold:
            triggers.append(FGMRES_ITERATIONS_TRIGGER)
        if enable_fgmres and probe.residual > residual_threshold:
            triggers.append(FGMRES_RESIDUAL_TRIGGER)
        if not triggers:
            continue

        nearest_i, nearest_anchor, nearest_distance = _nearest_anchor(sample, anchors)
        if nearest_distance <= max(min_insert_spacing, duplicate_tol):
            records[nearest_i] = _annotate_record(
                records[nearest_i],
                sample,
                tuple(triggers),
                evp_distance,
                evp_radius,
                probe,
                nearest_anchor,
                nearest_distance,
            )
            continue
        if max_insertions is not None and insertions >= max_insertions:
            continue

        records.append(
            AnchorTriggerRecord(
                omega=sample,
                sample_omega=sample,
                triggers=tuple(triggers),
                inserted=True,
                evp_distance=evp_distance,
                evp_radius=evp_radius,
                fgmres_iterations=probe.iterations,
                fgmres_residual=probe.residual,
                nearest_anchor_before=nearest_anchor,
                nearest_anchor_distance_before=nearest_distance,
            )
        )
        insertions += 1
        records = _sort_records(records)
        anchors = np.asarray([record.omega for record in records], dtype=complex)

    records = _sort_records(records)
    return np.asarray([record.omega for record in records], dtype=complex), tuple(records)


def _cfg_get(cfg: Cfg, key: str, default: Any = None) -> Any:
    # An explicit ``None`` value counts as unset for anchor controls.
    value = cfg_get(cfg, key, default, section="adaptive_anchors")
    return default if value is None else value


def _as_1d_complex(values: Sequence[complex] | Array, name: str) -> Array:
    arr = np.asarray(values, dtype=complex).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not bool(np.all(np.isfinite(arr.real) & np.isfinite(arr.imag))):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _sort_unique_complex(values: Array) -> Array:
    return np.sort_complex(np.unique(np.asarray(values, dtype=complex).reshape(-1)))


def _sort_records(records: Sequence[AnchorTriggerRecord]) -> list[AnchorTriggerRecord]:
    return sorted(records, key=lambda record: (record.omega.real, record.omega.imag))


def _initial_anchors(omega_grid: Array, cfg: Cfg) -> Array:
    raw = _cfg_get(cfg, "coarse_anchors")
    if raw is not None:
        return _sort_unique_complex(_as_1d_complex(raw, "coarse_anchors"))
    count = min(
        _positive_int(
            _cfg_get(cfg, "coarse_anchor_count", min(2, omega_grid.size)), "coarse_anchor_count"
        ),
        omega_grid.size,
    )
    return omega_grid[_coarse_indices(omega_grid, count)]


def _coarse_indices(omega_grid: Array, count: int) -> Array:
    if count == 1:
        return np.array([0], dtype=int)
    indices = np.unique(np.rint(np.linspace(0, omega_grid.size - 1, count)).astype(int))
    while indices.size < count:
        candidates = np.setdiff1d(
            np.arange(omega_grid.size, dtype=int), indices, assume_unique=True
        )
        distances = np.min(
            np.abs(omega_grid[candidates, None] - omega_grid[indices][None, :]), axis=1
        )
        indices = np.sort(np.r_[indices, candidates[int(np.argmax(distances))]].astype(int))
    return indices.astype(int)


def _sample_omegas(omega_grid: Array, cfg: Cfg) -> Array:
    raw = _cfg_get(cfg, "sample_omegas")
    return (
        omega_grid.copy()
        if raw is None
        else _sort_unique_complex(_as_1d_complex(raw, "sample_omegas"))
    )


def _evp_radius(band_marks: BandMarks | None, cfg: Cfg) -> float:
    raw = _cfg_get(cfg, "evp_distance_radius")
    if raw is None:
        raw = getattr(band_marks, "danger_radius", 0.0) if band_marks is not None else 0.0
    radius = float(raw)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError(f"evp_distance_radius must be a finite non-negative float, got {raw!r}")
    return radius


def _band_distance_at(sample: complex, band_marks: BandMarks | None) -> float:
    if band_marks is None:
        return float("inf")
    omegas = np.asarray(band_marks.omegas, dtype=complex).reshape(-1)
    distances = np.asarray(band_marks.distances, dtype=float).reshape(-1)
    if omegas.size != distances.size:
        raise ValueError("band_marks.omegas and band_marks.distances must have the same length")
    finite = np.isfinite(omegas.real) & np.isfinite(omegas.imag) & np.isfinite(distances)
    if omegas.size == 0 or not bool(np.any(finite)):
        return float("inf")
    return float(distances[finite][int(np.argmin(np.abs(omegas[finite] - sample)))])


def _threshold(raw: Any, *, default: float, name: str) -> float:
    value = default if raw is None else float(raw)
    if np.isnan(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative threshold or infinity, got {raw!r}")
    return value


def _positive_int(raw: Any, name: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(raw: Any, name: str) -> int:
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _nonnegative_float(raw: Any, name: str) -> float:
    value = float(raw)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")
    return value


def _nearest_anchor(sample: complex, anchors: Array) -> tuple[int, complex, float]:
    distances = np.abs(np.asarray(anchors, dtype=complex).reshape(-1) - sample)
    index = int(np.argmin(distances))
    return index, complex(anchors[index]), float(distances[index])


def _annotate_record(
    record: AnchorTriggerRecord,
    sample: complex,
    triggers: tuple[str, ...],
    evp_distance: float,
    evp_radius: float,
    probe: IterProbeResult,
    nearest_anchor_before: complex,
    nearest_distance_before: float,
) -> AnchorTriggerRecord:
    return replace(
        record,
        sample_omega=sample,
        triggers=tuple(dict.fromkeys((*record.triggers, *triggers))),
        evp_distance=evp_distance,
        evp_radius=evp_radius,
        fgmres_iterations=probe.iterations,
        fgmres_residual=probe.residual,
        nearest_anchor_before=nearest_anchor_before,
        nearest_anchor_distance_before=nearest_distance_before,
    )


def _probe_result(
    sample: complex, anchors: Array, omega_grid: Array, cfg: Cfg, probe_fn: Any
) -> IterProbeResult:
    raw = (
        _cheap_probe_from_cfg(sample, omega_grid, cfg)
        if probe_fn is None
        else _call_probe(probe_fn, sample, anchors, cfg)
    )
    return _coerce_probe_result(raw)


def _call_probe(
    probe_fn: IterProbe, sample: complex, anchors: Array, cfg: Cfg
) -> IterProbeResult | None:
    return probe_fn(sample, anchors, cfg)


def _cheap_probe_from_cfg(sample: complex, omega_grid: Array, cfg: Cfg) -> IterProbeResult:
    iterations = _measurement_at(
        _cfg_get(cfg, "fgmres_iterations"), sample, omega_grid, default=0.0
    )
    residual = _measurement_at(_cfg_get(cfg, "fgmres_residuals"), sample, omega_grid, default=0.0)
    return IterProbeResult(iterations=int(iterations), residual=float(residual))


def _measurement_at(raw: Any, sample: complex, omega_grid: Array, *, default: float) -> float:
    if raw is None:
        return default
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size == 0:
        return default
    if values.size != omega_grid.size:
        raise ValueError("FGMRES measurement arrays must align with omega_range")
    return float(values[int(np.argmin(np.abs(omega_grid - sample)))])


def _coerce_probe_result(raw: IterProbeResult | None) -> IterProbeResult:
    if raw is None:
        raw = IterProbeResult()
    if not isinstance(raw, IterProbeResult):
        raise TypeError("iter_probe must return IterProbeResult or None")
    if raw.iterations < 0:
        raise ValueError("iter_probe returned negative iterations")
    if not np.isfinite(raw.residual) or raw.residual < 0.0:
        raise ValueError("iter_probe returned a non-finite or negative residual")
    return raw


__all__ = [
    "AnchorTriggerRecord",
    "COARSE_TRIGGER",
    "EVP_DISTANCE_TRIGGER",
    "FGMRES_ITERATIONS_TRIGGER",
    "FGMRES_RESIDUAL_TRIGGER",
    "IterProbeResult",
    "choose_anchors_adaptive",
]
