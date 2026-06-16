# SPDX-License-Identifier: GPL-3.0-only
"""Adaptive pre-sweep anchor placement for stable-regime resolvent sweeps.

The chooser is intentionally a pure PRE-sweep helper: it reads a requested
frequency grid, S6 :class:`~linstabpy.solvers.router.band_marking.BandMarks`,
and optional measured FGMRES probe data, then returns a fixed anchor set for the
caller to pass into :func:`linstabpy.solvers.router.stable_sweep.sweep_stable_regime`.
No in-sweep state or per-RHS information is used here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from linstabpy.solvers.router.band_marking import BandMarks

Array = np.ndarray

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
    """Provenance for one final anchor returned by the adaptive chooser.

    ``inserted`` is ``False`` for anchors belonging to the initial coarse set.
    ``triggers`` records why the anchor is present or, for a coarse anchor, why
    it was retained/annotated when a triggering sample was already covered.
    """

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

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record for logs or manifests."""
        return {
            "omega": self.omega,
            "sample_omega": self.sample_omega,
            "triggers": self.triggers,
            "trigger": self.trigger,
            "inserted": self.inserted,
            "evp_distance": self.evp_distance,
            "evp_radius": self.evp_radius,
            "fgmres_iterations": self.fgmres_iterations,
            "fgmres_residual": self.fgmres_residual,
            "nearest_anchor_before": self.nearest_anchor_before,
            "nearest_anchor_distance_before": self.nearest_anchor_distance_before,
        }


IterProbe = Callable[[complex, Array, Mapping[str, Any] | object | None], IterProbeResult | Mapping[str, Any] | Sequence[Any] | None]


def choose_anchors_adaptive(
    omega_range: Sequence[complex] | Array,
    band_marks: BandMarks | None,
    cfg: Mapping[str, Any] | object | None,
) -> tuple[Array, tuple[AnchorTriggerRecord, ...]]:
    """Choose a deterministic adaptive anchor set before a stable sweep.

    Parameters
    ----------
    omega_range:
        Requested forcing frequencies for the sweep.  Values are sorted and
        de-duplicated before selecting the coarse set and sample points.
    band_marks:
        S6 band-marking output.  Only ``BandMarks.omegas``, ``distances``, and
        ``danger_radius`` are read.  If ``None``, the EVP-distance trigger is
        disabled unless an explicit finite radius and distances are supplied by
        a future caller.
    cfg:
        Mapping/object with optional ``adaptive_anchors`` subsection.  Common
        keys are ``coarse_anchor_count`` (default: 2), ``coarse_anchors``,
        ``sample_omegas``, ``evp_distance_radius`` (default:
        ``band_marks.danger_radius``), ``fgmres_iter_threshold`` (default: 40),
        ``fgmres_residual_threshold`` (default: 1e-6), and an injectable
        ``iter_probe``/``fgmres_probe`` callable.  Without a callable, a cheap
        deterministic probe reads optional measured arrays/mappings from cfg and
        otherwise returns zero iterations/residual.

    Returns
    -------
    (anchors, records)
        ``anchors`` is a sorted complex-valued NumPy array.  ``records`` has one
        :class:`AnchorTriggerRecord` per final anchor, including the initial
        coarse anchors (``trigger == "coarse"``) and all adaptive insertions.
    """
    omega_grid = _sort_unique_complex(_as_1d_complex(omega_range, "omega_range"))
    initial_anchors = _initial_anchors(omega_grid, cfg)
    samples = _sample_omegas(omega_grid, cfg)
    evp_radius = _evp_radius(band_marks, cfg)
    iter_threshold = _threshold(
        _cfg_first(cfg, ("fgmres_iter_threshold", "iter_threshold", "iterations_threshold")),
        default=40.0,
        name="fgmres_iter_threshold",
    )
    residual_threshold = _threshold(
        _cfg_first(cfg, ("fgmres_residual_threshold", "residual_threshold", "relres_threshold")),
        default=1.0e-6,
        name="fgmres_residual_threshold",
    )
    min_insert_spacing = _nonnegative_float(
        _cfg_first(cfg, ("min_insert_spacing", "anchor_min_spacing"), default=0.0),
        "min_insert_spacing",
    )
    duplicate_tol = _nonnegative_float(
        _cfg_first(cfg, ("duplicate_tol", "anchor_duplicate_tol"), default=1.0e-12),
        "duplicate_tol",
    )
    max_insertions_raw = _cfg_first(cfg, ("max_insertions", "max_adaptive_insertions"), default=None)
    max_insertions = None if max_insertions_raw is None else _nonnegative_int(max_insertions_raw, "max_insertions")
    enable_evp = _as_bool(_cfg_first(cfg, ("enable_evp_distance", "use_evp_distance"), default=True))
    enable_fgmres = _as_bool(_cfg_first(cfg, ("enable_fgmres_probe", "use_fgmres_probe"), default=True))
    probe_fn = _cfg_first(cfg, ("iter_probe", "fgmres_probe"), default=None)

    records = [
        AnchorTriggerRecord(
            omega=complex(anchor),
            sample_omega=complex(anchor),
            triggers=(COARSE_TRIGGER,),
            inserted=False,
        )
        for anchor in initial_anchors
    ]
    anchors = np.asarray([record.omega for record in records], dtype=complex)
    insertions = 0

    for sample in samples:
        sample = complex(sample)
        evp_distance = _band_distance_at(sample, band_marks)
        probe = _probe_result(sample, anchors, omega_grid, cfg, probe_fn) if enable_fgmres else IterProbeResult()
        triggers: list[str] = []
        if enable_evp and np.isfinite(evp_radius) and evp_radius > 0.0 and evp_distance < evp_radius:
            triggers.append(EVP_DISTANCE_TRIGGER)
        if enable_fgmres and probe.iterations > iter_threshold:
            triggers.append(FGMRES_ITERATIONS_TRIGGER)
        if enable_fgmres and probe.residual > residual_threshold:
            triggers.append(FGMRES_RESIDUAL_TRIGGER)
        if not triggers:
            continue

        nearest_i, nearest_anchor, nearest_distance = _nearest_anchor(sample, anchors)
        spacing_tol = max(min_insert_spacing, duplicate_tol)
        if nearest_distance <= spacing_tol:
            records[nearest_i] = _annotate_record(
                records[nearest_i],
                sample=sample,
                triggers=tuple(triggers),
                evp_distance=evp_distance,
                evp_radius=evp_radius,
                probe=probe,
                nearest_anchor_before=nearest_anchor,
                nearest_distance_before=nearest_distance,
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
    anchors = np.asarray([record.omega for record in records], dtype=complex)
    return anchors, tuple(records)


def _cfg_get(cfg: Mapping[str, Any] | object | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg[key]
        nested = cfg.get("adaptive_anchors")
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
        solver = cfg.get("solver")
        router = solver.get("router") if isinstance(solver, Mapping) else None
        nested = router.get("adaptive_anchors") if isinstance(router, Mapping) else None
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    nested = getattr(cfg, "adaptive_anchors", None)
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    if nested is not None and hasattr(nested, key):
        return getattr(nested, key)
    return default


def _cfg_first(cfg: Mapping[str, Any] | object | None, keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return value
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return False
    raise ValueError(f"expected boolean-like value, got {value!r}")


def _as_1d_complex(values: Sequence[complex] | Array, name: str) -> Array:
    arr = np.asarray(values, dtype=complex).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    finite = np.isfinite(arr.real) & np.isfinite(arr.imag)
    if not bool(np.all(finite)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _sort_unique_complex(values: Array) -> Array:
    return np.sort_complex(np.unique(np.asarray(values, dtype=complex).reshape(-1)))


def _sort_records(records: Sequence[AnchorTriggerRecord]) -> list[AnchorTriggerRecord]:
    return sorted(records, key=lambda record: (record.omega.real, record.omega.imag))


def _initial_anchors(omega_grid: Array, cfg: Mapping[str, Any] | object | None) -> Array:
    raw = _cfg_first(cfg, ("coarse_anchors", "initial_anchors", "anchors"), default=None)
    if raw is not None:
        return _sort_unique_complex(_as_1d_complex(raw, "coarse_anchors"))
    count = _positive_int(
        _cfg_first(cfg, ("coarse_anchor_count", "initial_anchor_count", "anchor_count"), default=min(2, omega_grid.size)),
        "coarse_anchor_count",
    )
    count = min(count, omega_grid.size)
    return omega_grid[_coarse_indices(omega_grid, count)]


def _coarse_indices(omega_grid: Array, count: int) -> Array:
    if count == 1:
        return np.array([0], dtype=int)
    indices = np.unique(np.rint(np.linspace(0, omega_grid.size - 1, count)).astype(int))
    while indices.size < count:
        candidates = np.setdiff1d(np.arange(omega_grid.size, dtype=int), indices, assume_unique=True)
        distances = np.min(np.abs(omega_grid[candidates, None] - omega_grid[indices][None, :]), axis=1)
        chosen = int(candidates[int(np.argmax(distances))])
        indices = np.sort(np.r_[indices, chosen].astype(int))
    return indices.astype(int)


def _sample_omegas(omega_grid: Array, cfg: Mapping[str, Any] | object | None) -> Array:
    raw = _cfg_first(cfg, ("sample_omegas", "probe_omegas", "samples"), default=None)
    if raw is None:
        return omega_grid.copy()
    return _sort_unique_complex(_as_1d_complex(raw, "sample_omegas"))


def _evp_radius(band_marks: BandMarks | None, cfg: Mapping[str, Any] | object | None) -> float:
    raw = _cfg_first(cfg, ("evp_distance_radius", "distance_radius", "near_pole_radius", "danger_radius"), default=None)
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
    if omegas.size == 0:
        return float("inf")
    finite = np.isfinite(omegas.real) & np.isfinite(omegas.imag) & np.isfinite(distances)
    if not bool(np.any(finite)):
        return float("inf")
    valid_omegas = omegas[finite]
    valid_distances = distances[finite]
    return float(valid_distances[int(np.argmin(np.abs(valid_omegas - sample)))])


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
    *,
    sample: complex,
    triggers: tuple[str, ...],
    evp_distance: float,
    evp_radius: float,
    probe: IterProbeResult,
    nearest_anchor_before: complex,
    nearest_distance_before: float,
) -> AnchorTriggerRecord:
    merged = tuple(dict.fromkeys((*record.triggers, *triggers)))
    return replace(
        record,
        sample_omega=sample,
        triggers=merged,
        evp_distance=evp_distance,
        evp_radius=evp_radius,
        fgmres_iterations=probe.iterations,
        fgmres_residual=probe.residual,
        nearest_anchor_before=nearest_anchor_before,
        nearest_anchor_distance_before=nearest_distance_before,
    )


def _probe_result(
    sample: complex,
    anchors: Array,
    omega_grid: Array,
    cfg: Mapping[str, Any] | object | None,
    probe_fn: Any,
) -> IterProbeResult:
    if probe_fn is None:
        raw = _cheap_probe_from_cfg(sample, omega_grid, cfg)
    else:
        raw = _call_probe(probe_fn, sample, anchors, cfg)
    return _coerce_probe_result(raw)


def _call_probe(probe_fn: Callable[..., Any], sample: complex, anchors: Array, cfg: Mapping[str, Any] | object | None) -> Any:
    attempts = (
        (sample, anchors, cfg),
        (sample, anchors),
        (sample,),
    )
    last_error: TypeError | None = None
    for args in attempts:
        try:
            return probe_fn(*args)
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _cheap_probe_from_cfg(sample: complex, omega_grid: Array, cfg: Mapping[str, Any] | object | None) -> IterProbeResult:
    iterations = _measurement_at(
        _cfg_first(cfg, ("fgmres_iterations", "iterations", "iter_measurements"), default=None),
        sample,
        omega_grid,
        cfg,
        default=0.0,
    )
    residual = _measurement_at(
        _cfg_first(cfg, ("fgmres_residuals", "residuals", "residual_measurements"), default=None),
        sample,
        omega_grid,
        cfg,
        default=0.0,
    )
    return IterProbeResult(iterations=int(iterations), residual=float(residual))


def _measurement_at(
    raw: Any,
    sample: complex,
    omega_grid: Array,
    cfg: Mapping[str, Any] | object | None,
    *,
    default: float,
) -> float:
    if raw is None:
        return default
    if isinstance(raw, Mapping):
        if sample in raw:
            return float(raw[sample])
        if sample.imag == 0.0 and sample.real in raw:
            return float(raw[sample.real])
        keys = np.asarray([complex(key) for key in raw.keys()], dtype=complex)
        if keys.size == 0:
            return default
        key = list(raw.keys())[int(np.argmin(np.abs(keys - sample)))]
        return float(raw[key])

    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size == 0:
        return default
    measurement_omegas_raw = _cfg_first(
        cfg,
        ("fgmres_measurement_omegas", "measurement_omegas", "probe_measurement_omegas"),
        default=None,
    )
    measurement_omegas = omega_grid if measurement_omegas_raw is None else _as_1d_complex(measurement_omegas_raw, "measurement_omegas")
    if measurement_omegas.size != values.size:
        if omega_grid.size == values.size:
            measurement_omegas = omega_grid
        else:
            raise ValueError("FGMRES measurement arrays must align with measurement_omegas or omega_range")
    return float(values[int(np.argmin(np.abs(measurement_omegas - sample)))])


def _coerce_probe_result(raw: IterProbeResult | Mapping[str, Any] | Sequence[Any] | None) -> IterProbeResult:
    if raw is None:
        return IterProbeResult()
    if isinstance(raw, IterProbeResult):
        result = raw
    elif isinstance(raw, Mapping):
        iterations = raw.get("iterations", raw.get("iters", raw.get("fgmres_iterations", 0)))
        residual = raw.get("residual", raw.get("rel_residual", raw.get("fgmres_residual", 0.0)))
        result = IterProbeResult(iterations=int(iterations), residual=float(residual))
    else:
        values = list(raw)
        if len(values) < 2:
            raise ValueError("iter_probe sequence result must contain (iterations, residual)")
        result = IterProbeResult(iterations=int(values[0]), residual=float(values[1]))
    if result.iterations < 0:
        raise ValueError("iter_probe returned negative iterations")
    if not np.isfinite(result.residual) or result.residual < 0.0:
        raise ValueError("iter_probe returned a non-finite or negative residual")
    return result


__all__ = [
    "AnchorTriggerRecord",
    "COARSE_TRIGGER",
    "EVP_DISTANCE_TRIGGER",
    "FGMRES_ITERATIONS_TRIGGER",
    "FGMRES_RESIDUAL_TRIGGER",
    "IterProbeResult",
    "choose_anchors_adaptive",
]
