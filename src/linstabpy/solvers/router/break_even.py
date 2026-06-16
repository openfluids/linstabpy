# SPDX-License-Identifier: GPL-3.0-only
"""Measured frequency-count break-even routing for future resolvent maps.

The chooser compares direct per-frequency sparse factorizations with an
anchor-FGMRES sweep using measured costs on the supplied operator pattern:

``direct cost   ≈ N_f · T_fact``
``sweep cost    ≈ K_anchors · T_fact + N_f · iters · T_apply``
``N_f*          = K_anchors / (1 - iters · T_apply / T_fact)``

``T_fact`` is measured by one real SciPy ``splu`` factorization on fabricated,
well-conditioned values sharing the supplied sparse pattern.  ``T_apply`` is
measured by one representative solve with that factor.  If
``iters · T_apply >= T_fact``, the sweep cannot amortize its anchor
factorizations, so the decision remains ``"direct"`` for every frequency count.
This module is intentionally additive: it exposes a pre-routing decision for a
future entry point and does not wire itself into any solver path.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .ram_autoselect import _DEFAULT_REORDER, _SPLU_REORDER, _as_csc_pattern, _well_conditioned_matrix_with_pattern

RoutingPath = Literal["direct", "fgmres-sweep"]
Timer = Callable[[], float]


@dataclass(frozen=True)
class BreakEvenProbeResult:
    """Measured cost ingredients for :func:`break_even_decision`."""

    t_fact_seconds: float
    t_apply_seconds: float
    iterations: int
    n: int
    nnz: int
    reorder: str
    backend: str = "scipy-splu"


@dataclass(frozen=True)
class RoutingDecision:
    """Cost-routing decision for a frequency sweep."""

    path: RoutingPath
    crossover_nf: float
    t_fact_seconds: float
    t_apply_seconds: float
    iterations: int
    k_anchors: int
    n_frequencies: int
    note: str


def break_even_decision(
    L0_pattern: sp.spmatrix | Any,
    n_frequencies: int,
    cfg: Mapping[str, Any] | object | None,
) -> RoutingDecision:
    """Choose ``"direct"`` or ``"fgmres-sweep"`` from a measured cost model.

    The production path runs a real sparse-LU timing probe on ``L0_pattern``.
    Tests may inject ``break_even_probe`` in ``cfg`` to make the measurement
    deterministic; the injected probe must still return the measured quantities
    that drive the crossover formula.
    """
    pattern = _as_csc_pattern(L0_pattern)
    n_freq = _positive_int(n_frequencies, "n_frequencies")
    k_anchors = _k_anchors(cfg)

    probe_fn = _cfg_get(cfg, "break_even_probe", None)
    if probe_fn is None:
        probe = _timing_probe(pattern, cfg)
    else:
        probe = probe_fn(pattern, cfg)
    if not isinstance(probe, BreakEvenProbeResult):
        raise TypeError("break_even_probe must return BreakEvenProbeResult")
    _validate_probe(probe)

    variable_sweep_cost = probe.iterations * probe.t_apply_seconds
    if variable_sweep_cost >= probe.t_fact_seconds:
        crossover = float("inf")
        path: RoutingPath = "direct"
        reason = "sweep never amortizes because iters*T_apply >= T_fact"
    else:
        ratio = variable_sweep_cost / probe.t_fact_seconds
        crossover = float(k_anchors / (1.0 - ratio))
        path = "fgmres-sweep" if n_freq >= crossover else "direct"
        reason = (
            "n_frequencies is at/above measured crossover"
            if path == "fgmres-sweep"
            else "n_frequencies is below measured crossover"
        )

    note = (
        f"measured break-even via {probe.backend} probe "
        f"(n={probe.n}, nnz={probe.nnz}, reorder={probe.reorder}); "
        f"T_fact={probe.t_fact_seconds:.17g} s, "
        f"T_apply={probe.t_apply_seconds:.17g} s, "
        f"iters={probe.iterations}, K_anchors={k_anchors}, "
        f"crossover_nf={crossover:.17g}; path={path}: {reason}"
    )
    return RoutingDecision(
        path=path,
        crossover_nf=crossover,
        t_fact_seconds=probe.t_fact_seconds,
        t_apply_seconds=probe.t_apply_seconds,
        iterations=probe.iterations,
        k_anchors=k_anchors,
        n_frequencies=n_freq,
        note=note,
    )


def _timing_probe(pattern: sp.spmatrix, cfg: Mapping[str, Any] | object | None) -> BreakEvenProbeResult:
    """Measure one sparse-LU factorization and one representative apply."""
    pattern = _as_csc_pattern(pattern)
    reorder = str(_cfg_get(cfg, "probe_reorder", _cfg_get(cfg, "reorder", _DEFAULT_REORDER))).strip().lower()
    if reorder not in _SPLU_REORDER:
        reorder = _DEFAULT_REORDER
    timer = _timer(cfg)
    probe_matrix = _well_conditioned_matrix_with_pattern(pattern)

    fact_start = _timer_value(timer, "timer")
    try:
        lu = spla.splu(probe_matrix, permc_spec=_SPLU_REORDER[reorder])
    except RuntimeError as exc:
        raise RuntimeError("break-even sparse-LU timing probe failed for supplied pattern") from exc
    fact_end = _timer_value(timer, "timer")
    t_fact = fact_end - fact_start

    rhs = _representative_rhs(pattern.shape[0])
    apply_start = _timer_value(timer, "timer")
    solution = lu.solve(rhs)
    apply_end = _timer_value(timer, "timer")
    t_apply = apply_end - apply_start
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("break-even timing probe produced a non-finite representative solve")

    result = BreakEvenProbeResult(
        t_fact_seconds=float(t_fact),
        t_apply_seconds=float(t_apply),
        iterations=_representative_iterations(cfg),
        n=int(pattern.shape[0]),
        nnz=int(pattern.nnz),
        reorder=reorder,
    )
    _validate_probe(result)
    return result


def _cfg_get(cfg: Mapping[str, Any] | object | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg[key]
        nested = cfg.get("break_even")
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
        return default
    nested = getattr(cfg, "break_even", None)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    return getattr(nested, key, default) if nested is not None else default


def _cfg_first(cfg: Mapping[str, Any] | object | None, keys: Sequence[str]) -> Any:
    for key in keys:
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return value
    return None


def _k_anchors(cfg: Mapping[str, Any] | object | None) -> int:
    raw = _cfg_first(cfg, ("k_anchors", "K_anchors", "n_anchors"))
    if raw is not None:
        return _positive_int(raw, "k_anchors")
    anchors = _cfg_get(cfg, "anchors", None)
    if anchors is not None:
        try:
            return _positive_int(len(anchors), "anchors")
        except TypeError as exc:
            raise ValueError("anchors must be a sized collection when k_anchors is not provided") from exc
    return 1


def _representative_iterations(cfg: Mapping[str, Any] | object | None) -> int:
    raw = _cfg_first(
        cfg,
        (
            "representative_iterations",
            "representative_iters",
            "fgmres_iterations",
            "fgmres_iters",
            "iterations",
            "iters",
        ),
    )
    if raw is None:
        return 1
    return _nonnegative_int(raw, "representative_iterations")


def _timer(cfg: Mapping[str, Any] | object | None) -> Timer:
    timer = _cfg_get(cfg, "timer", time.perf_counter)
    if not callable(timer):
        raise TypeError("timer must be callable")
    return timer


def _timer_value(timer: Timer, name: str) -> float:
    try:
        value = float(timer())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must return a finite float") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must return a finite float")
    return value


def _representative_rhs(n: int) -> np.ndarray:
    index = np.arange(int(n), dtype=float)
    return (1.0 + 0.01 * index) + 1j * (0.5 - 0.005 * index)


def _validate_probe(probe: BreakEvenProbeResult) -> None:
    if not np.isfinite(probe.t_fact_seconds) or probe.t_fact_seconds <= 0.0:
        raise ValueError("break-even probe returned non-positive T_fact")
    if not np.isfinite(probe.t_apply_seconds) or probe.t_apply_seconds <= 0.0:
        raise ValueError("break-even probe returned non-positive T_apply")
    _nonnegative_int(probe.iterations, "iterations")
    _positive_int(probe.n, "n")
    _positive_int(probe.nnz, "nnz")
    if not str(probe.reorder).strip():
        raise ValueError("break-even probe returned an empty reorder label")
    if not str(probe.backend).strip():
        raise ValueError("break-even probe returned an empty backend label")


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


__all__ = ["BreakEvenProbeResult", "RoutingDecision", "break_even_decision"]
