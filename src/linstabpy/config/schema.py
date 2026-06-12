# SPDX-License-Identifier: GPL-3.0-only
"""Lenient schema checks for fully merged JSONC configuration dictionaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ConfigValidationError(ValueError):
    pass


TOP_LEVEL_ALLOWED = frozenset(
    {
        "defaults",
        "cases",
        "outputs",
        "$include",
        "output_root",
        "base_config",
        "bc",
        "solver",
        "variants",
        "fd_order",
        "alpha_filter",
        "assembly_backend",
        "n_modes",
        "model",
        "nx",
        "ny",
        "x_periodicity",
        "sponge",
        "memory_guard",
        "y_symmetry",
        "paths",
        "physics",
        "numerics",
        "analysis",
        "Re",
        "Ma",
        "Pr",
        "kappa",
        "T_0",
        "omega",
        "forcing_zero_vars",
        "reduced_overrides",
        "St",
        "m",
        "omega_target",
        "ncv",
        "beta",
        "dx",
        "x_min_factor",
        "x_max_base",
        "y_i",
        "y_max",
        "kx",
        "xrange",
        "yrange",
        "Lx",
        "Ly",
        "parity_tolerances",
        "examples",
        "trials",
        "assembly_backends",
        "coarsen",
        "x_cavity_start",
        "L_cavity",
        "mask_x_left",
        "mask_x_right",
        "deform",
        "omega_shift_fallback",
        "use_omega_shift_fallback",
        "conformal_scale",
        "y_stretch_yi",
        "temporal_targets",
        "temporal_quotas",
        "alpha_temp",
        "n_periods",
        "dedupe_tol",
        "spatial_omega_imag_shift",
        "run_temporal",
        "run_spatial",
        "n_modes_stability",
        "stability_search_extra_modes",
        "n_modes_resolvent",
        "run_stability",
        "run_resolvent",
        "stride_ksi",
        "stride_eta",
        "penalty",
        "baseflow_input",
        "baseflow_preprocess",
        "omega_targets",
        "direct_ksp",
        "factor_precision",
        "schema_version",
    }
)

_ALLOWED_FD_ORDERS = frozenset({2, 4, 6, 8})

_INT_FIELDS = frozenset(
    {
        "n_modes",
        "ncv",
        "nx",
        "ny",
        "m",
        "coarsen",
        "n_periods",
        "x_min_factor",
        "n_modes_stability",
        "stability_search_extra_modes",
        "n_modes_resolvent",
        "stride_ksi",
        "stride_eta",
    }
)

_NUMERIC_FIELDS = frozenset(
    {
        "Re",
        "Ma",
        "Pr",
        "kappa",
        "T_0",
        "St",
        "beta",
        "omega",
        "dx",
        "kx",
        "Lx",
        "Ly",
        "x_max_base",
        "y_i",
        "y_max",
        "penalty",
        "alpha_temp",
        "dedupe_tol",
        "conformal_scale",
        "y_stretch_yi",
        "omega_shift_fallback",
        "spatial_omega_imag_shift",
    }
)

_BOOL_FIELDS = frozenset(
    {
        "x_periodicity",
        "y_symmetry",
        "run_temporal",
        "run_spatial",
        "run_stability",
        "run_resolvent",
        "use_omega_shift_fallback",
    }
)

_STRING_FIELDS = frozenset(
    {
        "model",
        "assembly_backend",
        "output_root",
        "base_config",
        "factor_precision",
        "schema_version",
        "$include",
    }
)


def validate_config(cfg: dict, *, source: str | None = None) -> None:
    """Validate known config hazards without requiring any optional fields."""
    violations: list[str] = []

    for key in cfg:
        if key not in TOP_LEVEL_ALLOWED:
            violations.append(f"unknown top-level key {key!r} (possible typo)")

    for path, value in _iter_fd_order_values(cfg):
        if isinstance(value, bool) or not isinstance(value, int) or value not in _ALLOWED_FD_ORDERS:
            allowed = ", ".join(str(order) for order in sorted(_ALLOWED_FD_ORDERS))
            violations.append(f"{path} = {value!r} (allowed: {allowed})")

    for key in sorted(_INT_FIELDS & cfg.keys()):
        value = cfg[key]
        if isinstance(value, bool):
            violations.append(f"{key}: expected int, got bool")
        elif not isinstance(value, int):
            violations.append(f"{key}: expected int, got {type(value).__name__}")

    for key in sorted(_NUMERIC_FIELDS & cfg.keys()):
        value = cfg[key]
        if isinstance(value, bool):
            violations.append(f"{key}: expected int or float, got bool")
        elif not isinstance(value, (int, float)):
            violations.append(f"{key}: expected int or float, got {type(value).__name__}")

    for key in sorted(_BOOL_FIELDS & cfg.keys()):
        value = cfg[key]
        if not isinstance(value, bool):
            violations.append(f"{key}: expected bool, got {type(value).__name__}")

    for key in sorted(_STRING_FIELDS & cfg.keys()):
        value = cfg[key]
        if not isinstance(value, str):
            violations.append(f"{key}: expected str, got {type(value).__name__}")

    if violations:
        prefix = f"{source}: " if source else ""
        raise ConfigValidationError(prefix + "; ".join(violations))


def _iter_fd_order_values(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else str(key)
            if key == "fd_order":
                yield item_path, item
            yield from _iter_fd_order_values(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_fd_order_values(item, f"{path}[{index}]")
