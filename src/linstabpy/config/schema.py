# SPDX-License-Identifier: GPL-3.0-only
"""Lenient schema checks for fully merged YAML case dictionaries."""

from __future__ import annotations

from collections.abc import Iterable
from difflib import get_close_matches
from typing import Any


class ConfigValidationError(ValueError):
    pass


# Nested blocks plus the Martini mesh keys that generated-mesh cases still
# keep at the top level. Flat Song keys stay until that driver has a wrapper.
TOP_LEVEL_ALLOWED = frozenset(
    {
        "paths",
        "physics",
        "numerics",
        "analysis",
        "preprocess",
        "bc",
        "solver",
        "cases",
        "schema_version",
        "fd_order",
        "alpha_filter",
        "assembly_backend",
        "n_modes",
        "model",
        "nx",
        "ny",
        "x_periodicity",
        "sponge",
        "y_symmetry",
        "Re",
        "Ma",
        "Pr",
        "kappa",
        "T_0",
        "omega",
        "forcing_zero_vars",
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
        # Flat Martini example keys (standalone drivers). floquet_shift is Ex. 3's
        # counterpart of numerics.floquet_shift in driver/run.py ("continuum" |
        # "matlab_parity"); zero_dmudT (Ex. 4) and zero_dmudT_martini_parity (Ex. 1)
        # zero dmu/dT after sutherland_air for LinStab2D 2-D parity; bc_mask_layout
        # (Ex. 2) is the observation-mask layout ("kron" | "tile").
        "floquet_shift",
        "zero_dmudT",
        "zero_dmudT_martini_parity",
        "bc_mask_layout",
    }
)

_ALLOWED_FD_ORDERS = frozenset({2, 4, 6, 8})

# Normalized `floquet_shift` keys accepted by
# `operators.assembly._resolve_floquet_shift_mode` (lowercased, `-` -> `_`).
_ALLOWED_FLOQUET_SHIFT_KEYS = frozenset(
    {"matlab", "matlab_parity", "parity", "linstab2d", "continuum", "correct", "physics"}
)

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
        "mumps_blr_tol",
    }
)

_BOOL_FIELDS = frozenset(
    {
        "zero_dmudT",
        "zero_dmudT_martini_parity",
        "x_periodicity",
        "y_symmetry",
        "run_temporal",
        "run_spatial",
        "run_stability",
        "run_resolvent",
        "use_omega_shift_fallback",
        "mumps_blr",
    }
)

_STRING_FIELDS = frozenset(
    {
        "model",
        "bc_mask_layout",
        "assembly_backend",
        "output_root",
        "base_config",
        "factor_precision",
        "mumps_ordering",
        "schema_version",
        "$include",
    }
)


def validate_config(cfg: dict, *, source: str | None = None) -> None:
    """Validate known config hazards without requiring any optional fields."""
    violations: list[str] = []

    for key in cfg:
        if key not in TOP_LEVEL_ALLOWED:
            hint = ""
            if isinstance(key, str):
                match = get_close_matches(key, sorted(TOP_LEVEL_ALLOWED), n=1, cutoff=0.6)
                if match:
                    hint = f"; did you mean {match[0]!r}?"
            violations.append(f"unknown top-level key {key!r} (possible typo{hint})")

    for path, value in _iter_fd_order_values(cfg):
        if isinstance(value, bool) or not isinstance(value, int) or value not in _ALLOWED_FD_ORDERS:
            allowed = ", ".join(str(order) for order in sorted(_ALLOWED_FD_ORDERS))
            violations.append(f"{path} = {value!r} (allowed: {allowed})")

    for path, value in _iter_floquet_shift_values(cfg):
        if not isinstance(value, str):
            violations.append(
                f"{path} = {value!r} (expected str, one of 'matlab_parity' or 'continuum')"
            )
        elif value.strip().lower().replace("-", "_") not in _ALLOWED_FLOQUET_SHIFT_KEYS:
            violations.append(
                f"{path} = {value!r} (unknown mode; expected 'matlab_parity' or 'continuum')"
            )

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


def _iter_floquet_shift_values(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else str(key)
            if key == "floquet_shift":
                yield item_path, item
            yield from _iter_floquet_shift_values(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_floquet_shift_values(item, f"{path}[{index}]")
