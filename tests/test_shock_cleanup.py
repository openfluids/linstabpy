# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from linstabpy.baseflow.shock_cleanup import apply_optional_shock_cleanup


def _regular_grid(nx: int = 81, ny: int = 41) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx, dtype=float)
    y = np.linspace(0.0, 0.25, ny, dtype=float)
    return np.meshgrid(x, y)


def _shock_like_fields(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = 1.0 - 0.5 * (1.0 + np.tanh((X - 0.55) / 0.02))
    v = np.zeros_like(u)
    rho = 1.0 + 0.02 * np.sin(24.0 * np.pi * X) * np.cos(8.0 * np.pi * Y)
    w = np.zeros_like(u)
    T = 1.0 + 0.01 * np.cos(18.0 * np.pi * X)
    sponge = np.zeros_like(u)
    return rho, u, v, w, T, sponge


def test_shock_cleanup_disabled_is_noop() -> None:
    X, Y = _regular_grid()
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)

    rho_out, u_out, v_out, w_out, T_out, sponge_out, info = apply_optional_shock_cleanup(
        X, Y, rho, u, v, w, T, sponge, {"enabled": False}
    )

    assert info["enabled"] is False
    assert info["summary"] == "disabled"
    assert_allclose(rho_out, rho)
    assert_allclose(u_out, u)
    assert_allclose(v_out, v)
    assert_allclose(w_out, w)
    assert_allclose(T_out, T)
    assert_allclose(sponge_out, sponge)


def test_shock_cleanup_detects_and_buffers_shock_region() -> None:
    X, Y = _regular_grid()
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    cfg = {
        "enabled": True,
        "sensor": {
            "enabled": True,
            "kind": "compression_ducros",
            "compression_threshold": -5.0,
            "ducros_threshold": 0.8,
            "buffer_cells": 2,
        },
        "filter": {"enabled": False},
        "sponge": {"enabled": False},
    }

    *_fields, info = apply_optional_shock_cleanup(X, Y, rho, u, v, w, T, sponge, cfg)

    assert info["sensor"]["enabled"] is True
    assert info["sensor"]["mask_fraction"] > 0.0
    assert info["sensor"]["buffered_mask_fraction"] >= info["sensor"]["mask_fraction"]
    assert np.count_nonzero(info["buffered_mask"]) >= np.count_nonzero(info["mask"])


def test_masked_filter_preserves_shock_core_and_changes_outer_field() -> None:
    X, Y = _regular_grid()
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    cfg = {
        "enabled": True,
        "sensor": {
            "enabled": True,
            "kind": "compression_ducros",
            "compression_threshold": -5.0,
            "ducros_threshold": 0.8,
            "buffer_cells": 3,
            "blend_sigma": 0.0,
        },
        "filter": {
            "enabled": True,
            "passes": 2,
            "directions": ["x", "y"],
            "fields": ["rho", "u", "T"],
        },
        "sponge": {"enabled": False},
    }

    rho_out, u_out, _v_out, _w_out, T_out, _sponge_out, info = apply_optional_shock_cleanup(
        X, Y, rho, u, v, w, T, sponge, cfg
    )

    mask = np.asarray(info["buffered_mask"], dtype=bool)
    assert np.any(mask)
    assert_allclose(rho_out[mask], rho[mask], atol=1e-12)
    assert_allclose(u_out[mask], u[mask], atol=1e-12)
    assert_allclose(T_out[mask], T[mask], atol=1e-12)
    assert info["filter"]["relative_l2_change"]["rho"] > 0.0
    assert not np.allclose(rho_out, rho)


def test_shock_sponge_merge_modes_are_isolated() -> None:
    X, Y = _regular_grid()
    rho, u, v, w, T, _unused_sponge = _shock_like_fields(X, Y)
    base_sponge = np.full_like(X, 0.5)
    sensor = {
        "enabled": True,
        "kind": "compression_ducros",
        "compression_threshold": -5.0,
        "ducros_threshold": 0.8,
        "buffer_cells": 2,
    }

    cfg_add = {
        "enabled": True,
        "sensor": sensor,
        "filter": {"enabled": False},
        "sponge": {"enabled": True, "amplitude": 1.0, "mode": "add"},
    }
    cfg_max = {
        "enabled": True,
        "sensor": sensor,
        "filter": {"enabled": False},
        "sponge": {"enabled": True, "amplitude": 1.0, "mode": "max"},
    }

    *_add_fields, sponge_add, info_add = apply_optional_shock_cleanup(X, Y, rho, u, v, w, T, base_sponge, cfg_add)
    *_max_fields, sponge_max, info_max = apply_optional_shock_cleanup(X, Y, rho, u, v, w, T, base_sponge, cfg_max)

    assert info_add["sponge"]["enabled"] is True
    assert info_max["sponge"]["enabled"] is True
    assert np.max(sponge_add) > np.max(sponge_max)
    assert np.min(sponge_max) >= 0.5


def test_modified_ducros_placeholder_raises() -> None:
    X, Y = _regular_grid()
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    cfg = {
        "enabled": True,
        "sensor": {"enabled": True, "kind": "modified_ducros"},
        "filter": {"enabled": False},
        "sponge": {"enabled": False},
    }

    with pytest.raises(NotImplementedError):
        apply_optional_shock_cleanup(X, Y, rho, u, v, w, T, sponge, cfg)
