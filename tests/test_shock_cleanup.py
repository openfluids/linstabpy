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


def _shock_like_fields(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = 1.0 - 0.5 * (1.0 + np.tanh((X - 0.55) / 0.02))
    v = np.zeros_like(u)
    rho = 1.0 + 0.02 * np.sin(24.0 * np.pi * X) * np.cos(8.0 * np.pi * Y)
    w = np.zeros_like(u)
    T = 1.0 + 0.01 * np.cos(18.0 * np.pi * X)
    sponge = np.zeros_like(u)
    return rho, u, v, w, T, sponge


def test_shock_cleanup_rejects_bad_config_and_filter_lists() -> None:
    X, Y = _regular_grid(8, 6)
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    apply_optional_shock_cleanup(X, Y, rho, u, v, w, T, sponge, None)
    with pytest.raises(TypeError, match="JSON object"):
        apply_optional_shock_cleanup(X, Y, rho, u, v, w, T, sponge, "on")
    bad_filter = {"enabled": True, "sensor": {"enabled": True}, "filter": {"enabled": True}}
    with pytest.raises(ValueError, match="directions"):
        apply_optional_shock_cleanup(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {**bad_filter, "filter": {"enabled": True, "directions": []}},
        )
    with pytest.raises(ValueError, match="fields"):
        apply_optional_shock_cleanup(
            X, Y, rho, u, v, w, T, sponge, {**bad_filter, "filter": {"enabled": True, "fields": []}}
        )
    with pytest.raises(ValueError, match="only"):
        apply_optional_shock_cleanup(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {**bad_filter, "filter": {"enabled": True, "fields": ["pressure"]}},
        )


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

    *_add_fields, sponge_add, info_add = apply_optional_shock_cleanup(
        X, Y, rho, u, v, w, T, base_sponge, cfg_add
    )
    *_max_fields, sponge_max, info_max = apply_optional_shock_cleanup(
        X, Y, rho, u, v, w, T, base_sponge, cfg_max
    )

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


def test_shock_cleanup_rejects_filter_without_sensor() -> None:
    X, Y = _regular_grid(8, 6)
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    with pytest.raises(ValueError, match="sensor.enabled"):
        apply_optional_shock_cleanup(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {"enabled": True, "sensor": {"enabled": False}, "filter": {"enabled": True}},
        )


def test_shock_cleanup_rejects_duplicate_and_unknown_lists() -> None:
    from linstabpy.baseflow.shock_cleanup import (
        _binomial5_filter,
        _normalize_directions,
        _normalize_fields,
    )

    assert _normalize_directions(["x", "x", "y"]) == [1, 0]
    assert _normalize_fields(["rho", "rho", "T"]) == ["rho", "T"]
    with pytest.raises(ValueError, match="only 'x'"):
        _normalize_directions(["z"])
    field = np.ones((4, 5))
    with pytest.raises(ValueError, match="2-D"):
        _binomial5_filter(np.ones(4), axis=0)
    with pytest.raises(ValueError, match="axis"):
        _binomial5_filter(field, axis=2)


def test_binomial5_kernel_is_1_4_6_4_1_over_16() -> None:
    """Compact binomial smoother: constants and linears stay put; impulse is 1-4-6-4-1."""
    from linstabpy.baseflow.shock_cleanup import _binomial5_filter

    constant = np.full((7, 6), 2.5)
    np.testing.assert_allclose(_binomial5_filter(constant, axis=0), constant)
    np.testing.assert_allclose(_binomial5_filter(constant, axis=1), constant)

    y = np.arange(7.0)[:, None]
    x = np.arange(6.0)[None, :]
    linear = 3.0 * y + 0.5 * x
    # Edge pad (mode="edge") is not a linear extension, so only the interior
    # two cells in from the filtered axis must stay on the plane.
    np.testing.assert_allclose(_binomial5_filter(linear, axis=0)[2:-2, :], linear[2:-2, :])
    np.testing.assert_allclose(_binomial5_filter(linear, axis=1)[:, 2:-2], linear[:, 2:-2])

    impulse = np.zeros((9, 7))
    impulse[4, 3] = 1.0
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    out0 = _binomial5_filter(impulse, axis=0)
    np.testing.assert_allclose(out0[2:7, 3], kernel)
    np.testing.assert_allclose(out0[:, :3], 0.0)
    np.testing.assert_allclose(out0[:, 4:], 0.0)
    out1 = _binomial5_filter(impulse, axis=1)
    np.testing.assert_allclose(out1[4, 1:6], kernel)
    np.testing.assert_allclose(out1[:4, :], 0.0)
    np.testing.assert_allclose(out1[5:, :], 0.0)


def test_compression_only_sensor_masks_on_divergence() -> None:
    X, Y = _regular_grid()
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    cfg = {
        "enabled": True,
        "sensor": {
            "enabled": True,
            "kind": "compression_only",
            "compression_threshold": -5.0,
            "buffer_cells": 1,
            "blend_sigma": 1.0,
        },
        "filter": {"enabled": False},
        "sponge": {"enabled": False},
    }
    *_fields, info = apply_optional_shock_cleanup(X, Y, rho, u, v, w, T, sponge, cfg)
    assert info["sensor"]["kind"] == "compression_only"
    assert info["sensor"]["mask_fraction"] > 0.0
    assert float(np.max(info["preserve_weight"])) == pytest.approx(1.0)


def test_shock_cleanup_loud_fails_on_sensor_and_sponge_numbers() -> None:
    X, Y = _regular_grid(8, 6)
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    base = {"enabled": True, "sensor": {"enabled": True}, "filter": {"enabled": False}}
    with pytest.raises(ValueError, match="sensor.kind"):
        apply_optional_shock_cleanup(
            X, Y, rho, u, v, w, T, sponge, {**base, "sensor": {"enabled": True, "kind": "unknown"}}
        )
    with pytest.raises(ValueError, match="buffer_cells"):
        apply_optional_shock_cleanup(
            X, Y, rho, u, v, w, T, sponge, {**base, "sensor": {"enabled": True, "buffer_cells": -1}}
        )
    with pytest.raises(ValueError, match="blend_sigma"):
        apply_optional_shock_cleanup(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {**base, "sensor": {"enabled": True, "blend_sigma": -0.1}},
        )
    with pytest.raises(ValueError, match="amplitude"):
        apply_optional_shock_cleanup(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {**base, "sponge": {"enabled": True, "amplitude": -1.0}},
        )
    with pytest.raises(ValueError, match="mode"):
        apply_optional_shock_cleanup(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {**base, "sponge": {"enabled": True, "amplitude": 1.0, "mode": "min"}},
        )
    with pytest.raises(ValueError, match="smooth_sigma"):
        apply_optional_shock_cleanup(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {**base, "sponge": {"enabled": True, "amplitude": 1.0, "smooth_sigma": -0.2}},
        )
    with pytest.raises(ValueError, match="passes"):
        apply_optional_shock_cleanup(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {**base, "filter": {"enabled": True, "passes": -1}},
        )


def test_shock_filter_zero_passes_and_zero_amplitude_sponge_are_noops() -> None:
    X, Y = _regular_grid()
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    cfg = {
        "enabled": True,
        "sensor": {"enabled": True, "kind": "compression_ducros", "compression_threshold": -5.0},
        "filter": {"enabled": True, "passes": 0, "fields": ["rho"]},
        "sponge": {"enabled": True, "amplitude": 0.0, "smooth_sigma": 1.0},
    }
    rho_out, _u, _v, _w, _T, sponge_out, info = apply_optional_shock_cleanup(
        X, Y, rho, u, v, w, T, sponge, cfg
    )
    assert_allclose(rho_out, rho)
    assert_allclose(sponge_out, sponge)
    assert info["filter"]["passes"] == 0


def test_shock_sponge_smooths_when_sigma_is_positive() -> None:
    X, Y = _regular_grid()
    rho, u, v, w, T, sponge = _shock_like_fields(X, Y)
    cfg = {
        "enabled": True,
        "sensor": {
            "enabled": True,
            "kind": "compression_ducros",
            "compression_threshold": -5.0,
            "buffer_cells": 2,
        },
        "filter": {"enabled": False},
        "sponge": {"enabled": True, "amplitude": 2.0, "mode": "add", "smooth_sigma": 1.5},
    }
    *_fields, sponge_out, info = apply_optional_shock_cleanup(X, Y, rho, u, v, w, T, sponge, cfg)
    assert info["sponge"]["enabled"] is True
    assert float(np.max(sponge_out)) > 0.0
