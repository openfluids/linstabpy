# SPDX-License-Identifier: GPL-3.0-only
"""Packed-field loader contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples.song.reference_paths import BASEFLOW_NPZ
from linstabpy.baseflow import load_case_fields
from linstabpy.baseflow.preprocess import _binomial5_filter, apply_preprocess


def _write_fields(root: Path, *, nx: int = 8, ny: int = 6) -> Path:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 0.5, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    packed = root / "fields.npz"
    np.savez_compressed(
        packed,
        x=X,
        y=Y,
        rho=np.ones_like(X),
        u=2.15 * np.ones_like(X),
        v=0.1 * np.ones_like(X),
        w=0.2 * np.ones_like(X),
        T=np.ones_like(X),
        sponge=np.zeros_like(X),
    )
    return packed


def test_load_case_fields_reads_npz(tmp_path: Path) -> None:
    packed = _write_fields(tmp_path)
    X, Y, bf, sponge, info = load_case_fields(
        packed, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33
    )
    assert X.shape == (6, 8)
    assert np.allclose(bf.U, 2.15)
    assert info["summary"] == "disabled"
    assert sponge.shape == X.shape


def test_load_case_fields_zero_spanwise(tmp_path: Path) -> None:
    packed = _write_fields(tmp_path)
    _X, _Y, bf, _sponge, _info = load_case_fields(
        packed,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        zero_spanwise=True,
    )
    assert np.allclose(bf.W, 0.0)


def test_load_case_fields_stride_and_transpose(tmp_path: Path) -> None:
    packed = _write_fields(tmp_path, nx=8, ny=6)
    X, _Y, bf, _sponge, _info = load_case_fields(
        packed,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        transpose_input=True,
        stride_ksi=2,
        stride_eta=3,
    )
    assert X.shape == (3, 3)
    assert bf.U.shape == X.shape


def test_load_case_fields_boundary_sponge(tmp_path: Path) -> None:
    packed = _write_fields(tmp_path)
    _X, _Y, _bf, sponge, info = load_case_fields(
        packed,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        preprocess={
            "enabled": True,
            "sponge": {
                "scale": 1.0,
                "boundary": {
                    "enabled": True,
                    "mode": "add",
                    "points": [2, 0, 0, 0],
                    "amplitudes": [1.0, 0.0, 0.0, 0.0],
                },
            },
        },
    )
    assert float(sponge.max()) > 0.0
    assert "boundary_sponge" in " ".join(info["applied"])
    assert float(sponge[-1, 3]) > 0.0
    assert float(sponge[0, 3]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("points", "edge_check"),
    [
        ([0, 2, 0, 0], lambda s: (float(s[2, -1]) > 0.0, float(s[2, 0]) == pytest.approx(0.0))),
        ([0, 0, 2, 0], lambda s: (float(s[0, 3]) > 0.0, float(s[-1, 3]) == pytest.approx(0.0))),
        ([0, 0, 0, 2], lambda s: (float(s[2, 0]) > 0.0, float(s[2, -1]) == pytest.approx(0.0))),
    ],
)
def test_load_case_fields_boundary_sponge_each_edge(tmp_path: Path, points, edge_check) -> None:
    packed = _write_fields(tmp_path)
    _X, _Y, _bf, sponge, info = load_case_fields(
        packed,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        preprocess={
            "enabled": True,
            "sponge": {
                "scale": 1.0,
                "boundary": {
                    "enabled": True,
                    "mode": "add",
                    "points": points,
                    "amplitudes": [1.0, 1.0, 1.0, 1.0],
                },
            },
        },
    )
    on_edge, off_edge = edge_check(sponge)
    assert on_edge
    assert off_edge
    assert "boundary_sponge" in " ".join(info["applied"])


def test_load_case_fields_boundary_sponge_rejects_bad_points(tmp_path: Path) -> None:
    packed = _write_fields(tmp_path)
    with pytest.raises(ValueError, match="four values"):
        load_case_fields(
            packed,
            Re=1.0e5,
            Ma=2.15,
            Pr=0.72,
            kappa=1.4,
            T_0=153.33,
            preprocess={
                "enabled": True,
                "sponge": {
                    "boundary": {
                        "enabled": True,
                        "points": [2, 0, 0],
                        "amplitudes": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            },
        )


def test_load_case_fields_filter_rejects_unknown_direction(tmp_path: Path) -> None:
    packed = _write_fields(tmp_path)
    with pytest.raises(ValueError, match="x' and/or 'y"):
        load_case_fields(
            packed,
            Re=1.0e5,
            Ma=2.15,
            Pr=0.72,
            kappa=1.4,
            T_0=153.33,
            preprocess={
                "enabled": True,
                "filter": {"enabled": True, "passes": 1, "fields": ["u"], "directions": ["z"]},
            },
        )


def test_load_case_fields_filter_changes_u(tmp_path: Path) -> None:
    packed = _write_fields(tmp_path)
    data = np.load(packed)
    noisy = dict(data)
    noisy["u"] = data["u"] + 0.1 * np.sin(20.0 * data["x"])
    noisy_path = tmp_path / "noisy.npz"
    np.savez_compressed(noisy_path, **noisy)
    _X, _Y, bf_raw, _s, _i = load_case_fields(
        noisy_path, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33
    )
    _X2, _Y2, bf_filt, _s2, info = load_case_fields(
        noisy_path,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        preprocess={
            "enabled": True,
            "filter": {"enabled": True, "passes": 2, "fields": ["u"], "directions": ["x", "y"]},
        },
    )
    assert info["enabled"] is True
    assert "explicit_filter" in info["summary"]
    assert float(np.linalg.norm(bf_filt.U - bf_raw.U)) > 0.0


def test_song_32deg_load_is_transposed_and_shaped() -> None:
    packed = BASEFLOW_NPZ
    if not packed.is_file():
        pytest.skip("song_32deg.npz is not in this checkout")
    X, Y, bf, sponge, info = load_case_fields(
        packed,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        transpose_input=True,
        preprocess={"enabled": False},
    )
    raw = np.load(packed)
    assert X.shape == raw["x"].T.shape
    assert np.array_equal(X, raw["x"].T)
    assert np.array_equal(bf.U, raw["u"].T)
    assert np.array_equal(sponge, raw["sponge"].T)
    assert info["summary"] == "disabled"


def test_preprocess_null_boundary_block_is_a_noop(tmp_path: Path) -> None:
    packed = _write_fields(tmp_path)
    _X, _Y, _bf, sponge, info = load_case_fields(
        packed,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        preprocess={"enabled": True, "sponge": {"boundary": None}},
    )
    np.testing.assert_array_equal(sponge, np.zeros_like(sponge))
    assert info["sponge"]["boundary"]["enabled"] is False


def test_load_case_fields_rejects_missing_npz(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_case_fields(tmp_path / "missing.npz", Re=1.0, Ma=1.0, Pr=1.0, kappa=1.4, T_0=300.0)


def _preprocess_fields(tmp_path: Path):
    packed = _write_fields(tmp_path)
    data = np.load(packed)
    return (
        data["x"],
        data["y"],
        data["rho"],
        data["u"],
        data["v"],
        data["w"],
        data["T"],
        np.ones_like(data["sponge"]),
    )


def test_apply_preprocess_rejects_non_mappings(tmp_path: Path) -> None:
    fields = _preprocess_fields(tmp_path)
    with pytest.raises(TypeError, match="preprocess must be a mapping"):
        apply_preprocess(*fields, preprocess_cfg=["enabled"])
    with pytest.raises(TypeError, match="preprocess.sponge must be a mapping"):
        apply_preprocess(*fields, preprocess_cfg={"enabled": True, "sponge": "thick"})
    with pytest.raises(TypeError, match="preprocess.sponge.boundary"):
        apply_preprocess(*fields, preprocess_cfg={"enabled": True, "sponge": {"boundary": "wall"}})
    with pytest.raises(TypeError, match="preprocess.filter must be a mapping"):
        apply_preprocess(*fields, preprocess_cfg={"enabled": True, "filter": "on"})


def test_apply_preprocess_scales_and_blends_boundary_sponge(tmp_path: Path) -> None:
    X, Y, rho, u, v, w, T, sponge = _preprocess_fields(tmp_path)
    *_rest, scaled, info = apply_preprocess(
        X, Y, rho, u, v, w, T, sponge, {"enabled": True, "sponge": {"scale": 2.0}}
    )
    np.testing.assert_allclose(scaled, 2.0 * sponge)
    assert any(item.startswith("sponge_scale=") for item in info["applied"])

    *_rest, replaced, _info = apply_preprocess(
        X,
        Y,
        rho,
        u,
        v,
        w,
        T,
        sponge,
        {
            "enabled": True,
            "sponge": {
                "boundary": {
                    "enabled": True,
                    "mode": "replace",
                    "points": [2, 0, 0, 0],
                    "amplitudes": [3.0, 0.0, 0.0, 0.0],
                }
            },
        },
    )
    assert float(replaced.max()) == pytest.approx(3.0)
    assert float(replaced[0, 3]) == pytest.approx(0.0)

    *_rest, maximised, _info = apply_preprocess(
        X,
        Y,
        rho,
        u,
        v,
        w,
        T,
        sponge * 0.0,
        {
            "enabled": True,
            "sponge": {
                "boundary": {
                    "enabled": True,
                    "mode": "max",
                    "points": [0, 0, 2, 0],
                    "amplitudes": [0.0, 0.0, 4.0, 0.0],
                }
            },
        },
    )
    assert float(maximised.max()) == pytest.approx(4.0)

    with pytest.raises(ValueError, match="add, max, replace"):
        apply_preprocess(
            X,
            Y,
            rho,
            u,
            v,
            w,
            T,
            sponge,
            {
                "enabled": True,
                "sponge": {
                    "boundary": {"enabled": True, "mode": "multiply", "points": [1, 0, 0, 0]}
                },
            },
        )


def test_apply_preprocess_filter_loud_fails_and_none_blocks(tmp_path: Path) -> None:
    fields = _preprocess_fields(tmp_path)
    *_rest, out, info = apply_preprocess(
        *fields,
        preprocess_cfg={"enabled": True, "sponge": None, "filter": None},
    )
    assert info["enabled"] is True
    np.testing.assert_array_equal(out, fields[-1])

    with pytest.raises(ValueError, match="non-empty list"):
        apply_preprocess(
            *fields,
            preprocess_cfg={"enabled": True, "filter": {"enabled": True, "directions": []}},
        )
    with pytest.raises(ValueError, match="Unsupported filter field"):
        apply_preprocess(
            *fields,
            preprocess_cfg={
                "enabled": True,
                "filter": {
                    "enabled": True,
                    "passes": 1,
                    "fields": ["pressure"],
                    "directions": ["x"],
                },
            },
        )
    with pytest.raises(ValueError, match="four values"):
        apply_preprocess(
            *fields,
            preprocess_cfg={
                "enabled": True,
                "sponge": {"boundary": {"enabled": True, "points": None, "amplitudes": [1, 0]}},
            },
        )


def test_apply_preprocess_drops_duplicate_filter_directions(tmp_path: Path) -> None:
    X, Y, rho, u, v, w, T, sponge = _preprocess_fields(tmp_path)
    u = np.zeros_like(u)
    u[3, 4] = 1.0
    base = {
        "enabled": True,
        "filter": {"enabled": True, "passes": 1, "fields": ["u"], "directions": ["x"]},
    }
    dup = {
        "enabled": True,
        "filter": {
            "enabled": True,
            "passes": 1,
            "fields": ["u"],
            "directions": ["x", "X", "x"],
        },
    }
    out_once = apply_preprocess(
        X, Y, rho.copy(), u.copy(), v.copy(), w.copy(), T.copy(), sponge.copy(), base
    )
    out_dup = apply_preprocess(
        X, Y, rho.copy(), u.copy(), v.copy(), w.copy(), T.copy(), sponge.copy(), dup
    )
    np.testing.assert_allclose(out_once[3], out_dup[3])
    assert out_dup[-1]["filter"]["directions"] == ["x"]
    assert out_once[3][3, 4] == pytest.approx(6.0 / 16.0)


def test_binomial5_filter_rejects_bad_shape_and_axis() -> None:
    with pytest.raises(ValueError, match="2-D"):
        _binomial5_filter(np.ones(4), axis=0)
    with pytest.raises(ValueError, match="axis must be 0 or 1"):
        _binomial5_filter(np.ones((3, 3)), axis=2)


def test_preprocess_binomial5_kernel_is_1_4_6_4_1_over_16() -> None:
    constant = np.full((7, 6), 2.5)
    np.testing.assert_allclose(_binomial5_filter(constant, axis=0), constant)
    np.testing.assert_allclose(_binomial5_filter(constant, axis=1), constant)
    y = np.arange(7.0)[:, None]
    x = np.arange(6.0)[None, :]
    linear = 3.0 * y + 0.5 * x
    np.testing.assert_allclose(_binomial5_filter(linear, axis=0)[2:-2, :], linear[2:-2, :])
    np.testing.assert_allclose(_binomial5_filter(linear, axis=1)[:, 2:-2], linear[:, 2:-2])
    impulse = np.zeros((9, 7))
    impulse[4, 3] = 1.0
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    out0 = _binomial5_filter(impulse, axis=0)
    np.testing.assert_allclose(out0[2:7, 3], kernel)
    out1 = _binomial5_filter(impulse, axis=1)
    np.testing.assert_allclose(out1[4, 1:6], kernel)


def test_apply_preprocess_can_enable_shock_cleanup(tmp_path: Path) -> None:
    fields = _preprocess_fields(tmp_path)
    *_rest, _sponge, info = apply_preprocess(
        *fields,
        preprocess_cfg={"enabled": True, "shock": {"enabled": True}},
    )
    assert "shock_cleanup" in info["applied"]


def test_apply_preprocess_filter_leaves_the_caller_arrays_alone(tmp_path: Path) -> None:
    fields = _preprocess_fields(tmp_path)
    rng = np.random.default_rng(1)
    for arr in fields[2:7]:
        arr += 0.1 * rng.standard_normal(arr.shape)
    before = [np.array(arr, copy=True) for arr in fields[2:7]]
    out = apply_preprocess(
        *fields, preprocess_cfg={"enabled": True, "filter": {"enabled": True, "passes": 1}}
    )
    for arr, ref in zip(fields[2:7], before, strict=True):
        np.testing.assert_array_equal(arr, ref)
    assert not np.array_equal(out[2], before[0])
