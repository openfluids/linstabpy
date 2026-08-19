# SPDX-License-Identifier: GPL-3.0-only
"""Packed-field loader contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linstabpy.baseflow import load_case_fields
from examples.song.reference_paths import BASEFLOW_NPZ


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


def test_load_case_fields_rejects_missing_npz(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_case_fields(tmp_path / "missing.npz", Re=1.0, Ma=1.0, Pr=1.0, kappa=1.4, T_0=300.0)
