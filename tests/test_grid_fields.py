# SPDX-License-Identifier: GPL-3.0-only
"""Grid-field loader: npy directory or one npz, then a BaseFlow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linstabpy.baseflow import load_grid_fields, load_hamada_case


def _write_npy_dir(root: Path, *, nx: int = 6, ny: int = 4) -> Path:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 0.5, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    # meshgrid xy: X[j,i] = x[i], shape (ny, nx)
    fields = {
        "x": X,
        "y": Y,
        "rho": np.ones_like(X),
        "u": 2.15 * np.ones_like(X),
        "v": 0.1 * np.ones_like(X),
        "w": np.zeros_like(X),
        "T": np.ones_like(X),
        "sponge": np.zeros_like(X),
    }
    for name, arr in fields.items():
        np.save(root / f"{name}.npy", arr)
    return root


def test_load_grid_fields_from_npy_directory(tmp_path: Path) -> None:
    _write_npy_dir(tmp_path)
    X, Y, bf, sponge = load_grid_fields(
        tmp_path, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33
    )
    assert X.shape == (4, 6)
    assert Y.shape == X.shape
    assert bf.U.shape == X.shape
    assert np.allclose(bf.U, 2.15)
    assert np.allclose(bf.T, 1.0)
    assert sponge.shape == X.shape


def test_load_grid_fields_from_npz(tmp_path: Path) -> None:
    _write_npy_dir(tmp_path)
    packed = tmp_path / "case.npz"
    np.savez_compressed(
        packed,
        **{name: np.load(tmp_path / f"{name}.npy") for name in ("x", "y", "rho", "u", "v", "w", "T", "sponge")},
    )
    X_dir, _, bf_dir, _ = load_grid_fields(tmp_path, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33)
    X_npz, _, bf_npz, _ = load_grid_fields(packed, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33)
    assert np.array_equal(X_dir, X_npz)
    assert np.array_equal(bf_dir.U, bf_npz.U)


def test_load_grid_fields_allows_asymmetric_stride(tmp_path: Path) -> None:
    _write_npy_dir(tmp_path, nx=8, ny=6)
    X, Y, bf, sponge = load_grid_fields(
        tmp_path,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        stride_ksi=2,
        stride_eta=3,
    )
    assert X.shape == (2, 4)
    assert bf.U.shape == X.shape
    assert sponge.shape == X.shape


def test_load_hamada_case_still_rejects_asymmetric_stride(tmp_path: Path) -> None:
    _write_npy_dir(tmp_path)
    with pytest.raises(ValueError, match="Asymmetric strides"):
        load_hamada_case(tmp_path, stride_ksi=2, stride_eta=1)
