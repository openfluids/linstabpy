# SPDX-License-Identifier: GPL-3.0-only
"""Grid-field loader: npy directory or one npz, then a BaseFlow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linstabpy.baseflow import load_case_fields, load_grid_fields


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
    X, Y, bf, sponge = load_grid_fields(tmp_path, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33)
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
        **{
            name: np.load(tmp_path / f"{name}.npy")
            for name in ("x", "y", "rho", "u", "v", "w", "T", "sponge")
        },
    )
    X_dir, _, bf_dir, _ = load_grid_fields(
        tmp_path, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33
    )
    X_npz, _, bf_npz, _ = load_grid_fields(
        packed, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33
    )
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


def test_load_case_fields_can_refuse_asymmetric_stride(tmp_path: Path) -> None:
    _write_npy_dir(tmp_path)
    with pytest.raises(ValueError, match="Asymmetric strides"):
        load_case_fields(
            tmp_path,
            Re=1.0e5,
            Ma=2.15,
            Pr=0.72,
            kappa=1.4,
            T_0=153.33,
            stride_ksi=2,
            stride_eta=1,
            refuse_asymmetric_stride=True,
        )


def test_read_grid_fields_rejects_bad_sources(tmp_path: Path) -> None:
    from linstabpy.baseflow.grid_fields import read_grid_fields

    txt = tmp_path / "fields.txt"
    txt.write_text("not npz")
    with pytest.raises(ValueError, match=".npz"):
        read_grid_fields(txt)
    packed = tmp_path / "partial.npz"
    np.savez_compressed(packed, x=np.ones((2, 2)))
    with pytest.raises(FileNotFoundError, match="missing fields"):
        read_grid_fields(packed)
    with pytest.raises(FileNotFoundError, match="not found"):
        read_grid_fields(tmp_path / "no_such_dir")
    _write_npy_dir(tmp_path)
    (tmp_path / "rho.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Missing field file"):
        read_grid_fields(tmp_path)


def test_read_grid_fields_rejects_a_1d_array(tmp_path: Path) -> None:
    packed = tmp_path / "fields.npz"
    np.savez_compressed(
        packed,
        x=np.linspace(0.0, 1.0, 4),
        y=np.ones((2, 2)),
        rho=np.ones((2, 2)),
        u=np.ones((2, 2)),
        v=np.ones((2, 2)),
        w=np.ones((2, 2)),
        T=np.ones((2, 2)),
        sponge=np.ones((2, 2)),
    )
    with pytest.raises(ValueError, match="must be 2-D"):
        load_grid_fields(packed, Re=1.0, Ma=1.0, Pr=0.72, kappa=1.4, T_0=1.0)


def test_load_case_fields_rejects_bad_stride_and_shape(tmp_path: Path) -> None:
    _write_npy_dir(tmp_path)
    with pytest.raises(ValueError, match="stride"):
        load_case_fields(tmp_path, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33, stride_ksi=0)
    np.save(tmp_path / "u.npy", np.ones((3, 3)))
    with pytest.raises(ValueError, match="does not match"):
        load_case_fields(tmp_path, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33)


def test_load_case_fields_transposes_packed_axes(tmp_path: Path) -> None:
    _write_npy_dir(tmp_path, nx=6, ny=4)
    u = np.zeros((4, 6))
    u[1, 5] = 7.0
    np.save(tmp_path / "u.npy", u.T)
    for name in ("x", "y", "rho", "v", "w", "T", "sponge"):
        arr = np.load(tmp_path / f"{name}.npy")
        np.save(tmp_path / f"{name}.npy", arr.T)
    X, _, bf, sponge, _info = load_case_fields(
        tmp_path,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        transpose_input=True,
    )
    assert X.shape == (4, 6)
    assert bf.U.shape == (4, 6)
    assert bf.U[1, 5] == pytest.approx(7.0)
    assert sponge.shape == X.shape


def test_load_grid_fields_zeros_spanwise_velocity(tmp_path: Path) -> None:
    _write_npy_dir(tmp_path)
    np.save(tmp_path / "w.npy", np.full((4, 6), 3.0))
    _, _, bf_kept, _ = load_grid_fields(tmp_path, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33)
    np.testing.assert_allclose(bf_kept.W, 3.0)
    np.testing.assert_allclose(bf_kept.U, 2.15)
    _, _, bf, _ = load_grid_fields(
        tmp_path,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        zero_spanwise=True,
    )
    np.testing.assert_allclose(bf.W, 0.0)
    assert bf.W.shape == bf.U.shape
    np.testing.assert_allclose(bf.U, 2.15)
