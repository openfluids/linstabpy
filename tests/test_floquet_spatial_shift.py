# SPDX-License-Identifier: GPL-3.0-only
"""Floquet D2RZ shift and spatial finite difference in kx."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.operators.assembly import _apply_floquet_shift, _resolve_floquet_shift_mode
from linstabpy.operators.spatial import build_spatial_operator


def test_floquet_matlab_parity_shifts_d2rz() -> None:
    ngp = 3
    eye = sp.eye(ngp, format="csc")
    dz = eye.copy()
    d2z = 2.0 * eye
    d2rz = 3.0 * eye
    i_alpha = 0.5j

    shifted_dz, shifted_d2z, shifted_d2rz = _apply_floquet_shift(
        dz, d2z, d2rz, i_alpha, ngp, mode="matlab_parity"
    )

    np.testing.assert_allclose(
        shifted_d2rz[0].toarray(),
        (d2rz + i_alpha * eye).toarray(),
    )
    np.testing.assert_allclose(
        shifted_dz[0].toarray(),
        (dz + i_alpha * eye).toarray(),
    )
    # MATLAB uses the already-shifted DZ in D2Z.
    np.testing.assert_allclose(
        shifted_d2z[0].toarray(),
        (d2z + 2.0 * i_alpha * shifted_dz[0] + i_alpha**2 * eye).toarray(),
    )


def test_floquet_continuum_shifts_d2rz_with_dr() -> None:
    ngp = 3
    eye = sp.eye(ngp, format="csc")
    dz = eye.copy()
    d2z = 2.0 * eye
    d2rz = 3.0 * eye
    dr = 4.0 * eye
    i_alpha = 0.5j

    shifted_dz, shifted_d2z, shifted_d2rz = _apply_floquet_shift(
        dz, d2z, d2rz, i_alpha, ngp, DR=dr, mode="continuum"
    )

    np.testing.assert_allclose(
        shifted_d2rz[0].toarray(),
        (d2rz + i_alpha * dr).toarray(),
    )
    np.testing.assert_allclose(
        shifted_d2z[0].toarray(),
        (d2z + 2.0 * i_alpha * dz + i_alpha**2 * eye).toarray(),
    )


def test_spatial_finite_difference_is_in_kx(monkeypatch) -> None:
    from linstabpy.baseflow.baseflow import BaseFlow
    from linstabpy.baseflow.viscosity import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import _assemble_from_prepared, _prepare_coefficients

    mesh = Mesh.create(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=6, ny=5, fd_order=2)
    ones = np.ones_like(mesh.X)
    baseflow = BaseFlow(
        Re=100.0,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=300.0,
        RHO=ones,
        U=ones,
        V=0.1 * ones,
        W=np.zeros_like(ones),
        T=ones,
    )
    sutherland_air(baseflow)

    kx_used: list[float] = []
    floquet_used: list[float] = []
    real_prepare = _prepare_coefficients
    real_assemble = _assemble_from_prepared

    def spy_prepare(mesh_arg, baseflow_arg, kx=0.0, model="2D", m=0):
        kx_used.append(float(kx))
        return real_prepare(mesh_arg, baseflow_arg, kx=kx, model=model, m=m)

    def spy_assemble(prep, floquet_exp=0.0, **kwargs):
        floquet_used.append(float(floquet_exp))
        return real_assemble(prep, floquet_exp=floquet_exp, **kwargs)

    monkeypatch.setattr("linstabpy.operators.spatial._prepare_coefficients", spy_prepare)
    monkeypatch.setattr("linstabpy.operators.spatial._assemble_from_prepared", spy_assemble)

    build_spatial_operator(mesh, baseflow, model="2D", floquet_exp=0.25, kx=0.0, delta=1.0)

    assert kx_used == [0.0, 1.0, -1.0]
    assert floquet_used == [0.25, 0.25, 0.25]


def test_spatial_delta_must_be_positive() -> None:
    from linstabpy.baseflow.baseflow import BaseFlow
    from linstabpy.mesh import Mesh

    mesh = Mesh.create(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=4, ny=4, fd_order=2)
    ones = np.ones_like(mesh.X)
    baseflow = BaseFlow(
        Re=100.0,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=300.0,
        RHO=ones,
        U=ones,
        V=np.zeros_like(ones),
        W=np.zeros_like(ones),
        T=ones,
    )
    with pytest.raises(ValueError, match="delta must be positive"):
        build_spatial_operator(mesh, baseflow, delta=0.0)


def test_reconstruct_floquet_1d_and_both_2d_layouts() -> None:
    from linstabpy.utils.floquet import reconstruct_floquet

    x = np.linspace(0.0, 1.0, 3)
    y = np.linspace(0.0, 2.0, 2)
    X, _Y = np.meshgrid(x, y, indexing="xy")
    used = np.array([0, 2, 4], dtype=int)
    beta = 0.5
    values = np.array([1.0, 2.0, 3.0], dtype=complex)
    rec = reconstruct_floquet(X, values, used, beta)
    assert rec.shape == X.shape
    flat = rec.ravel(order="F")
    phase = np.exp(1j * beta * X).ravel(order="F")
    np.testing.assert_allclose(flat[used], phase[used] * values)
    unused = np.setdiff1d(np.arange(X.size), used)
    np.testing.assert_allclose(flat[unused], 0.0)

    v_ngp_first = np.column_stack([values, 2.0 * values])
    rec2 = reconstruct_floquet(X, v_ngp_first, used, beta)
    assert rec2.shape == (2,) + X.shape
    np.testing.assert_allclose(rec2[0], rec)
    np.testing.assert_allclose(rec2[1], 2.0 * rec)

    v_fields_first = np.vstack([values, 3.0 * values])
    rec3 = reconstruct_floquet(X, v_fields_first, used, beta)
    assert rec3.shape == (2,) + X.shape
    np.testing.assert_allclose(rec3[0], rec)
    np.testing.assert_allclose(rec3[1], 3.0 * rec)


def test_reconstruct_floquet_rejects_bad_shapes() -> None:
    from linstabpy.utils.floquet import reconstruct_floquet

    X = np.zeros((2, 3))
    used = np.arange(4, dtype=int)
    with pytest.raises(ValueError, match="1-D V"):
        reconstruct_floquet(X, np.ones(3, dtype=complex), used, 0.0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        reconstruct_floquet(X, np.ones((2, 2, 2), dtype=complex), used, 0.0)
    with pytest.raises(ValueError, match="one dimension"):
        reconstruct_floquet(X, np.ones((2, 3), dtype=complex), used, 0.0)


def test_floquet_shift_mode_names_matlab_and_continuum() -> None:
    assert _resolve_floquet_shift_mode("matlab") == "matlab_parity"
    assert _resolve_floquet_shift_mode("physics") == "continuum"
    with pytest.raises(ValueError, match="Floquet shift mode"):
        _resolve_floquet_shift_mode("random")
