# SPDX-License-Identifier: GPL-3.0-only
"""Floquet D2RZ shift and spatial finite difference in kx."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from linstabpy.operators.assembly import _apply_floquet_shift
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

    monkeypatch.setattr(
        "linstabpy.operators.spatial._prepare_coefficients", spy_prepare
    )
    monkeypatch.setattr(
        "linstabpy.operators.spatial._assemble_from_prepared", spy_assemble
    )

    build_spatial_operator(mesh, baseflow, model="2D", floquet_exp=0.25, kx=0.0, delta=1.0)

    assert kx_used == [0.0, 1.0, -1.0]
    assert floquet_used == [0.25, 0.25, 0.25]
