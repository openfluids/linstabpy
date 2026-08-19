# SPDX-License-Identifier: GPL-3.0-only
"""Metamorphic check for the shared SciPy/PETSc LNSE assembly path."""

from __future__ import annotations

import numpy as np
import pytest


def _tiny_uniform_case():
    from linstabpy.baseflow import BaseFlow, sutherland_air
    from linstabpy.mesh import Mesh

    mesh = Mesh.create(
        xrange=(0.0, 1.0),
        yrange=(0.0, 0.5),
        nx=6,
        ny=5,
        fd_order=2,
    )
    ny, nx = mesh.X.shape
    ones = np.ones((ny, nx))
    zeros = np.zeros((ny, nx))
    baseflow = BaseFlow(
        Re=1.0e4,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=ones.copy(),
        U=ones.copy(),
        V=zeros.copy(),
        W=zeros.copy(),
        T=ones.copy(),
    )
    sutherland_air(baseflow)
    return mesh, baseflow


def test_scipy_and_direct_petsc_lnse_assembly_match_within_tolerance():
    """The two public entry points must use one equivalent orchestration path."""
    pytest.importorskip("petsc4py")

    from linstabpy.operators.assembly import build_lnse_operator, build_lnse_operator_petsc
    from linstabpy.utils.petsc_core import petsc_to_scipy, require_complex

    try:
        require_complex({"mpi": {"petsc_matrix_mode": "local"}})
    except RuntimeError as exc:
        pytest.skip(str(exc))

    mesh, baseflow = _tiny_uniform_case()
    L_scipy, idx_scipy = build_lnse_operator(mesh, baseflow, kx=0.25, model="2D")
    L_petsc, idx_petsc = build_lnse_operator_petsc(
        mesh,
        baseflow,
        kx=0.25,
        model="2D",
        solver_cfg={"mpi": {"petsc_matrix_mode": "local"}},
    )
    try:
        L_petsc_sp = petsc_to_scipy(L_petsc, format="csc")
        diff = (L_scipy - L_petsc_sp).tocoo()
        max_abs = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))

        assert idx_scipy.keys() == idx_petsc.keys()
        for key in idx_scipy:
            np.testing.assert_array_equal(idx_scipy[key], idx_petsc[key])
        assert max_abs <= 1e-10
    finally:
        L_petsc.destroy()
