# SPDX-License-Identifier: GPL-3.0-only
"""Integration tests for the Case 1 case wiring."""

from __future__ import annotations

import numpy as np
import pytest

from linstabpy.baseflow import load_case_fields, sutherland_air
from linstabpy.boundary_conditions.dirichlet import apply_dirichlet
from linstabpy.boundary_conditions.neumann import apply_neumann
from linstabpy.mesh import Mesh
from linstabpy.operators.assembly import build_lnse_operator, build_lnse_system
from linstabpy.operators.energy_norm import compute_chu_norm
from linstabpy.stability import eigenvalues as temporal_eigenvalues
from linstabpy.utils.petsc_core import compare_petsc_against_scipy, petsc_matvec_residual
from tests._paths import unpack_song_baseflow

BASEFLOW_DIR = unpack_song_baseflow()


def test_load_case_fields_shapes() -> None:
    X, Y, bf, sponge, _info = load_case_fields(
        BASEFLOW_DIR, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33,
        stride_ksi=25, stride_eta=25, transpose_input=True,
    )
    assert (
        X.shape == Y.shape == bf.RHO.shape == bf.U.shape == bf.V.shape == bf.W.shape == bf.T.shape
    )
    assert sponge.shape == X.shape
    assert X.ndim == 2
    assert X.shape[0] >= 4 and X.shape[1] >= 4


def test_case1_build_lnse_system_petsc_matches_scipy_operator() -> None:
    X, Y, bf, sponge, _info = load_case_fields(
        BASEFLOW_DIR, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33,
        stride_ksi=15, stride_eta=15, transpose_input=True,
    )
    sutherland_air(bf)

    ny, nx = X.shape
    mesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=nx, ny=ny, fd_order=4).deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    bc_recipe = [
        {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
        {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
        {"type": "neumann", "borders": "b", "variables": "T"},
        {"type": "neumann", "borders": "r", "variables": "ruvwT"},
    ]
    penalty = 1e8

    L0_sp, _, bc_sp = build_lnse_system(
        mesh,
        bf,
        kx=0.0,
        model="2D",
        bc_recipe=bc_recipe,
        bc_penalty=penalty,
        backend="scipy",
    )
    L0_petsc, _, bc_petsc = build_lnse_system(
        mesh,
        bf,
        kx=0.0,
        model="2D",
        bc_recipe=bc_recipe,
        bc_penalty=penalty,
        backend="petsc",
        solver_cfg={"petsc_matrix_mode": "local"},
    )
    try:
        np.testing.assert_array_equal(bc_petsc, bc_sp)

        sparse_report = compare_petsc_against_scipy(
            L0_sp,
            L0_petsc,
            abs_tol=1e-10,
            rel_tol=1e-10,
        )
        assert sparse_report["pass"], sparse_report
        assert sparse_report["max_abs_residual"] < 1e-10

        matvec_report = petsc_matvec_residual(
            L0_sp,
            L0_petsc,
            seed=11,
            trials=2,
            abs_tol=1e-10,
            rel_tol=1e-10,
        )
        assert matvec_report["pass"], matvec_report
        assert matvec_report["max_rel_residual"] < 1e-10
    finally:
        L0_petsc.destroy()


def test_case1_operator_and_small_temporal_eigs() -> None:
    X, Y, bf, sponge, _info = load_case_fields(
        BASEFLOW_DIR, Re=1.0e5, Ma=2.15, Pr=0.72, kappa=1.4, T_0=153.33,
        stride_ksi=15, stride_eta=15, transpose_input=True,
    )
    sutherland_air(bf)

    ny, nx = X.shape
    mesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=nx, ny=ny, fd_order=4).deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    L0, idx = build_lnse_operator(mesh, bf, kx=0.0, model="2D")
    penalty = 1e8
    L0, _ = apply_dirichlet(L0, idx, "lt", "ruvwT", mesh.ngp, penalty=penalty)
    L0, _ = apply_dirichlet(L0, idx, "b", "ruvw", mesh.ngp, penalty=penalty)
    L0, _ = apply_neumann(L0, mesh, idx, "b", "T", penalty=penalty)
    L0, _ = apply_neumann(L0, mesh, idx, "r", "ruvwT", penalty=penalty)

    p = mesh.used_ind
    W_diag, _ = compute_chu_norm(
        mesh.W.ravel(order="F")[p],
        bf.RHO.ravel(order="F")[p],
        bf.T.ravel(order="F")[p],
        bf.kappa,
        bf.Ma,
        mesh.ngp,
    )
    omega, modes, info = temporal_eigenvalues(
        L0,
        n_modes=2,
        target=2.0 * np.pi,
        W_diag=W_diag,
        return_info=True,
    )

    assert omega.shape == (2,)
    assert modes.shape == (mesh.ngp * 5, 2)
    assert int(info["nconv"]) >= 2
    assert int(info["ncv_used"]) >= 3
    residuals = np.asarray(info["residual_estimates"], dtype=float)
    assert residuals.shape == (2,)
    assert np.all(np.isfinite(residuals))
