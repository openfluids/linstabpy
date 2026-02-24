"""Smoke tests for Hamada case integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linstab2d.baseflow import load_hamada_case, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.boundary_conditions.neumann import apply_neumann
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.eigenvalue import temporal_eigenvalues


CASE_DIR = Path(__file__).resolve().parent.parent / "hamada_test"

pytestmark = pytest.mark.skipif(not CASE_DIR.exists(), reason="hamada_test data is not present")


def test_load_hamada_case_shapes() -> None:
    X, Y, bf, sponge = load_hamada_case(CASE_DIR, stride_ksi=25, stride_eta=25)
    assert X.shape == Y.shape == bf.RHO.shape == bf.U.shape == bf.V.shape == bf.W.shape == bf.T.shape
    assert sponge.shape == X.shape
    assert X.ndim == 2
    assert X.shape[0] >= 4 and X.shape[1] >= 4


def test_hamada_operator_and_small_temporal_eigs() -> None:
    X, Y, bf, sponge = load_hamada_case(CASE_DIR, stride_ksi=30, stride_eta=30)
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
    omega, modes = temporal_eigenvalues(L0, n_modes=2, target=2.0 * np.pi, W_diag=W_diag)

    assert omega.shape == (2,)
    assert modes.shape == (mesh.ngp * 5, 2)
