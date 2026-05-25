# SPDX-License-Identifier: GPL-3.0-only
"""Shared fixtures for linstabpy test suite."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mesh fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_mesh_2d():
    """20x30 mesh on [0,2]x[0,1], order 4."""
    from linstabpy.mesh import Mesh

    return Mesh.create(
        xrange=(0.0, 2.0),
        yrange=(0.0, 1.0),
        nx=20,
        ny=30,
        fd_order=4,
    )


@pytest.fixture
def small_mesh_periodic():
    """32x20 mesh with x-periodicity."""
    from linstabpy.mesh import Mesh

    return Mesh.create(
        xrange=(0.0, 2 * np.pi),
        yrange=(0.0, 1.0),
        nx=32,
        ny=20,
        fd_order=4,
        x_periodicity=True,
    )


# ---------------------------------------------------------------------------
# Base flow fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_baseflow(small_mesh_2d):
    """Uniform base flow: U=1, V=W=0, RHO=T=1, MU=1."""
    from linstabpy.baseflow import BaseFlow, sutherland_air

    ny, nx = small_mesh_2d.X.shape
    ones = np.ones((ny, nx))
    zeros = np.zeros((ny, nx))
    bf = BaseFlow(
        Re=1e4,
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
    sutherland_air(bf)
    return bf


@pytest.fixture
def blasius_baseflow_small():
    """Blasius on a 30x20 mesh, Re=1e4, Ma=0.3."""
    from linstabpy.baseflow import load_blasius, sutherland_air
    from linstabpy.mesh import Mesh

    mesh = Mesh.create(
        xrange=(200.0, 300.0),
        yrange=(0.0, 30.0),
        nx=30,
        ny=20,
        fd_order=4,
    )
    bf = load_blasius(
        mesh.X,
        mesh.Y,
        Re=1e4,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
    )
    sutherland_air(bf)
    return mesh, bf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Synthetic jet builder (cylindrical / axy model)
# ---------------------------------------------------------------------------


def build_synthetic_jet(nx=15, ny=10, m=0, borders="ltr", variables="uvwT"):
    """Build a synthetic axisymmetric sech² jet problem.

    Uses a deterministic tanh-derivative velocity profile on a y_symmetry mesh.
    No external data dependencies (.mat files).

    Returns dict with:
        mesh, bf, L0, idx, ngp, W_diag, invW_diag, bc_indices
    """
    from linstabpy.baseflow import BaseFlow, sutherland_air
    from linstabpy.boundary_conditions import apply_dirichlet
    from linstabpy.mesh import Mesh
    from linstabpy.operators import build_lnse_operator, compute_chu_norm

    mesh = Mesh.create(
        xrange=(0.0, 10.0),
        yrange=(0.0, 3.0),
        nx=nx,
        ny=ny,
        fd_order=4,
        y_symmetry=True,
    )
    R = mesh.Y
    U_axial = 0.9 / np.cosh(R) ** 2
    bf = BaseFlow(
        Re=1e4,
        Ma=0.9,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=np.ones_like(R),
        U=U_axial,
        V=np.zeros_like(R),
        W=np.zeros_like(R),
        T=np.ones_like(R),
    )
    sutherland_air(bf)

    L0, idx = build_lnse_operator(mesh, bf, model="axy", m=m)
    ngp = mesh.ngp
    L0, bc_indices = apply_dirichlet(L0, idx, borders, variables, ngp)

    p = mesh.used_ind
    RHO_p = bf.RHO.ravel(order="F")[p]
    T_p = bf.T.ravel(order="F")[p]
    R_p = mesh.Y.ravel(order="F")[p]
    W_int = mesh.W.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(
        W_int,
        RHO_p,
        T_p,
        bf.kappa,
        bf.Ma,
        ngp,
        model="axy",
        R=R_p,
    )
    return {
        "mesh": mesh,
        "bf": bf,
        "L0": L0,
        "idx": idx,
        "ngp": ngp,
        "W_diag": W_diag,
        "invW_diag": invW_diag,
        "bc_indices": bc_indices,
    }
