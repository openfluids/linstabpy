"""Shared fixtures for linstab2d test suite."""

from __future__ import annotations

import numpy as np
import pytest

from linstab2d.mesh import Mesh
from linstab2d.baseflow import BaseFlow, sutherland_air, load_blasius


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

FD_ORDERS = [2, 4, 6, 8]


@pytest.fixture(params=FD_ORDERS, ids=[f"order{o}" for o in FD_ORDERS])
def fd_order(request):
    """Parametrized FD order."""
    return request.param


# ---------------------------------------------------------------------------
# Mesh fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_mesh_2d():
    """20x30 mesh on [0,2]x[0,1], order 4."""
    return Mesh.create(
        xrange=(0.0, 2.0),
        yrange=(0.0, 1.0),
        nx=20,
        ny=30,
        fd_order=4,
    )


@pytest.fixture
def small_mesh_2d_order(fd_order):
    """20x30 mesh parametrized over FD orders."""
    return Mesh.create(
        xrange=(0.0, 2.0),
        yrange=(0.0, 1.0),
        nx=20,
        ny=30,
        fd_order=fd_order,
    )


@pytest.fixture
def small_mesh_periodic():
    """32x20 mesh with x-periodicity."""
    return Mesh.create(
        xrange=(0.0, 2 * np.pi),
        yrange=(0.0, 1.0),
        nx=32,
        ny=20,
        fd_order=4,
        x_periodicity=True,
    )


@pytest.fixture
def small_mesh_symmetric():
    """20x30 mesh with y-symmetry (axisymmetric pole)."""
    return Mesh.create(
        xrange=(0.0, 2.0),
        yrange=(0.0, 1.0),
        nx=20,
        ny=30,
        fd_order=4,
        y_symmetry=True,
    )


@pytest.fixture
def deformed_mesh(small_mesh_2d):
    """small_mesh_2d with sinh stretching in Y."""
    s = 2.0
    Y_comp = small_mesh_2d.Y
    y_max = Y_comp.max()
    Y_new = y_max * np.sinh(s * Y_comp / y_max) / np.sinh(s)
    return small_mesh_2d.deform(Y_new=Y_new)


# ---------------------------------------------------------------------------
# Base flow fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_baseflow(small_mesh_2d):
    """Uniform base flow: U=1, V=W=0, RHO=T=1, MU=1."""
    ny, nx = small_mesh_2d.X.shape
    ones = np.ones((ny, nx))
    zeros = np.zeros((ny, nx))
    bf = BaseFlow(
        Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15,
        RHO=ones.copy(), U=ones.copy(), V=zeros.copy(),
        W=zeros.copy(), T=ones.copy(),
    )
    sutherland_air(bf)
    return bf


@pytest.fixture
def blasius_baseflow_small():
    """Blasius on a 30x20 mesh, Re=1e4, Ma=0.3."""
    mesh = Mesh.create(
        xrange=(200.0, 300.0),
        yrange=(0.0, 30.0),
        nx=30,
        ny=20,
        fd_order=4,
    )
    bf = load_blasius(
        mesh.X, mesh.Y,
        Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15,
    )
    sutherland_air(bf)
    return mesh, bf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_polynomial_field(mesh, px, py):
    """Create x^px * y^py as a flat vector in Fortran order on the mesh."""
    X = mesh.X
    Y = mesh.Y
    field_2d = X**px * Y**py
    return field_2d.ravel(order="F")[mesh.used_ind]
