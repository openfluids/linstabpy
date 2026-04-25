"""Tests for the Q_diss_x / Q_diss_y fields on the Mesh dataclass (Phase 1, Step 3).

Three checks:
  - test_default_off  : flag absent -> both fields are None (T1 gate prerequisite).
  - test_enabled_populates : flag on -> both fields are csr_matrix with correct
    shape and H*Q_diss_x is symmetric (SPSD passthrough).
  - test_explicit_order : order_diss parameter is honoured independently of fd_order.

Mesh is imported lazily inside each test method to match the conftest.py pattern
and avoid module-level petsc4py import failures on machines without PETSc.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstab2d.mesh.sbp import build_sbp_dissipation_matrices

# Small mesh dimensions used throughout.
_NX = 20
_NY = 15


class TestMeshDissipationDefaultOff:
    """T1 prerequisite: flag absent -> Q_diss fields are None."""

    def test_default_off(self) -> None:
        from linstab2d.mesh import Mesh

        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=_NX,
            ny=_NY,
            fd_order=4,
        )
        assert mesh.Q_diss_x is None, "Q_diss_x must be None when flag is absent"
        assert mesh.Q_diss_y is None, "Q_diss_y must be None when flag is absent"

    def test_flag_false(self) -> None:
        from linstab2d.mesh import Mesh

        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=_NX,
            ny=_NY,
            fd_order=4,
            numerics_operator_dissipation={"enabled": False},
        )
        assert mesh.Q_diss_x is None, "Q_diss_x must be None when enabled=False"
        assert mesh.Q_diss_y is None, "Q_diss_y must be None when enabled=False"


class TestMeshDissipationEnabled:
    """Flag on -> Q_diss fields populated with correct shape and SPSD property."""

    def test_enabled_populates_type(self) -> None:
        from linstab2d.mesh import Mesh

        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=_NX,
            ny=_NY,
            fd_order=4,
            numerics_operator_dissipation={"enabled": True},
        )
        assert mesh.Q_diss_x is not None, "Q_diss_x must not be None when enabled"
        assert mesh.Q_diss_y is not None, "Q_diss_y must not be None when enabled"
        assert isinstance(mesh.Q_diss_x, sp.csr_matrix), (
            f"Q_diss_x must be csr_matrix, got {type(mesh.Q_diss_x)}"
        )
        assert isinstance(mesh.Q_diss_y, sp.csr_matrix), (
            f"Q_diss_y must be csr_matrix, got {type(mesh.Q_diss_y)}"
        )

    def test_enabled_shape(self) -> None:
        from linstab2d.mesh import Mesh

        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=_NX,
            ny=_NY,
            fd_order=4,
            numerics_operator_dissipation={"enabled": True},
        )
        assert mesh.Q_diss_x.shape == (_NX, _NX), (
            f"Q_diss_x shape {mesh.Q_diss_x.shape} != ({_NX}, {_NX})"
        )
        assert mesh.Q_diss_y.shape == (_NY, _NY), (
            f"Q_diss_y shape {mesh.Q_diss_y.shape} != ({_NY}, {_NY})"
        )

    def test_enabled_spsd_symmetry(self) -> None:
        """H * Q_diss_x must be symmetric (SPSD passthrough, T2 gate)."""
        from linstab2d.mesh import Mesh

        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=_NX,
            ny=_NY,
            fd_order=4,
            numerics_operator_dissipation={"enabled": True},
        )
        # Reconstruct H_norm for the same n and h used inside Mesh.create.
        hx_ref = 1.0 / (_NX - 1)
        _, H_norm = build_sbp_dissipation_matrices(
            n=_NX, h=hx_ref, order_interior=4,
        )
        M = (H_norm @ mesh.Q_diss_x).toarray()
        sym_err = np.max(np.abs(M - M.T))
        assert sym_err < 1e-10, (
            f"H * Q_diss_x not symmetric: max|M - M.T| = {sym_err:.2e}"
        )


class TestMeshDissipationExplicitOrder:
    """order_diss parameter is honoured independently of fd_order."""

    def test_explicit_order_differs(self) -> None:
        """Meshes built with order_diss=4 and order_diss=6 must differ."""
        from linstab2d.mesh import Mesh

        mesh_o4 = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=_NX,
            ny=_NY,
            fd_order=4,
            numerics_operator_dissipation={"enabled": True, "order_diss": 4},
        )
        mesh_o6 = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=_NX,
            ny=_NY,
            fd_order=4,
            numerics_operator_dissipation={"enabled": True, "order_diss": 6},
        )
        # The two dissipation operators must differ when order_diss differs.
        same = np.allclose(
            mesh_o4.Q_diss_x.toarray(),
            mesh_o6.Q_diss_x.toarray(),
            atol=1e-14,
        )
        assert not same, (
            "Q_diss_x with order_diss=4 and order_diss=6 must differ; "
            "parameter is not being honoured"
        )
