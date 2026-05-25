# SPDX-License-Identifier: GPL-3.0-only
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

from linstabpy.mesh.sbp import build_sbp_dissipation_matrices

# Small mesh dimensions used throughout.
_NX = 20
_NY = 15


class TestMeshDissipationDefaultOff:
    """T1 prerequisite: flag absent -> Q_diss fields are None."""

    def test_default_off(self) -> None:
        from linstabpy.mesh import Mesh

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
        from linstabpy.mesh import Mesh

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
        from linstabpy.mesh import Mesh

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
        from linstabpy.mesh import Mesh

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
        from linstabpy.mesh import Mesh

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
            n=_NX,
            h=hx_ref,
            order_interior=4,
        )
        M = (H_norm @ mesh.Q_diss_x).toarray()
        sym_err = np.max(np.abs(M - M.T))
        assert sym_err < 1e-10, f"H * Q_diss_x not symmetric: max|M - M.T| = {sym_err:.2e}"


class TestMeshDissipationExplicitOrder:
    """order_diss must match fd_order in Phase 1; mixed orders raise."""

    def test_matching_orders_succeed(self) -> None:
        """Meshes built with order_diss == fd_order must succeed."""
        import pytest

        from linstabpy.mesh import Mesh

        for order in (4, 6, 8):
            mesh = Mesh.create(
                xrange=(0.0, 1.0),
                yrange=(0.0, 1.0),
                nx=_NX,
                ny=_NY,
                fd_order=order,
                numerics_operator_dissipation={"enabled": True, "order_diss": order},
            )
            assert mesh.Q_diss_x is not None
            assert mesh.Q_diss_y is not None

    def test_mixed_orders_raise(self) -> None:
        """Phase 1 enforces order_diss == fd_order; mixed orders must raise.

        Without this guard, a config like fd_order=4, order_diss=6 would
        silently build a fd6 dissipation operator on a fd4 mesh, breaking
        the SPSD-in-the-correct-H-norm argument that Phase 1's energy proof
        relies on. Phase 3 (decoupled orders, MSN 2004 §4) is the legitimate
        place to lift this restriction; until then it must be a hard error.
        """
        import pytest

        from linstabpy.mesh import Mesh

        with pytest.raises(NotImplementedError, match="order_diss"):
            Mesh.create(
                xrange=(0.0, 1.0),
                yrange=(0.0, 1.0),
                nx=_NX,
                ny=_NY,
                fd_order=4,
                numerics_operator_dissipation={"enabled": True, "order_diss": 6},
            )
