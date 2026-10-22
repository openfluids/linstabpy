# SPDX-License-Identifier: GPL-3.0-only
"""Tests for the Q_diss_x / Q_diss_y fields on the Mesh dataclass (Phase 1, Step 3).

Three checks:
  - test_default_off  : flag absent -> both fields are None (T1 gate prerequisite).
  - test_enabled_populates : flag on -> both fields are csr_matrix with correct
    shape and H*Q_diss_x is symmetric (SPSD passthrough).
  - test_explicit_order : order_diss may differ from fd_order.

Each test method imports Mesh lazily to match the conftest.py pattern.
That avoids module-level petsc4py import failures on machines without PETSc.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.mesh.sbp import build_sbp_dissipation_matrices

# Small mesh dimensions used throughout.
_NX = 20
_NY = 16  # fd8 needs n >= 2r = 16


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
        silently build a fd6 dissipation operator on a fd4 mesh.
        That would break the SPSD-in-the-correct-H-norm argument that Phase
        1's energy proof relies on.
        Phase 3 (decoupled orders, MSN 2004 §4) is the legitimate place to
        lift this restriction.
        Until then mixed orders must raise.
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


def _runs(coord, level_coord, step):
    """Independent line runs: indices of used points on each grid line, split at holes."""
    for level in np.unique(level_coord):
        pos = np.flatnonzero(level_coord == level)
        pos = pos[np.argsort(coord[pos])]
        cuts = np.flatnonzero(np.diff(coord[pos]) > 1.5 * step) + 1
        yield from np.split(pos, cuts)


class TestMaskedMeshDissipation:
    """Mesh.mask rebuilds Q_diss on the used points, one operator per line run."""

    def _masked(self):
        from linstabpy.mesh import Mesh

        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=25,
            ny=25,
            fd_order=4,
            numerics_operator_dissipation={"enabled": True},
        )
        keep = np.ones(mesh.X.shape, dtype=bool)
        keep[9:13, 7:12] = False
        return mesh, mesh.mask(keep)

    def test_blocks_match_the_line_runs_and_are_semidefinite(self) -> None:
        from linstabpy.mesh.sbp import _build_sbp_norm_diag

        _, masked = self._masked()
        x = masked.X.ravel(order="F")[masked.used_ind]
        y = masked.Y.ravel(order="F")[masked.used_ind]
        h = 1.0 / 24
        for Q, along, across in ((masked.Q_diss_x, x, y), (masked.Q_diss_y, y, x)):
            assert Q.shape == (masked.ngp, masked.ngp)
            expected = sp.lil_matrix(Q.shape)
            for run in _runs(along, across, h):
                if run.size < 8:  # shorter than min_points(4): no dissipation on the run
                    continue
                Q_run, _ = build_sbp_dissipation_matrices(run.size, h, 4, 4)
                expected[np.ix_(run, run)] = Q_run
                HQ = (sp.diags(_build_sbp_norm_diag(run.size, h, 4)) @ Q_run).toarray()
                np.testing.assert_allclose(HQ, HQ.T, atol=1e-12 * np.abs(HQ).max())
                assert np.linalg.eigvalsh(0.5 * (HQ + HQ.T)).min() >= -1e-12 * np.abs(HQ).max()
            assert abs(Q - expected.tocsr()).max() == 0.0

    def test_hole_edge_differs_from_the_unmasked_line(self) -> None:
        mesh, masked = self._masked()
        ny = mesh.X.shape[0]
        full = sp.kron(mesh.Q_diss_x, sp.eye(ny), format="csr")[masked.used_ind][:, masked.used_ind]
        assert abs(masked.Q_diss_x - full).max() > 0.0

    def test_all_true_mask_equals_the_unmasked_operator(self) -> None:
        from linstabpy.mesh import Mesh

        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=20,
            ny=18,
            fd_order=6,
            numerics_operator_dissipation={"enabled": True, "order_diss": 6},
        )
        full = mesh.mask(np.ones(mesh.X.shape, dtype=bool))
        ny, nx = mesh.X.shape
        assert abs(full.Q_diss_x - sp.kron(mesh.Q_diss_x, sp.eye(ny))).max() == 0.0
        assert abs(full.Q_diss_y - sp.kron(sp.eye(nx), mesh.Q_diss_y)).max() == 0.0
