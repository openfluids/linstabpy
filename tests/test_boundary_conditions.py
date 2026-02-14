"""Tests for Dirichlet boundary condition enforcement."""

import numpy as np
import scipy.sparse as sp
import pytest

from linstab2d.mesh import Mesh
from linstab2d.baseflow import BaseFlow, sutherland_air
from linstab2d.operators import build_lnse_operator
from linstab2d.boundary_conditions import apply_dirichlet, apply_dirichlet_spatial
from linstab2d.operators import build_spatial_operator


class TestDirichletBC:
    """Tests for apply_dirichlet on the LNSE operator."""

    @pytest.fixture()
    def operator_setup(self, small_mesh_2d, uniform_baseflow):
        """Build L0, idx, and apply standard bottom/top BCs on uvwT."""
        mesh = small_mesh_2d  # 20x30, ngp=600
        bf = uniform_baseflow
        L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
        ngp = mesh.ngp
        return L0, idx, ngp

    # ------------------------------------------------------------------ #
    # 1. BC rows: diagonal equals penalty
    # ------------------------------------------------------------------ #
    def test_bc_rows_diagonal_equals_penalty(self, operator_setup):
        L0, idx, ngp = operator_setup
        penalty = 1e4
        L0_bc, bc_indices = apply_dirichlet(
            L0.copy(), idx, "bt", "uvwT", ngp, penalty=penalty
        )
        L0_bc_csc = L0_bc.tocsc()
        for i in bc_indices:
            assert L0_bc_csc[i, i] == pytest.approx(
                penalty, rel=1e-12
            ), f"Diagonal at BC index {i} should be {penalty}"

    # ------------------------------------------------------------------ #
    # 2. BC rows: off-diagonal ≈ 0
    # ------------------------------------------------------------------ #
    def test_bc_rows_off_diagonal_zero(self, operator_setup):
        L0, idx, ngp = operator_setup
        L0_bc, bc_indices = apply_dirichlet(L0.copy(), idx, "bt", "uvwT", ngp)
        L0_bc_csr = L0_bc.tocsr()
        for i in bc_indices:
            row = L0_bc_csr.getrow(i).toarray().ravel()
            row[i] = 0.0  # exclude diagonal
            assert np.sum(np.abs(row)) < 1e-10, (
                f"Off-diagonal entries in BC row {i} should be zero"
            )

    # ------------------------------------------------------------------ #
    # 3. BC columns: off-diagonal ≈ 0
    # ------------------------------------------------------------------ #
    def test_bc_columns_off_diagonal_zero(self, operator_setup):
        L0, idx, ngp = operator_setup
        L0_bc, bc_indices = apply_dirichlet(L0.copy(), idx, "bt", "uvwT", ngp)
        L0_bc_csc = L0_bc.tocsc()
        for j in bc_indices:
            col = L0_bc_csc.getcol(j).toarray().ravel()
            col[j] = 0.0  # exclude diagonal
            assert np.sum(np.abs(col)) < 1e-10, (
                f"Off-diagonal entries in BC column {j} should be zero"
            )

    # ------------------------------------------------------------------ #
    # 4. bc_indices returned as sorted unique array
    # ------------------------------------------------------------------ #
    def test_bc_indices_sorted_unique(self, operator_setup):
        L0, idx, ngp = operator_setup
        _, bc_indices = apply_dirichlet(L0.copy(), idx, "bt", "uvwT", ngp)
        assert isinstance(bc_indices, np.ndarray)
        # Sorted
        assert np.all(bc_indices[:-1] <= bc_indices[1:]), (
            "bc_indices should be sorted"
        )
        # Unique
        assert len(bc_indices) == len(np.unique(bc_indices)), (
            "bc_indices should contain unique values"
        )

    # ------------------------------------------------------------------ #
    # 5. Index count: 4 variables × (20 bottom + 20 top) = 160
    # ------------------------------------------------------------------ #
    def test_bc_index_count(self, operator_setup):
        L0, idx, ngp = operator_setup
        _, bc_indices = apply_dirichlet(L0.copy(), idx, "bt", "uvwT", ngp)
        # Count expected: unique grid pts from bi ∪ ti × 4 variables
        n_boundary_pts = len(np.unique(np.concatenate([idx["bi"], idx["ti"]])))
        expected = 4 * n_boundary_pts
        assert len(bc_indices) == expected, (
            f"Expected {expected} BC indices, got {len(bc_indices)}"
        )

    # ------------------------------------------------------------------ #
    # 6. Interior rows/columns unchanged
    # ------------------------------------------------------------------ #
    def test_interior_rows_unchanged(self, operator_setup):
        L0, idx, ngp = operator_setup
        L0_orig = L0.copy()
        L0_bc, bc_indices = apply_dirichlet(L0.copy(), idx, "bt", "uvwT", ngp)

        L0_orig_csr = L0_orig.tocsr()
        L0_bc_csr = L0_bc.tocsr()

        # Pick an interior DOF that is not in bc_indices
        bc_set = set(bc_indices)
        n = L0.shape[0]
        interior_dof = None
        for i in range(n):
            if i not in bc_set:
                interior_dof = i
                break
        assert interior_dof is not None, "Could not find an interior DOF"

        # Check that the row is unchanged (excluding BC columns)
        orig_row = L0_orig_csr.getrow(interior_dof).toarray().ravel()
        bc_row = L0_bc_csr.getrow(interior_dof).toarray().ravel()

        # Interior entries at non-BC columns should be identical
        interior_cols = np.array([j for j in range(n) if j not in bc_set])
        np.testing.assert_allclose(
            bc_row[interior_cols],
            orig_row[interior_cols],
            atol=1e-14,
            err_msg="Interior row entries at non-BC columns should be unchanged",
        )

    # ------------------------------------------------------------------ #
    # 7. Empty borders → no-op
    # ------------------------------------------------------------------ #
    def test_empty_borders_noop(self, operator_setup):
        L0, idx, ngp = operator_setup
        L0_orig = L0.copy()
        L0_bc, bc_indices = apply_dirichlet(L0.copy(), idx, "", "uvwT", ngp)

        assert len(bc_indices) == 0, "Empty borders should produce no BC indices"
        # Operator should be unchanged
        diff = (L0_bc - L0_orig).tocsr()
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-14, (
            "Operator should be unchanged with empty borders"
        )

    # ------------------------------------------------------------------ #
    # 8. Invalid border character raises ValueError
    # ------------------------------------------------------------------ #
    def test_invalid_border_raises(self, operator_setup):
        L0, idx, ngp = operator_setup
        with pytest.raises(ValueError):
            apply_dirichlet(L0.copy(), idx, "x", "uvwT", ngp)

    # ------------------------------------------------------------------ #
    # 9. Invalid variable character raises ValueError
    # ------------------------------------------------------------------ #
    def test_invalid_variable_raises(self, operator_setup):
        L0, idx, ngp = operator_setup
        with pytest.raises(ValueError):
            apply_dirichlet(L0.copy(), idx, "bt", "q", ngp)

    # ------------------------------------------------------------------ #
    # 10. apply_dirichlet_spatial: BCs span both halves of doubled system
    # ------------------------------------------------------------------ #
    def test_spatial_bc_spans_both_halves(self, small_mesh_2d, uniform_baseflow):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        ngp = mesh.ngp
        ndof = 5 * ngp  # single-half DOF count

        Lc, R, Lw, idx = build_spatial_operator(mesh, bf)
        Lc_bc, R_bc, bc_indices = apply_dirichlet_spatial(
            Lc.copy(), R.copy(), idx, "bt", "uvwT", ngp
        )

        # bc_indices should contain indices in both halves [0, ndof) and [ndof, 2*ndof)
        lower_half = bc_indices[bc_indices < ndof]
        upper_half = bc_indices[bc_indices >= ndof]

        assert len(lower_half) > 0, "BC indices should include first half of system"
        assert len(upper_half) > 0, "BC indices should include second half of system"

        # The boundary indices in both halves should correspond (shifted by ndof)
        np.testing.assert_array_equal(
            lower_half,
            upper_half - ndof,
            err_msg="BC indices in upper half should mirror lower half shifted by ndof",
        )
