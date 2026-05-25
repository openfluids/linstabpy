# SPDX-License-Identifier: GPL-3.0-only
"""Tests for Dirichlet boundary condition enforcement."""

import numpy as np
import scipy.sparse as sp
import pytest

from linstabpy.operators import build_lnse_operator
from linstabpy.boundary_conditions import (
    analyze_bc_recipe,
    apply_dirichlet,
    apply_dirichlet_identity_pair,
    apply_dirichlet_spatial,
    apply_neumann,
    build_dirichlet_restriction_matrices,
)
from linstabpy.operators import build_spatial_operator


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


class TestDirichletAdapters:
    """Tests for MATLAB SetDirBC/SetDirichletBC parity helpers."""

    def test_build_dirichlet_restriction_matrices(self):
        n_dofs = 6
        dirichlet_indices = np.array([1, 4], dtype=int)

        t_cont, t_exp, free = build_dirichlet_restriction_matrices(
            n_dofs, dirichlet_indices
        )

        np.testing.assert_array_equal(free, np.array([0, 2, 3, 5], dtype=int))
        assert t_cont.shape == (4, 6)
        assert t_exp.shape == (6, 4)

        x = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        np.testing.assert_allclose(t_cont @ x, x[free])

        y = np.array([1.0, 2.0, 3.0, 4.0])
        x_full = t_exp @ y
        expected_full = np.zeros(n_dofs)
        expected_full[free] = y
        np.testing.assert_allclose(x_full, expected_full)

        np.testing.assert_allclose((t_cont @ t_exp).toarray(), np.eye(len(free)))

    def test_apply_dirichlet_identity_pair(self):
        lhs = np.arange(1.0, 17.0).reshape(4, 4)
        rhs = np.arange(21.0, 37.0).reshape(4, 4)
        lhs0 = lhs.copy()
        rhs0 = rhs.copy()

        lhs_bc, rhs_bc = apply_dirichlet_identity_pair(
            sp.csc_matrix(lhs),
            sp.csc_matrix(rhs),
            np.array([1, 3], dtype=int),
        )

        lhs_bc = lhs_bc.toarray()
        rhs_bc = rhs_bc.toarray()

        for i in [1, 3]:
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(lhs_bc[i], expected)
            np.testing.assert_allclose(rhs_bc[i], expected)

        np.testing.assert_allclose(lhs_bc[0], lhs0[0])
        np.testing.assert_allclose(lhs_bc[2], lhs0[2])
        np.testing.assert_allclose(rhs_bc[0], rhs0[0])
        np.testing.assert_allclose(rhs_bc[2], rhs0[2])

    def test_adapter_invalid_indices_raise(self):
        with pytest.raises(ValueError):
            build_dirichlet_restriction_matrices(4, np.array([5], dtype=int))

        lhs = sp.eye(4, format="csc")
        rhs = sp.eye(4, format="csc")
        with pytest.raises(ValueError):
            apply_dirichlet_identity_pair(lhs, rhs, np.array([-1], dtype=int))


class TestNeumannBC:
    """Tests for apply_neumann on the LNSE operator."""

    @staticmethod
    def _row_scales(
        border: str,
        gradX_x: np.ndarray,
        gradX_y: np.ndarray,
        gradY_x: np.ndarray,
        gradY_y: np.ndarray,
        imesh: int,
    ) -> tuple[float, float]:
        if border in ("l", "r"):
            sign = -1.0 if border == "l" else 1.0
            return sign * gradY_y[imesh], sign * (-gradY_x[imesh])
        sign = 1.0 if border == "b" else -1.0
        return sign * (-gradX_y[imesh]), sign * gradX_x[imesh]

    # ------------------------------------------------------------------ #
    # 1. No-op for empty borders
    # ------------------------------------------------------------------ #
    def test_neumann_empty_borders_noop(self, small_mesh_2d, uniform_baseflow):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
        L0_bc, bc_indices = apply_neumann(L0.copy(), mesh, idx, "", "uvwT")

        assert len(bc_indices) == 0, "Empty borders should produce no BC indices"
        diff = (L0_bc - L0).tocsr()
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-14, (
            "Operator should be unchanged with empty borders"
        )

    # ------------------------------------------------------------------ #
    # 2. Boundary row matches curvilinear-cofactor formula (left border)
    # ------------------------------------------------------------------ #
    def test_neumann_left_boundary_row_matches_formula(self, small_mesh_2d, uniform_baseflow):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
        ngp = mesh.ngp
        penalty = 3.0

        L0_bc, bc_indices = apply_neumann(
            L0.copy(), mesh, idx, "l", "u", penalty=penalty
        )

        dof = ngp + idx["li"][0]  # first left-border u-dof
        assert dof in set(bc_indices)

        imesh = int(idx["li"][0])
        x_flat = mesh.X.ravel(order="F")[mesh.used_ind]
        y_flat = mesh.Y.ravel(order="F")[mesh.used_ind]
        gradX_x = mesh.Dx @ x_flat
        gradX_y = mesh.Dy @ x_flat
        gradY_x = mesh.Dx @ y_flat
        gradY_y = mesh.Dy @ y_flat

        sx, sy = self._row_scales("l", gradX_x, gradX_y, gradY_x, gradY_y, imesh)
        expected = penalty * (sx * mesh.Dx.getrow(imesh) + sy * mesh.Dy.getrow(imesh))

        row = L0_bc.tocsc().getrow(dof).toarray().ravel()
        expected_row = np.zeros(L0.shape[0], dtype=row.dtype)
        expected_row[ngp : 2 * ngp] = expected.toarray().ravel()

        np.testing.assert_allclose(
            row,
            expected_row,
            atol=1e-11,
            err_msg="Neumann BC row should match mesh-derivative expression",
        )

    # ------------------------------------------------------------------ #
    # 3. Curvilinear branch coverage for all faces (l/r/b/t)
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "border",
        [("l"), ("r"), ("b"), ("t")],
    )
    def test_neumann_all_boundary_normals(
        self,
        small_mesh_2d,
        uniform_baseflow,
        border,
    ):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
        ngp = mesh.ngp
        penalty = 4.0

        border_key = {"l": "li", "r": "ri", "b": "bi", "t": "ti"}[border]
        mesh_id = int(idx[border_key][0])
        dof = ngp + mesh_id  # u-block

        L0_bc, bc_indices = apply_neumann(
            L0.copy(), mesh, idx, border, "u", penalty=penalty
        )
        assert dof in set(bc_indices)

        x_flat = mesh.X.ravel(order="F")[mesh.used_ind]
        y_flat = mesh.Y.ravel(order="F")[mesh.used_ind]
        gradX_x = mesh.Dx @ x_flat
        gradX_y = mesh.Dy @ x_flat
        gradY_x = mesh.Dx @ y_flat
        gradY_y = mesh.Dy @ y_flat

        sx, sy = self._row_scales(border, gradX_x, gradX_y, gradY_x, gradY_y, mesh_id)
        expected = penalty * (sx * mesh.Dx.getrow(mesh_id) + sy * mesh.Dy.getrow(mesh_id))

        row = L0_bc.tocsc().getrow(dof).toarray().ravel()
        expected_row = np.zeros(L0.shape[0], dtype=row.dtype)
        expected_row[ngp : 2 * ngp] = expected.toarray().ravel()
        np.testing.assert_allclose(
            row,
            expected_row,
            atol=1e-11,
            err_msg=f"Neumann row mismatch for border '{border}'",
        )

    # ------------------------------------------------------------------ #
    # 4. Penalty scaling
    # ------------------------------------------------------------------ #
    def test_neumann_penalty_scaling(self, small_mesh_2d, uniform_baseflow):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
        ngp = mesh.ngp
        mesh_id = int(idx["li"][0])
        dof = ngp + mesh_id

        p1 = 1.0
        p2 = 7.5
        L0_p1, _ = apply_neumann(L0.copy(), mesh, idx, "l", "u", penalty=p1)
        L0_p2, _ = apply_neumann(L0.copy(), mesh, idx, "l", "u", penalty=p2)

        row1 = L0_p1.tocsc().getrow(dof).toarray().ravel()
        row2 = L0_p2.tocsc().getrow(dof).toarray().ravel()
        np.testing.assert_allclose(
            row2,
            row1 * (p2 / p1),
            atol=1e-11,
            err_msg="Neumann rows should scale linearly with penalty",
        )

    # ------------------------------------------------------------------ #
    # 5. Variable and border validation
    # ------------------------------------------------------------------ #
    def test_neumann_invalid_inputs_raise(self, small_mesh_2d, uniform_baseflow):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        L0, idx = build_lnse_operator(mesh, bf, kx=0.0)

        with pytest.raises(ValueError):
            apply_neumann(L0.copy(), mesh, idx, "m", "u")

        with pytest.raises(ValueError):
            apply_neumann(L0.copy(), mesh, idx, "l", "q")

    # ------------------------------------------------------------------ #
    # 6. Corner duplicates are rejected as ambiguous multi-normal rules
    # ------------------------------------------------------------------ #
    def test_neumann_corner_overlap_raises(self, small_mesh_2d, uniform_baseflow):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        L0, idx = build_lnse_operator(mesh, bf, kx=0.0)

        with pytest.raises(ValueError, match="Ambiguous Neumann BC"):
            apply_neumann(L0.copy(), mesh, idx, "lt", "u", penalty=2.5)

    def test_neumann_cartesian_parity_curvilinear_flag_off_vs_on(
        self,
        small_mesh_2d,
        uniform_baseflow,
    ):
        """On a Cartesian mesh J11=J22=1 J12=J21=0 so both paths must agree to 1e-12."""
        from linstabpy.boundary_conditions.neumann import (
            _build_neumann_rows,
            _collect_neumann_indices,
        )

        mesh = small_mesh_2d
        bf = uniform_baseflow
        _, idx = build_lnse_operator(mesh, bf, kx=0.0)
        ngp = mesh.ngp
        penalty = 5.0
        index_set, index_mesh, border_tags = _collect_neumann_indices(
            idx, "b", "u", ngp
        )
        rows_off = _build_neumann_rows(
            mesh,
            index_set,
            index_mesh,
            border_tags,
            ngp,
            penalty,
            curvilinear_normals=False,
        )
        rows_on = _build_neumann_rows(
            mesh,
            index_set,
            index_mesh,
            border_tags,
            ngp,
            penalty,
            curvilinear_normals=True,
        )
        assert len(rows_off) == len(rows_on)
        for (dof_off, cols_off, vals_off), (dof_on, cols_on, vals_on) in zip(
            rows_off, rows_on
        ):
            assert dof_off == dof_on
            off = dict(zip(cols_off.astype(int), vals_off))
            on = dict(zip(cols_on.astype(int), vals_on))
            cols = np.array(sorted(set(off) | set(on)), dtype=int)
            np.testing.assert_allclose(
                np.array([off.get(int(col), 0.0) for col in cols]),
                np.array([on.get(int(col), 0.0) for col in cols]),
                atol=1e-12,
                err_msg=f"Cartesian: paths must agree at dof {dof_off}",
            )

    def test_neumann_sheared_mesh_correct_normal(self):
        """Sheared mesh X=xi+0.3*eta Y=eta. Right boundary curvilinear normal differs from axis-aligned by J21=0.3."""
        from linstabpy.mesh.mesh import Mesh
        from linstabpy.boundary_conditions.neumann import (
            _build_neumann_rows,
            _collect_neumann_indices,
        )

        n = 20
        mesh = Mesh.create((0.0, 1.0), (0.0, 1.0), n, n, fd_order=2)
        xi = mesh.X.copy()
        eta = mesh.Y.copy()
        X_new = xi + 0.3 * eta
        Y_new = eta.copy()
        mesh_def = mesh.deform(X_new=X_new, Y_new=Y_new)
        ngp = mesh_def.ngp
        idx = mesh_def.idx
        penalty = 1.0
        # Right boundary (constant-xi): curvilinear uses J22=1 J21=0.3 -> scale_y=-0.3
        # Axis-aligned uses gradY_y=1 gradY_x=0 -> scale_y=0
        # They differ by the J21=0.3 shear component
        index_set_r, index_mesh_r, border_tags_r = _collect_neumann_indices(
            idx, "r", "u", ngp
        )
        rows_curv_r = _build_neumann_rows(
            mesh_def,
            index_set_r,
            index_mesh_r,
            border_tags_r,
            ngp,
            penalty,
            curvilinear_normals=True,
        )
        rows_axis_r = _build_neumann_rows(
            mesh_def,
            index_set_r,
            index_mesh_r,
            border_tags_r,
            ngp,
            penalty,
            curvilinear_normals=False,
        )
        assert len(rows_curv_r) > 0, "No right-boundary rows found"
        diffs = []
        for (dof_c, cols_c, vals_c), (dof_a, cols_a, vals_a) in zip(
            rows_curv_r, rows_axis_r
        ):
            assert dof_c == dof_a
            curv = dict(zip(cols_c.astype(int), vals_c))
            axis = dict(zip(cols_a.astype(int), vals_a))
            cols = sorted(set(curv) | set(axis))
            diffs.append(
                float(
                    max(
                        abs(curv.get(int(col), 0.0) - axis.get(int(col), 0.0))
                        for col in cols
                    )
                )
            )
        assert max(diffs) > 0.1, (
            f"Curvilinear and axis-aligned must differ on sheared mesh (J21=0.3), "
            f"but max diff={max(diffs):.6f}"
        )

    def test_neumann_petsc_matches_scipy_under_curvilinear_flag(
        self,
        small_mesh_2d,
        uniform_baseflow,
    ):
        """scipy and PETSc paths must produce identical rows with curvilinear_normals=True."""
        pytest.importorskip("petsc4py")
        from petsc4py import PETSc
        from linstabpy.boundary_conditions import apply_neumann
        from linstabpy.boundary_conditions.neumann import apply_neumann_petsc

        mesh = small_mesh_2d
        bf = uniform_baseflow
        L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
        penalty = 3.0
        L0_scipy, bc_idx_scipy = apply_neumann(
            L0.copy(), mesh, idx, "b", "u", penalty=penalty, curvilinear_normals=True
        )
        csr = L0.tocsr()
        A = PETSc.Mat().createAIJ(
            size=csr.shape,
            csr=(csr.indptr, csr.indices, csr.data.astype(complex)),
        )
        A.setUp()
        A.assemble()
        A_petsc, bc_idx_petsc = apply_neumann_petsc(
            A, mesh, idx, "b", "u", penalty=penalty, curvilinear_normals=True
        )
        np.testing.assert_array_equal(
            np.sort(bc_idx_scipy),
            np.sort(bc_idx_petsc),
            err_msg="BC index sets must match between scipy and PETSc paths",
        )
        L0_scipy_csr = L0_scipy.tocsr()
        for dof in bc_idx_scipy:
            row_scipy = L0_scipy_csr.getrow(int(dof))
            cols_scipy = row_scipy.indices
            vals_scipy = row_scipy.data
            cols_petsc_raw, vals_petsc_raw = A_petsc.getRow(int(dof))
            cols_p = np.array(cols_petsc_raw, dtype=int)
            vals_p = np.array(vals_petsc_raw)
            for col, val_s in zip(cols_scipy, vals_scipy):
                mask = cols_p == col
                if mask.any():
                    np.testing.assert_allclose(
                        val_s,
                        vals_p[mask][0],
                        atol=1e-12,
                        err_msg=f"scipy/PETSc mismatch at dof={dof} col={col}",
                    )


class TestRecipeAnalysis:
    """Tests for sequential mixed BC recipe analysis."""

    def test_case1_recipe_overlap_and_effective_sets(self, small_mesh_2d, uniform_baseflow):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        _, idx = build_lnse_operator(mesh, bf, kx=0.0)
        ngp = mesh.ngp

        recipe = [
            {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
            {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
            {"type": "neumann", "borders": "b", "variables": "T"},
            {"type": "neumann", "borders": "r", "variables": "ruvwT"},
        ]
        analysis = analyze_bc_recipe(mesh, idx, recipe)

        left_bottom = np.intersect1d(idx["li"], idx["bi"])
        top_right = np.intersect1d(idx["ti"], idx["ri"])
        bottom_right = np.intersect1d(idx["bi"], idx["ri"])

        expected_overlap = np.sort(
            np.unique(
                np.concatenate(
                    [
                        4 * ngp + left_bottom,
                        np.concatenate([block * ngp + top_right for block in range(5)]),
                        np.concatenate([block * ngp + bottom_right for block in range(4)]),
                    ]
                )
            )
        )

        np.testing.assert_array_equal(analysis["overlap_indices"], expected_overlap)

        dirichlet_total = np.asarray(analysis["column_zeroed_dirichlet_indices"], dtype=int)
        effective_dirichlet = np.asarray(analysis["effective_dirichlet_indices"], dtype=int)
        expected_effective = np.setdiff1d(dirichlet_total, expected_overlap, assume_unique=True)
        np.testing.assert_array_equal(effective_dirichlet, expected_effective)

        assert len(analysis["rule_reports"]) == 4
        assert analysis["rule_reports"][2]["cross_type_overlap"].size == 1
        assert analysis["rule_reports"][3]["cross_type_overlap"].size == 9

    def test_case1_recipe_neumann_exclude_corners_removes_cross_type_overlap(
        self,
        small_mesh_2d,
        uniform_baseflow,
    ):
        mesh = small_mesh_2d
        bf = uniform_baseflow
        _, idx = build_lnse_operator(mesh, bf, kx=0.0)

        recipe = [
            {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
            {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
            {"type": "neumann", "borders": "b", "variables": "T", "exclude_corners": True},
            {"type": "neumann", "borders": "r", "variables": "ruvwT", "exclude_corners": True},
        ]
        analysis = analyze_bc_recipe(mesh, idx, recipe)

        assert analysis["overlap_indices"].size == 0
        assert analysis["rule_reports"][2]["cross_type_overlap"].size == 0
        assert analysis["rule_reports"][3]["cross_type_overlap"].size == 0


class TestDissipationBCInteraction:
    """T5 gate: dissipation must not smear Dirichlet rows after apply_bc_recipe."""

    _BC_RECIPE = [{"type": "dirichlet", "borders": "bt", "variables": "uvwT"}]
    _BC_PENALTY = 1e4
    _DISS_EPSILON = 0.1

    def _build_baseline(self, mesh, baseflow):
        """Build L0 (no dissipation) and apply BC recipe; return (L_bc, bc_indices)."""
        from linstabpy.boundary_conditions.recipe import apply_bc_recipe

        L0, idx = build_lnse_operator(mesh, baseflow, kx=0.0, dissipation_epsilon=0.0)
        L_bc, bc_indices = apply_bc_recipe(
            L0, mesh, idx, self._BC_RECIPE, penalty=self._BC_PENALTY
        )
        return L_bc, bc_indices, idx

    def _build_with_dissipation(self, baseflow):
        """Build L0 with dissipation and apply same BC recipe; return (L_bc, bc_indices)."""
        from linstabpy.mesh import Mesh
        from linstabpy.boundary_conditions.recipe import apply_bc_recipe

        mesh_diss = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=20,
            ny=30,
            fd_order=4,
            numerics_operator_dissipation={"enabled": True, "order_diss": 4},
        )
        L0, idx = build_lnse_operator(
            mesh_diss, baseflow, kx=0.0, dissipation_epsilon=self._DISS_EPSILON
        )
        L_bc, bc_indices = apply_bc_recipe(
            L0, mesh_diss, idx, self._BC_RECIPE, penalty=self._BC_PENALTY
        )
        return L_bc, bc_indices

    # ------------------------------------------------------------------ #
    # T5 gate: Dirichlet rows are penalty*I regardless of dissipation flag
    # ------------------------------------------------------------------ #
    def test_dirichlet_rows_unchanged_with_dissipation(self, small_mesh_2d, uniform_baseflow):
        """T5: Dirichlet rows must be exactly penalty*I after apply_bc_recipe,
        whether or not operator dissipation was subtracted before BCs were applied."""
        L_A, bc_indices_A, _ = self._build_baseline(small_mesh_2d, uniform_baseflow)
        L_B, bc_indices_B = self._build_with_dissipation(uniform_baseflow)

        assert len(bc_indices_A) > 0, "BC recipe produced no Dirichlet indices"

        # Check first 5 Dirichlet DOFs (avoid checking all 160 for speed)
        check_indices = bc_indices_A[:5]

        L_A_csr = L_A.tocsr()
        L_B_csr = L_B.tocsr()

        for i in check_indices:
            # --- variant A (no dissipation) ---
            row_A = L_A_csr.getrow(int(i)).toarray().ravel()
            off_diag_A = row_A.copy()
            off_diag_A[i] = 0.0
            assert row_A[i] == pytest.approx(self._BC_PENALTY, rel=1e-12), (
                f"Baseline: diagonal at BC DOF {i} should be {self._BC_PENALTY}, "
                f"got {row_A[i]}"
            )
            assert np.sum(np.abs(off_diag_A)) < 1e-10, (
                f"Baseline: off-diagonal entries at BC DOF {i} should be zero, "
                f"got sum_abs={np.sum(np.abs(off_diag_A))}"
            )
            nnz_A = int(L_A_csr.getrow(int(i)).nnz)
            assert nnz_A == 1, (
                f"Baseline: BC row {i} should have exactly 1 nonzero, got {nnz_A}"
            )

            # --- variant B (with dissipation) ---
            row_B = L_B_csr.getrow(int(i)).toarray().ravel()
            off_diag_B = row_B.copy()
            off_diag_B[i] = 0.0
            assert row_B[i] == pytest.approx(self._BC_PENALTY, rel=1e-12), (
                f"Dissipation-on: diagonal at BC DOF {i} should be {self._BC_PENALTY}, "
                f"got {row_B[i]}"
            )
            assert np.sum(np.abs(off_diag_B)) < 1e-10, (
                f"Dissipation-on: off-diagonal entries at BC DOF {i} should be zero, "
                f"got sum_abs={np.sum(np.abs(off_diag_B))}"
            )
            nnz_B = int(L_B_csr.getrow(int(i)).nnz)
            assert nnz_B == 1, (
                f"Dissipation-on: BC row {i} should have exactly 1 nonzero, got {nnz_B}"
            )

            # --- both variants must agree at the diagonal entry ---
            assert row_A[i] == pytest.approx(row_B[i], rel=1e-12), (
                f"Penalty diagonal at BC DOF {i} differs between baseline and "
                f"dissipation-on variants: {row_A[i]!r} vs {row_B[i]!r}"
            )

    # ------------------------------------------------------------------ #
    # Complementary check: dissipation IS visible in interior rows
    # ------------------------------------------------------------------ #
    def test_interior_rows_dissipated_correctly(self, small_mesh_2d, uniform_baseflow):
        """Dissipation must modify interior rows but leave Dirichlet rows intact.

        This is the positive half of the T5 gate: confirms that Q_diss_block
        is actually being subtracted (not silently skipped) and that the
        difference lives only in non-Dirichlet rows."""
        L_A, bc_indices_A, _ = self._build_baseline(small_mesh_2d, uniform_baseflow)
        L_B, bc_indices_B = self._build_with_dissipation(uniform_baseflow)

        n = L_A.shape[0]
        bc_set = set(int(k) for k in bc_indices_A)

        # Find the first interior DOF (not penalised by the Dirichlet recipe)
        interior_dof = None
        for candidate in range(n):
            if candidate not in bc_set:
                interior_dof = candidate
                break
        assert interior_dof is not None, "Could not find any interior DOF"

        L_A_csr = L_A.tocsr()
        L_B_csr = L_B.tocsr()

        row_A = L_A_csr.getrow(interior_dof).toarray().ravel()
        row_B = L_B_csr.getrow(interior_dof).toarray().ravel()

        max_diff = float(np.max(np.abs(row_B - row_A)))
        assert max_diff > 0.0, (
            f"Interior row {interior_dof} is identical between baseline and "
            f"dissipation-on variants — dissipation was not applied (max_diff={max_diff})"
        )
