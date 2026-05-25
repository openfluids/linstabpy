# SPDX-License-Identifier: GPL-3.0-only
"""Tests for harmonic resolvent infrastructure.

C1: HarmonicLayout — block indexing, offsets, slices, lifting
C2: build_harmonic_weights — block-diagonal replication, positivity
C3: assemble_harmonic_operator — block-Toeplitz structure, time derivative,
    steady-base-flow reduction, real-baseflow conjugacy
C4: apply_harmonic_bcs — lifted BC enforcement across harmonic blocks
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from linstabpy.linear_operators import (
    HarmonicLayout,
    apply_harmonic_bcs,
    assemble_harmonic_operator,
    build_harmonic_weights,
    harmonic_resolvent_analysis,
    projected_harmonic_resolvent_analysis,
)

# ---------------------------------------------------------------------------
# C1: HarmonicLayout
# ---------------------------------------------------------------------------


class TestHarmonicLayout:
    """Test HarmonicLayout bookkeeping."""

    def test_nblocks(self):
        layout = HarmonicLayout(Nh=3, n=10, omega=1.0)
        assert layout.nblocks == 7  # 2*3 + 1

    def test_total_dof(self):
        layout = HarmonicLayout(Nh=2, n=5, omega=1.0)
        assert layout.N == 5 * 5  # (2*2+1) * 5

    def test_harmonics(self):
        layout = HarmonicLayout(Nh=2, n=5, omega=1.0)
        expected = np.array([-2, -1, 0, 1, 2])
        np.testing.assert_array_equal(layout.harmonics, expected)

    def test_block_index(self):
        layout = HarmonicLayout(Nh=2, n=5, omega=1.0)
        # k=-2 → block 0, k=-1 → block 1, k=0 → block 2, k=1 → block 3, k=2 → block 4
        assert layout.block_index(-2) == 0
        assert layout.block_index(-1) == 1
        assert layout.block_index(0) == 2
        assert layout.block_index(1) == 3
        assert layout.block_index(2) == 4

    def test_block_index_out_of_range(self):
        layout = HarmonicLayout(Nh=2, n=5, omega=1.0)
        with pytest.raises(ValueError, match="outside"):
            layout.block_index(3)

    def test_block_offset(self):
        layout = HarmonicLayout(Nh=2, n=10, omega=1.0)
        assert layout.block_offset(-2) == 0
        assert layout.block_offset(0) == 20
        assert layout.block_offset(2) == 40

    def test_block_slice(self):
        layout = HarmonicLayout(Nh=1, n=8, omega=1.0)
        # k=-1 → block 0 → slice(0, 8)
        s = layout.block_slice(-1)
        assert s == slice(0, 8)
        # k=0 → block 1 → slice(8, 16)
        s = layout.block_slice(0)
        assert s == slice(8, 16)
        # k=1 → block 2 → slice(16, 24)
        s = layout.block_slice(1)
        assert s == slice(16, 24)

    def test_lift_indices(self):
        layout = HarmonicLayout(Nh=2, n=10, omega=1.0)
        base_idx = np.array([0, 3, 7])
        # k=1 → offset = 30
        lifted = layout.lift_indices(base_idx, k=1)
        np.testing.assert_array_equal(lifted, np.array([30, 33, 37]))

    def test_harmonic_frequencies(self):
        layout = HarmonicLayout(Nh=2, n=5, omega=0.5)
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(layout.harmonic_frequencies(), expected)


# ---------------------------------------------------------------------------
# C2: build_harmonic_weights
# ---------------------------------------------------------------------------


class TestBuildHarmonicWeights:
    """Test build_harmonic_weights."""

    def test_shape(self):
        layout = HarmonicLayout(Nh=2, n=10, omega=1.0)
        W_diag = np.ones(10)
        W_harm, invW_harm = build_harmonic_weights(layout, W_diag)
        assert W_harm.shape == (50,)
        assert invW_harm.shape == (50,)

    def test_block_replication(self):
        """Each harmonic block should have the same weight vector."""
        layout = HarmonicLayout(Nh=2, n=4, omega=1.0)
        W_diag = np.array([1.0, 2.0, 3.0, 4.0])
        W_harm, _ = build_harmonic_weights(layout, W_diag)
        for b in range(layout.nblocks):
            block = W_harm[b * 4 : (b + 1) * 4]
            np.testing.assert_array_equal(block, W_diag)

    def test_inverse(self):
        layout = HarmonicLayout(Nh=1, n=5, omega=1.0)
        rng = np.random.default_rng(700)
        W_diag = np.abs(rng.standard_normal(5)) + 0.1
        W_harm, invW_harm = build_harmonic_weights(layout, W_diag)
        np.testing.assert_allclose(W_harm * invW_harm, np.ones(layout.N))

    def test_size_mismatch(self):
        layout = HarmonicLayout(Nh=1, n=5, omega=1.0)
        with pytest.raises(ValueError, match="W_diag has"):
            build_harmonic_weights(layout, np.ones(7))


# ---------------------------------------------------------------------------
# C3: assemble_harmonic_operator
# ---------------------------------------------------------------------------


class TestAssembleHarmonicOperator:
    """Test assemble_harmonic_operator."""

    @staticmethod
    def _random_sparse(n, rng):
        """Create a random dense-ish sparse matrix."""
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return sp.csc_matrix(A)

    def test_steady_baseflow_diagonal(self):
        """For steady base flow (only L_0), the harmonic operator should be
        block-diagonal with blocks L_0 - i*k*omega*I.

        This is THE critical sanity check: harmonic resolvent at each harmonic k
        reduces to the standard resolvent at frequency k*omega.
        """
        rng = np.random.default_rng(710)
        n = 8
        Nh = 2
        omega = 0.5
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0 = self._random_sparse(n, rng)
        # Only the mean operator — steady base flow.
        T = assemble_harmonic_operator(
            layout, [L0], real_baseflow=True, include_time_derivative=True
        )
        T_dense = T.toarray()

        # Check each diagonal block.
        for k in layout.harmonics:
            s = layout.block_slice(k)
            block = T_dense[s, s]
            expected = L0.toarray() - 1j * k * omega * np.eye(n)
            np.testing.assert_allclose(
                block, expected, atol=1e-14, err_msg=f"Diagonal block k={k} mismatch"
            )

        # Off-diagonal blocks should be zero.
        for i in range(layout.nblocks):
            for j in range(layout.nblocks):
                if i == j:
                    continue
                si = layout.block_slice(layout.harmonics[i])
                sj = layout.block_slice(layout.harmonics[j])
                block = T_dense[si, sj]
                assert np.linalg.norm(block) < 1e-14, f"Off-diagonal block ({i},{j}) is not zero"

    def test_block_toeplitz_structure(self):
        """The (i,j) block should be L_{i-j} when |i-j| <= Nf, else zero."""
        rng = np.random.default_rng(720)
        n = 6
        Nh = 3
        Nf = 2
        omega = 1.0
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0 = self._random_sparse(n, rng)
        L1 = self._random_sparse(n, rng)
        L2 = self._random_sparse(n, rng)

        # A_hat only (no time derivative) for clean Toeplitz check.
        A_hat = assemble_harmonic_operator(
            layout,
            [L0, L1, L2],
            real_baseflow=True,
            include_time_derivative=False,
        )
        A_dense = A_hat.toarray()

        # Build full harmonic dict for reference.
        full_harmonics = {0: L0.toarray(), 1: L1.toarray(), 2: L2.toarray()}
        full_harmonics[-1] = L1.toarray().conj()
        full_harmonics[-2] = L2.toarray().conj()

        for i in range(layout.nblocks):
            for j in range(layout.nblocks):
                diff = i - j
                si = layout.block_slice(layout.harmonics[i])
                sj = layout.block_slice(layout.harmonics[j])
                block = A_dense[si, sj]
                if diff in full_harmonics:
                    np.testing.assert_allclose(
                        block,
                        full_harmonics[diff],
                        atol=1e-14,
                        err_msg=f"Block ({i},{j}), diff={diff} mismatch",
                    )
                else:
                    assert np.linalg.norm(block) < 1e-14, (
                        f"Block ({i},{j}), diff={diff} should be zero"
                    )

    def test_real_baseflow_conjugacy(self):
        """For real base flow, L_{-k} = conj(L_k). Verify blocks reflect this."""
        rng = np.random.default_rng(730)
        n = 5
        Nh = 2
        Nf = 1
        omega = 1.0
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0 = self._random_sparse(n, rng)
        L1 = self._random_sparse(n, rng)

        A_hat = assemble_harmonic_operator(
            layout,
            [L0, L1],
            real_baseflow=True,
            include_time_derivative=False,
        )
        A_dense = A_hat.toarray()

        # Block (i, i+1) = L_{-1} = conj(L_1).
        # Block (i+1, i) = L_{+1} = L_1.
        # Check a specific pair: block (0,1) has diff = -1 → L_{-1} = conj(L_1).
        s0 = layout.block_slice(layout.harmonics[0])
        s1 = layout.block_slice(layout.harmonics[1])
        block_neg1 = A_dense[s0, s1]  # diff = 0 - 1 = -1
        np.testing.assert_allclose(block_neg1, L1.toarray().conj(), atol=1e-14)

        block_pos1 = A_dense[s1, s0]  # diff = 1 - 0 = +1
        np.testing.assert_allclose(block_pos1, L1.toarray(), atol=1e-14)

    def test_time_derivative_diagonal(self):
        """The time derivative subtracts i*k*omega from each diagonal block."""
        rng = np.random.default_rng(740)
        n = 6
        Nh = 2
        omega = 0.7
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0 = self._random_sparse(n, rng)

        A_hat = assemble_harmonic_operator(
            layout,
            [L0],
            real_baseflow=True,
            include_time_derivative=False,
        )
        T = assemble_harmonic_operator(
            layout,
            [L0],
            real_baseflow=True,
            include_time_derivative=True,
        )
        diff = (A_hat - T).toarray()

        # diff should be diag(i*k*omega) repeated per block.
        expected = np.zeros(layout.N, dtype=complex)
        for k in layout.harmonics:
            s = layout.block_slice(k)
            expected[s.start : s.stop] = 1j * k * omega

        np.testing.assert_allclose(diff, np.diag(expected), atol=1e-14)

    def test_nf_exceeds_nh_raises(self):
        """Nf > Nh should raise ValueError."""
        n = 5
        layout = HarmonicLayout(Nh=1, n=n, omega=1.0)
        L0 = sp.eye(n, format="csc")
        L1 = sp.eye(n, format="csc")
        L2 = sp.eye(n, format="csc")

        # 3 harmonics → Nf=2, but Nh=1.
        with pytest.raises(ValueError, match="exceeds"):
            assemble_harmonic_operator(layout, [L0, L1, L2])

    def test_non_real_baseflow(self):
        """With real_baseflow=False, all 2*Nf+1 coefficients must be provided."""
        rng = np.random.default_rng(750)
        n = 4
        Nh = 2
        Nf = 1
        omega = 1.0
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L_neg1 = self._random_sparse(n, rng)
        L0 = self._random_sparse(n, rng)
        L_pos1 = self._random_sparse(n, rng)

        A_hat = assemble_harmonic_operator(
            layout,
            [L_neg1, L0, L_pos1],
            real_baseflow=False,
            include_time_derivative=False,
        )
        A_dense = A_hat.toarray()

        # Block (i, i+1) has diff=-1 → L_{-1}.
        s0 = layout.block_slice(layout.harmonics[0])
        s1 = layout.block_slice(layout.harmonics[1])
        block = A_dense[s0, s1]
        np.testing.assert_allclose(block, L_neg1.toarray(), atol=1e-14)

        # Block (i+1, i) has diff=+1 → L_{+1}.
        block = A_dense[s1, s0]
        np.testing.assert_allclose(block, L_pos1.toarray(), atol=1e-14)

    def test_output_shape_and_format(self):
        """Assembled operator should be CSC with correct dimensions."""
        n = 5
        Nh = 2
        layout = HarmonicLayout(Nh=Nh, n=n, omega=1.0)
        L0 = sp.eye(n, format="csc", dtype=complex)

        T = assemble_harmonic_operator(layout, [L0])
        assert T.shape == (layout.N, layout.N)
        assert sp.issparse(T)
        assert T.format == "csc"

    def test_harmonic_resolvent_equivalence(self):
        """For steady base flow, each harmonic block's resolvent gain should match
        the standard resolvent at frequency k*omega.

        This is the end-to-end validation: assemble the harmonic generator,
        invert each block, and compare singular values against inv(L0 - ikωI).
        """
        rng = np.random.default_rng(760)
        n = 8
        Nh = 2
        omega = 0.5
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        # Well-conditioned operator.
        L0_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        L0_np += n * np.eye(n)
        L0 = sp.csc_matrix(L0_np)

        T = assemble_harmonic_operator(layout, [L0])
        T_dense = T.toarray()

        # Invert the full harmonic generator.
        T_inv = la.inv(T_dense)

        for k in layout.harmonics:
            s = layout.block_slice(k)
            # Block of the inverse.
            block_inv = T_inv[s, s]
            # Reference: inv(L0 - ikωI).
            ref_inv = la.inv(L0_np - 1j * k * omega * np.eye(n))

            # Compare leading singular values.
            sv_block = la.svdvals(block_inv)
            sv_ref = la.svdvals(ref_inv)

            np.testing.assert_allclose(
                sv_block, sv_ref, rtol=1e-10, err_msg=f"Singular values mismatch at k={k}"
            )


# ---------------------------------------------------------------------------
# C4: apply_harmonic_bcs
# ---------------------------------------------------------------------------


class TestApplyHarmonicBCs:
    """Test harmonic BC lifting and enforcement."""

    @staticmethod
    def _random_sparse(n, rng):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return sp.csc_matrix(A)

    def test_bc_rows_cols_zeroed(self):
        """BC rows and columns should be zero except for penalty diagonal."""
        rng = np.random.default_rng(800)
        n = 10
        Nh = 1
        omega = 1.0
        penalty = 1e4
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0 = self._random_sparse(n, rng)
        T = assemble_harmonic_operator(layout, [L0])

        # Mark DOFs 0, 3, 7 as boundary in the base system.
        bc_base = np.array([0, 3, 7])
        T_bc = apply_harmonic_bcs(T, layout, bc_base, penalty=penalty)
        T_dense = T_bc.toarray()

        # Check every harmonic block's lifted BC indices.
        for k in layout.harmonics:
            lifted = layout.lift_indices(bc_base, k)
            for i in lifted:
                # Row should be all zeros except diagonal = penalty.
                row = T_dense[i, :].copy()
                row[i] = 0.0
                assert np.linalg.norm(row) < 1e-14, (
                    f"BC row {i} (k={k}) has non-zero off-diagonal entries"
                )
                assert abs(T_dense[i, i] - penalty) < 1e-14, (
                    f"BC diagonal at {i} (k={k}) is not {penalty}"
                )
                # Column should also be all zeros except diagonal.
                col = T_dense[:, i].copy()
                col[i] = 0.0
                assert np.linalg.norm(col) < 1e-14, (
                    f"BC column {i} (k={k}) has non-zero off-diagonal entries"
                )

    def test_interior_dofs_unchanged(self):
        """Non-BC DOFs should be unmodified by the BC application."""
        rng = np.random.default_rng(810)
        n = 8
        Nh = 1
        omega = 1.0
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0 = self._random_sparse(n, rng)
        T = assemble_harmonic_operator(layout, [L0])
        T_orig = T.toarray().copy()

        bc_base = np.array([0, 2])
        T_bc = apply_harmonic_bcs(T, layout, bc_base)
        T_dense = T_bc.toarray()

        # Build set of all lifted BC indices.
        all_bc = set()
        for k in layout.harmonics:
            for i in layout.lift_indices(bc_base, k):
                all_bc.add(int(i))

        # Interior rows/cols should be unchanged.
        for i in range(layout.N):
            if i in all_bc:
                continue
            for j in range(layout.N):
                if j in all_bc:
                    continue
                assert T_dense[i, j] == T_orig[i, j], f"Interior entry ({i},{j}) was modified"

    def test_off_diagonal_coupling_zeroed(self):
        """BC rows in off-diagonal (cross-harmonic) blocks should be zeroed."""
        rng = np.random.default_rng(820)
        n = 6
        Nh = 2
        Nf = 1
        omega = 1.0
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0 = self._random_sparse(n, rng)
        L1 = self._random_sparse(n, rng)
        T = assemble_harmonic_operator(
            layout,
            [L0, L1],
            real_baseflow=True,
        )

        # Before BC: off-diagonal blocks should have nonzero entries.
        T_dense_pre = T.toarray()
        s0 = layout.block_slice(-2)
        s1 = layout.block_slice(-1)
        assert np.linalg.norm(T_dense_pre[s0, s1]) > 1e-10, (
            "Precondition: off-diagonal block should be nonzero before BC"
        )

        bc_base = np.array([0, 1])
        T_bc = apply_harmonic_bcs(T, layout, bc_base)
        T_dense = T_bc.toarray()

        # After BC: BC rows in off-diagonal blocks should be zero.
        for k_row in layout.harmonics:
            lifted_rows = layout.lift_indices(bc_base, k_row)
            for k_col in layout.harmonics:
                if k_row == k_col:
                    continue
                s_col = layout.block_slice(k_col)
                for r in lifted_rows:
                    row_slice = T_dense[r, s_col]
                    assert np.linalg.norm(row_slice) < 1e-14, (
                        f"BC row {r} in cross-block (k_row={k_row}, k_col={k_col}) is not zero"
                    )

    def test_empty_bc_indices(self):
        """Empty BC index set should leave operator unchanged."""
        rng = np.random.default_rng(830)
        n = 5
        layout = HarmonicLayout(Nh=1, n=n, omega=1.0)
        L0 = self._random_sparse(n, rng)
        T = assemble_harmonic_operator(layout, [L0])
        T_orig = T.toarray().copy()

        T_bc = apply_harmonic_bcs(T, layout, np.array([], dtype=int))
        np.testing.assert_array_equal(T_bc.toarray(), T_orig)

    def test_bc_solveability(self):
        """Harmonic operator with BCs should be invertible (well-conditioned)."""
        rng = np.random.default_rng(840)
        n = 8
        Nh = 1
        omega = 0.5
        penalty = 1e4
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        L0_np += n * np.eye(n)
        L0 = sp.csc_matrix(L0_np)

        T = assemble_harmonic_operator(layout, [L0])
        bc_base = np.array([0, 1])
        T_bc = apply_harmonic_bcs(T, layout, bc_base, penalty=penalty)

        # Should be invertible.
        T_dense = T_bc.toarray()
        det = la.det(T_dense)
        assert abs(det) > 1e-10, "Harmonic operator with BCs is singular"

    def test_consistent_with_per_block_application(self):
        """Harmonic BCs should match applying Dirichlet penalty to each block
        independently — the gold-standard reference for correctness."""
        rng = np.random.default_rng(850)
        n = 6
        Nh = 2
        Nf = 1
        omega = 1.0
        penalty = 1e4
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0 = self._random_sparse(n, rng)
        L1 = self._random_sparse(n, rng)
        bc_base = np.array([0, 2, 5])

        # Method 1: Apply BCs to assembled harmonic operator.
        T = assemble_harmonic_operator(layout, [L0, L1], real_baseflow=True)
        T_bc = apply_harmonic_bcs(T, layout, bc_base, penalty=penalty)

        # Method 2: Build reference manually — apply BCs to each
        # coefficient THEN assemble.
        # For the mean operator L0: zero rows/cols at bc, set penalty.
        # For coupling operators L_k (k≠0): zero rows/cols at bc, NO penalty.
        def _apply_bc_to_block(mat, add_penalty):
            """Apply Dirichlet BCs to a single n×n block."""
            M = mat.toarray().copy()
            M[bc_base, :] = 0.0
            M[:, bc_base] = 0.0
            if add_penalty:
                M[bc_base, bc_base] = penalty
            return sp.csc_matrix(M)

        # For block-Toeplitz Dirichlet: zero BC rows/cols across ALL blocks,
        # add penalty only on diagonal blocks.
        # But apply_harmonic_bcs does full-row zeroing + penalty on global diagonal.
        # This is equivalent to: for each block (i,j), zero BC rows/cols,
        # and add penalty only when i==j.
        # Build reference dense matrix.
        T_ref = T.toarray().copy()
        all_lifted = np.concatenate([layout.lift_indices(bc_base, k) for k in layout.harmonics])
        all_lifted = np.unique(all_lifted)
        T_ref[all_lifted, :] = 0.0
        T_ref[:, all_lifted] = 0.0
        T_ref[all_lifted, all_lifted] = penalty

        np.testing.assert_allclose(
            T_bc.toarray(),
            T_ref,
            atol=1e-14,
            err_msg="Harmonic BCs don't match manual per-row application",
        )


# ---------------------------------------------------------------------------
# C5: Harmonic resolvent analysis
# ---------------------------------------------------------------------------


class TestHarmonicResolventAnalysis:
    """Test harmonic_resolvent_analysis via operator-composed HtrH + SLEPc EPS."""

    def test_steady_baseflow_gains_match_standard(self):
        """For steady base flow, harmonic resolvent gains should be the union
        of standard resolvent gains at each frequency k*omega.

        This is the critical end-to-end validation: the full SLEPc EPS
        pipeline on the operator-composed HtrH must recover the same gains
        as a dense SVD of inv(L0 - ikωI) at each harmonic.
        """
        rng = np.random.default_rng(900)
        n = 10
        Nh = 1  # 3 blocks: k = -1, 0, +1
        omega = 0.5
        n_modes = 6  # 2 per block × 3 blocks

        # Well-conditioned operator.
        L0_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        L0_np += n * np.eye(n)
        L0 = sp.csc_matrix(L0_np)

        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)
        W_diag = np.abs(rng.standard_normal(n)) + 0.1  # positive weights

        # Assemble steady harmonic generator (block-diagonal).
        T = assemble_harmonic_operator(layout, [L0])

        # Run harmonic resolvent.
        gains, responses, forcings = harmonic_resolvent_analysis(
            T,
            layout,
            n_modes,
            W_diag,
        )

        # Build reference: collect the top singular values from each
        # standard resolvent at frequency k*omega.
        ref_gains_all = []
        for k in layout.harmonics:
            L_sigma = L0_np - 1j * k * omega * np.eye(n)
            H_k = la.inv(L_sigma)
            # Weighted SVD: singular values of W^{1/2} H W^{-1/2}.
            W_sqrt = np.diag(np.sqrt(W_diag))
            invW_sqrt = np.diag(1.0 / np.sqrt(W_diag))
            H_weighted = W_sqrt @ H_k @ invW_sqrt
            sv = la.svdvals(H_weighted)
            ref_gains_all.extend(sv.tolist())

        ref_gains_all = np.sort(ref_gains_all)[::-1]
        ref_top = ref_gains_all[:n_modes]

        np.testing.assert_allclose(
            gains,
            ref_top,
            rtol=1e-8,
            err_msg="Harmonic gains don't match standard resolvent reference",
        )

    def test_output_shapes(self):
        """Check that output arrays have correct shapes."""
        rng = np.random.default_rng(910)
        n = 8
        Nh = 1
        omega = 1.0
        n_modes = 3
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        L0_np += n * np.eye(n)
        L0 = sp.csc_matrix(L0_np)

        T = assemble_harmonic_operator(layout, [L0])
        W_diag = np.ones(n)

        gains, responses, forcings = harmonic_resolvent_analysis(
            T,
            layout,
            n_modes,
            W_diag,
        )

        assert gains.shape == (n_modes,)
        assert responses.shape == (layout.N, n_modes)
        assert forcings.shape == (layout.N, n_modes)

    def test_gains_sorted_descending(self):
        """Gains should be sorted in descending order."""
        rng = np.random.default_rng(920)
        n = 8
        Nh = 1
        omega = 0.5
        n_modes = 4
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        L0_np += n * np.eye(n)
        L0 = sp.csc_matrix(L0_np)

        T = assemble_harmonic_operator(layout, [L0])
        W_diag = np.abs(rng.standard_normal(n)) + 0.1

        gains, _, _ = harmonic_resolvent_analysis(T, layout, n_modes, W_diag)

        # Check monotonically non-increasing.
        for i in range(len(gains) - 1):
            assert gains[i] >= gains[i + 1] - 1e-14

    def test_forcing_response_relationship(self):
        """Response should satisfy u = T^{-1} v / sigma."""
        rng = np.random.default_rng(930)
        n = 8
        Nh = 1
        omega = 0.5
        n_modes = 3
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)

        L0_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        L0_np += n * np.eye(n)
        L0 = sp.csc_matrix(L0_np)

        T = assemble_harmonic_operator(layout, [L0])
        W_diag = np.abs(rng.standard_normal(n)) + 0.1

        gains, responses, forcings = harmonic_resolvent_analysis(
            T,
            layout,
            n_modes,
            W_diag,
        )

        # Dense inverse for verification.
        T_inv = la.inv(T.toarray())

        for i in range(n_modes):
            sigma = gains[i]
            v_i = forcings[:, i]
            u_i = responses[:, i]

            # u_i should be proportional to T^{-1} v_i.
            Hinv_v = T_inv @ v_i
            # The response is T^{-1} v / sigma (from the implementation).
            u_ref = Hinv_v / sigma

            # Allow for arbitrary complex phase in eigenvector.
            # Check that |<u_i, u_ref>| / (||u_i|| ||u_ref||) ≈ 1.
            cos_angle = abs(np.vdot(u_i, u_ref)) / (np.linalg.norm(u_i) * np.linalg.norm(u_ref))
            assert cos_angle > 1 - 1e-8, (
                f"Mode {i}: response doesn't match T^{{-1}} forcing / sigma "
                f"(cos_angle={cos_angle:.2e})"
            )


# ---------------------------------------------------------------------------
# C6: Projected harmonic resolvent analysis
# ---------------------------------------------------------------------------


class TestProjectedHarmonicResolvent:
    """Test projected_harmonic_resolvent_analysis."""

    @staticmethod
    def _build_near_singular_system(n, Nh, omega, rng):
        """Build a harmonic system with a known null vector.

        Constructs T such that T @ null_vec ≈ 0 by building a random
        operator and then modifying it to have a small eigenvalue in a
        known direction.
        """
        layout = HarmonicLayout(Nh=Nh, n=n, omega=omega)
        N = layout.N

        # Build a random well-conditioned base operator.
        L0_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        L0_np += n * np.eye(n)
        L0 = sp.csc_matrix(L0_np)

        T = assemble_harmonic_operator(layout, [L0])
        T_dense = T.toarray()

        # Create a near-null direction (simulating dQ/dt).
        null_vec = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        null_vec /= np.linalg.norm(null_vec)

        # Modify T to have a near-zero eigenvalue in the null_vec direction.
        # T_new = T - (T @ null_vec) null_vec^H + eps * null_vec null_vec^H
        # This makes T_new @ null_vec = eps * null_vec.
        eps = 1e-10
        Tv = T_dense @ null_vec
        T_dense = (
            T_dense - np.outer(Tv, null_vec.conj()) + eps * np.outer(null_vec, null_vec.conj())
        )
        T_mod = sp.csc_matrix(T_dense)

        return layout, T_mod, null_vec, n

    def test_projection_removes_divergent_gain(self):
        """The unprojected resolvent should have a very large gain (from the
        near-null direction), while the projected resolvent should not."""
        rng = np.random.default_rng(950)
        n = 8
        Nh = 1
        omega = 0.5
        n_modes = 3

        layout, T_mod, null_vec, _ = self._build_near_singular_system(
            n,
            Nh,
            omega,
            rng,
        )
        W_diag = np.ones(n)

        # Unprojected: the leading gain should be huge (≈ 1/eps = 1e10).
        gains_unproj, _, _ = harmonic_resolvent_analysis(
            T_mod,
            layout,
            n_modes,
            W_diag,
        )
        assert gains_unproj[0] > 1e6, (
            f"Expected very large unprojected gain, got {gains_unproj[0]:.2e}"
        )

        # Projected: the divergent gain should be removed.
        gains_proj, _, _ = projected_harmonic_resolvent_analysis(
            T_mod,
            layout,
            n_modes,
            W_diag,
            dQdt=null_vec,
        )
        # The projected leading gain should be orders of magnitude smaller.
        assert gains_proj[0] < gains_unproj[0] * 1e-4, (
            f"Projected gain {gains_proj[0]:.2e} not much smaller than "
            f"unprojected {gains_unproj[0]:.2e}"
        )

    def test_projected_gains_match_dense_reference(self):
        """Projected gains should match a dense reference computation
        using explicit matrix projectors."""
        rng = np.random.default_rng(960)
        n = 8
        Nh = 1
        omega = 0.5
        n_modes = 3

        layout, T_mod, null_vec, _ = self._build_near_singular_system(
            n,
            Nh,
            omega,
            rng,
        )
        N = layout.N
        W_diag = np.abs(rng.standard_normal(n)) + 0.1

        # Compute projected gains via dense linear algebra.
        T_dense = T_mod.toarray()
        W_harm = np.tile(W_diag, layout.nblocks)
        invW_harm = 1.0 / W_harm

        # Build projectors.
        v = null_vec / np.linalg.norm(null_vec)
        w = la.solve(T_dense.conj().T, v)
        w /= np.linalg.norm(w)

        Pr = np.eye(N) - np.outer(v, v.conj())
        Pd = np.eye(N) - np.outer(w, w.conj())

        # H_proj = Pr @ inv(T) @ Pd
        T_inv = la.inv(T_dense)
        H_proj = Pr @ T_inv @ Pd

        # Weighted SVD: sv of W^{1/2} H_proj W^{-1/2}.
        W_sqrt = np.diag(np.sqrt(W_harm))
        invW_sqrt = np.diag(1.0 / np.sqrt(W_harm))
        H_weighted = W_sqrt @ H_proj @ invW_sqrt
        ref_sv = la.svdvals(H_weighted)
        ref_gains = np.sort(ref_sv)[::-1][:n_modes]

        # Compute via projected harmonic resolvent.
        gains_proj, _, _ = projected_harmonic_resolvent_analysis(
            T_mod,
            layout,
            n_modes,
            W_diag,
            dQdt=null_vec,
        )

        np.testing.assert_allclose(
            gains_proj, ref_gains, rtol=1e-6, err_msg="Projected gains don't match dense reference"
        )

    def test_output_shapes(self):
        """Check output array shapes."""
        rng = np.random.default_rng(970)
        n = 8
        Nh = 1
        omega = 0.5
        n_modes = 3

        layout, T_mod, null_vec, _ = self._build_near_singular_system(
            n,
            Nh,
            omega,
            rng,
        )
        W_diag = np.ones(n)

        gains, responses, forcings = projected_harmonic_resolvent_analysis(
            T_mod,
            layout,
            n_modes,
            W_diag,
            dQdt=null_vec,
        )

        assert gains.shape == (n_modes,)
        assert responses.shape == (layout.N, n_modes)
        assert forcings.shape == (layout.N, n_modes)

    def test_forcings_orthogonal_to_phase_direction(self):
        """Projected forcings should lie in the range of Pd, i.e.,
        they should be (nearly) orthogonal to w."""
        rng = np.random.default_rng(980)
        n = 8
        Nh = 1
        omega = 0.5
        n_modes = 3

        layout, T_mod, null_vec, _ = self._build_near_singular_system(
            n,
            Nh,
            omega,
            rng,
        )
        N = layout.N
        W_diag = np.ones(n)

        gains, responses, forcings = projected_harmonic_resolvent_analysis(
            T_mod,
            layout,
            n_modes,
            W_diag,
            dQdt=null_vec,
        )

        # Compute w (the domain projector direction).
        T_dense = T_mod.toarray()
        v = null_vec / np.linalg.norm(null_vec)
        w = la.solve(T_dense.conj().T, v)
        w /= np.linalg.norm(w)

        for i in range(n_modes):
            overlap = abs(np.vdot(w, forcings[:, i]))
            assert overlap < 1e-6, (
                f"Forcing mode {i} has overlap {overlap:.2e} with w (should be near zero)"
            )
