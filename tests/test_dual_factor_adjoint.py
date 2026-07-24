# SPDX-License-Identifier: GPL-3.0-only
"""Dual-factor adjoint healing tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.linalg.factorize import _run_adjoint_self_test, factorize


def _operator(n: int = 6) -> sp.csc_matrix:
    """Create a small non-normal complex operator for testing."""
    diagonal = 3.0 + 0.25 * np.arange(n) + 1j * (0.15 - 0.03 * np.arange(n))
    lower = 0.04 - 0.02j + 0.005 * np.arange(n - 1)
    upper = -0.03 + 0.015j - 0.004 * np.arange(n - 1)
    far = np.full(n - 2, 0.01 + 0.01j, dtype=complex)
    return sp.diags([far, lower, diagonal, upper, -0.5 * far], [-2, -1, 0, 1, 2], format="csc", dtype=complex)


def test_mumps_adjoint_keeps_single_factor_path() -> None:
    """MUMPS adjoint should NOT trigger dual-factor healing (defect <= 1e-10)."""
    L0 = _operator(6)
    factor = factorize(L0)
    
    # Trigger adjoint test by using solve_adjoint
    rhs = np.random.randn(6) + 1j * np.random.randn(6)
    _ = factor.solve_adjoint(rhs)
    
    # Verify dual-factor was NOT activated (MUMPS is correct)
    assert factor.metrics["dual_factor_active"] is False
    assert factor.metrics["adjoint_self_test_defect"] < 1.0e-10
    
    # Verify adjoint identity holds to high precision
    rng = np.random.RandomState(123)
    a = rng.randn(6) + 1j * rng.randn(6)
    b = rng.randn(6) + 1j * rng.randn(6)
    
    x1 = factor.solve(a)
    x2 = factor.solve_adjoint(b)
    
    lhs = np.vdot(x1, b)  # conj(x1) . b
    rhs_val = np.vdot(a, x2)  # conj(a) . x2
    
    denominator = max(np.abs(lhs), np.abs(rhs_val), 1.0e-300)
    defect = np.abs(lhs - rhs_val) / denominator
    assert defect < 1.0e-12


def test_adjoint_identity_hold_for_mumps() -> None:
    """Verify that MUMPS solve/solve_adjoint satisfy the adjoint identity."""
    L0 = _operator(8)
    factor = factorize(L0)
    
    # Trigger the test
    rhs = np.random.randn(8) + 1j * np.random.randn(8)
    _ = factor.solve_adjoint(rhs)
    
    # Run several probe pairs
    rng = np.random.RandomState(125)
    for _ in range(5):
        a = rng.randn(8) + 1j * rng.randn(8)
        b = rng.randn(8) + 1j * rng.randn(8)
        
        x1 = factor.solve(a)
        x2 = factor.solve_adjoint(b)
        
        # Check: <A^{-1} a, b> == <a, A^{-H} b>
        lhs = np.vdot(x1, b)
        rhs_val = np.vdot(a, x2)
        
        denominator = max(np.abs(lhs), np.abs(rhs_val), 1.0e-300)
        defect = np.abs(lhs - rhs_val) / denominator
        assert defect < 1.0e-12, f"Adjoint identity violated: defect = {defect:.3e}"


def test_metrics_recorded() -> None:
    """Verify metrics are correctly populated."""
    L0 = _operator(6)
    factor = factorize(L0)
    
    # Check initial state
    assert "dual_factor_active" in factor.metrics
    assert "adjoint_self_test_defect" in factor.metrics
    assert "adjoint_oracle_residual" in factor.metrics
    
    # Initially should be False/0
    assert factor.metrics["dual_factor_active"] is False
    assert factor.metrics["adjoint_self_test_defect"] == 0.0
    assert factor.metrics["adjoint_oracle_residual"] == 0.0
    
    # Trigger the test
    rhs = np.random.randn(6) + 1j * np.random.randn(6)
    _ = factor.solve_adjoint(rhs)
    
    # Now metrics should be populated
    assert isinstance(factor.metrics["dual_factor_active"], (bool, np.bool_))
    assert isinstance(factor.metrics["adjoint_self_test_defect"], (float, np.floating))
    assert isinstance(factor.metrics["adjoint_oracle_residual"], (float, np.floating))
    assert factor.metrics["adjoint_self_test_defect"] >= 0.0
    assert factor.metrics["adjoint_oracle_residual"] >= 0.0


def test_solve_transpose_alias() -> None:
    """Verify backward-compatible solve_transpose alias works."""
    L0 = _operator(6)
    factor = factorize(L0)
    
    rhs = np.random.randn(6) + 1j * np.random.randn(6)
    result1 = factor.solve_transpose(rhs)
    result2 = factor.solve_adjoint(rhs)
    
    assert np.allclose(result1, result2)


def test_solve_transpose_vec_alias() -> None:
    """Verify backward-compatible solve_transpose_vec alias works."""
    from petsc4py import PETSc
    
    L0 = _operator(6)
    factor = factorize(L0)
    
    rhs_arr = np.random.randn(6) + 1j * np.random.randn(6)
    rhs_vec = PETSc.Vec().createWithArray(rhs_arr, comm=PETSc.COMM_SELF)
    
    result1_vec = factor.solve_transpose_vec(rhs_vec)
    result2_vec = factor.solve_adjoint_vec(rhs_vec)
    
    # Both should be valid PETSc vectors
    assert result1_vec.getSize() == rhs_vec.getSize()
    assert result2_vec.getSize() == rhs_vec.getSize()
    
    result1_vec.destroy()
    result2_vec.destroy()
    rhs_vec.destroy()


# --- Heal-path coverage: the cases the auto-detect/heal logic actually relies on.
# These exercise the dual-factor MECHANISM and the TRUTH-GATE on this box (MUMPS,
# no GPU). The end-to-end auto-trigger needs a backend with a defective complex
# transpose-solve (STRUMPACK) and is validated on the GPU box.


def test_forward_solve_on_AH_equals_adjoint() -> None:
    """The heal mechanism: a FORWARD solve on A^H reproduces the adjoint solve.

    This is the numerical claim the dual factorization rests on -- redirecting the
    adjoint to A_H_factor.solve must equal the (correct, for MUMPS) conj->
    solveTranspose->conj adjoint on A.
    """
    L0 = _operator(8)
    factor = factorize(L0)
    factor_H = factorize(L0.conj().T.tocsc())  # explicit A^H = conj(A).T

    rng = np.random.RandomState(7)
    for _ in range(4):
        b = rng.randn(8) + 1j * rng.randn(8)
        x_adjoint = factor.solve_adjoint(b)   # MUMPS adjoint on A (correct)
        x_forward_H = factor_H.solve(b)       # forward solve on A^H == adjoint
        assert np.allclose(x_adjoint, x_forward_H, atol=1.0e-10)


def test_self_test_detects_defective_adjoint_and_passes_true_adjoint() -> None:
    """The detector + truth-gate logic: the adjoint-identity self-test must read
    ~0 for the true Hermitian adjoint and large for a plain TRANSPOSE (which is
    NOT the Hilbert adjoint of a complex operator). This is exactly the check the
    in-build truth-gate uses to reject a wrongly-formed A^H.
    """
    L0 = _operator(8)
    factor = factorize(L0)

    # True adjoint A^H = conj(A).T -> identity holds.
    factor_H = factorize(L0.conj().T.tocsc())
    good_defect = _run_adjoint_self_test(factor.solve, factor_H.solve, 8, k=3)
    assert good_defect < 1.0e-10

    # Plain transpose A^T (no conjugation) -> NOT the adjoint for complex A.
    factor_T = factorize(L0.T.tocsc())
    bad_defect = _run_adjoint_self_test(factor.solve, factor_T.solve, 8, k=3)
    assert bad_defect > 1.0e-6


def test_heal_engages_when_transpose_self_test_raises() -> None:
    """Heal activates when adjoint self-test RAISES (divergent KSP), not just finite-defect."""
    import unittest.mock as mock
    from linstabpy.linalg.factorize import _run_adjoint_self_test
    
    L0 = _operator(6)
    
    # Count how many times the monkeypatched self-test is called
    call_count = [0]
    original_self_test = _run_adjoint_self_test
    
    def mock_self_test(solve_func, solve_adjoint_func, n, k=3):
        """First call raises (simulating STRUMPACK divergence); subsequent calls delegate to real impl."""
        call_count[0] += 1
        if call_count[0] == 1:
            # First call (the initial adjoint test in _maybe_run_adjoint_test_and_heal)
            raise RuntimeError("KSP DIVERGED_ITS: transpose solve failed to converge")
        # Subsequent calls (the truth-gate and any other validations) use the real implementation
        return original_self_test(solve_func, solve_adjoint_func, n, k)
    
    # Patch the self-test at the module level so factorize sees it
    with mock.patch('linstabpy.linalg.factorize._run_adjoint_self_test', side_effect=mock_self_test):
        factor = factorize(L0)
        
        # Trigger adjoint use, which runs the self-test
        rhs = np.random.randn(6) + 1j * np.random.randn(6)
        result = factor.solve_adjoint(rhs)
        
        # Should not raise; heal should have engaged
        assert result is not None
        assert factor.metrics["dual_factor_active"] is True
        
        # The self-test was called at least twice: once (raised), then by truth-gate
        assert call_count[0] >= 2
    
    # Verify the adjoint result is sensible: match it against dense numpy solve
    factor_unpatched = factorize(L0)
    rhs = np.random.randn(6) + 1j * np.random.randn(6)
    
    # Get the adjoint via the healed dual-factor path
    x_healed = factor.solve_adjoint(rhs)
    
    # Get the adjoint via the unpatched (MUMPS, correct) path
    x_reference = factor_unpatched.solve_adjoint(rhs)
    
    # They should match to reasonable tolerance
    assert np.allclose(x_healed, x_reference, atol=1.0e-9, rtol=1.0e-8)
    
    factor_unpatched.destroy()
    factor.destroy()
