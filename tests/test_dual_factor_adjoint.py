# SPDX-License-Identifier: GPL-3.0-only
"""Dual-factor adjoint healing tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
import scipy.sparse as sp

from linstabpy.linalg.factorize import _run_adjoint_self_test, factorize


def _operator(n: int = 6) -> sp.csc_matrix:
    """Create a small non-normal complex operator for testing."""
    diagonal = 3.0 + 0.25 * np.arange(n) + 1j * (0.15 - 0.03 * np.arange(n))
    lower = 0.04 - 0.02j + 0.005 * np.arange(n - 1)
    upper = -0.03 + 0.015j - 0.004 * np.arange(n - 1)
    far = np.full(n - 2, 0.01 + 0.01j, dtype=complex)
    return sp.diags(
        [far, lower, diagonal, upper, -0.5 * far], [-2, -1, 0, 1, 2], format="csc", dtype=complex
    )


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

    lhs = np.vdot(x1, b)
    rhs_val = np.vdot(a, x2)

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

    # Now metrics must hold values
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
# no GPU).
# The end-to-end auto-trigger needs a backend with a defective complex
# transpose-solve (STRUMPACK).
# The GPU box validates that path.


def test_forward_solve_on_AH_equals_adjoint() -> None:
    """The heal mechanism: a FORWARD solve on A^H reproduces the adjoint solve.

    This is the numerical claim the dual factorization rests on.
    Redirecting the adjoint to A_H_factor.solve must equal the MUMPS adjoint
    on A (conj, then solveTranspose, then conj).
    """
    L0 = _operator(8)
    factor = factorize(L0)
    factor_H = factorize(L0.conj().T.tocsc())  # explicit A^H = conj(A).T

    rng = np.random.RandomState(7)
    for _ in range(4):
        b = rng.randn(8) + 1j * rng.randn(8)
        x_adjoint = factor.solve_adjoint(b)  # MUMPS adjoint on A (correct)
        x_forward_H = factor_H.solve(b)  # forward solve on A^H == adjoint
        assert np.allclose(x_adjoint, x_forward_H, atol=1.0e-10)


def test_self_test_detects_defective_adjoint_and_passes_true_adjoint() -> None:
    """Check the adjoint-identity self-test on a true adjoint and a transpose.

    The self-test must read ~0 for the true Hermitian adjoint.
    It must read large for a plain TRANSPOSE.
    A plain transpose is not the Hilbert adjoint of a complex operator.
    The in-build truth-gate uses this check to reject a mis-formed A^H.
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

    # Count how many times the monkeypatched self-test runs.
    call_count = [0]
    original_self_test = _run_adjoint_self_test

    def mock_self_test(solve_func, solve_adjoint_func, n, k=3, probe_family=None):
        """First call raises (simulating STRUMPACK divergence); subsequent calls delegate to real impl."""
        call_count[0] += 1
        if call_count[0] == 1:
            # First call (the initial adjoint test in _maybe_run_adjoint_test_and_heal)
            raise RuntimeError("KSP DIVERGED_ITS: transpose solve failed to converge")
        # Subsequent calls (the truth-gate and any other validations) use the real implementation
        return original_self_test(solve_func, solve_adjoint_func, n, k, probe_family=probe_family)

    # Patch the self-test at the module level so factorize sees it
    with mock.patch(
        "linstabpy.linalg.factorize._run_adjoint_self_test", side_effect=mock_self_test
    ):
        factor = factorize(L0)

        # Trigger adjoint use, which runs the self-test
        rhs = np.random.randn(6) + 1j * np.random.randn(6)
        result = factor.solve_adjoint(rhs)

        # Must not raise; heal must engage.
        assert result is not None
        assert factor.metrics["dual_factor_active"] is True

        # The self-test ran at least twice: once (raised), then by truth-gate.
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


def _shifted_same_pattern(L0: sp.csc_matrix, shift: complex) -> sp.csc_matrix:
    """Same nonzero pattern as ``L0``, different values (a diagonal shift)."""
    n = L0.shape[0]
    return (L0 + sp.diags(np.full(n, shift), format="csc")).tocsc()


def _heal_forcing_self_test():
    """Self-test stand-in: the first (native) call raises, every later call is real."""
    from linstabpy.linalg.factorize import _run_adjoint_self_test as real_self_test

    calls = [0]

    def mock_self_test(solve_func, solve_adjoint_func, n, k=3, probe_family=None):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("simulated defective native transpose (STRUMPACK-like)")
        return real_self_test(solve_func, solve_adjoint_func, n, k, probe_family=probe_family)

    return mock_self_test


def test_refactorize_keeps_healed_adjoint_on_the_new_operator() -> None:
    """After refactorize the healed K^H factor must belong to the NEW operator.

    Regression: ``refactorize_operator`` refreshed the forward factor only.
    It left ``factor_H`` (and the pristine operator the heal transposes) at
    the FIRST operator.
    A line of shifted operators on a healed backend then forward-solved
    K_new and adjoint-solved K_old^H from the second point on.
    """
    import unittest.mock as mock

    n = 6
    L0 = _operator(n)
    L1 = _shifted_same_pattern(L0, 0.7 - 0.4j)
    rng = np.random.default_rng(7)
    rhs = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    with mock.patch(
        "linstabpy.linalg.factorize._run_adjoint_self_test", side_effect=_heal_forcing_self_test()
    ):
        factor = factorize(L0)
        factor.solve_adjoint(rhs)  # first adjoint use: the heal engages on L0
        assert factor.metrics["dual_factor_active"] is True
        factor.refactorize(L1)
        x = factor.solve_adjoint(rhs)
        y = factor.solve(rhs)

    resid_new = np.linalg.norm(L1.conj().T @ x - rhs) / np.linalg.norm(rhs)
    resid_old = np.linalg.norm(L0.conj().T @ x - rhs) / np.linalg.norm(rhs)
    assert resid_new < 1.0e-10, resid_new
    assert resid_old > 1.0e-3, resid_old  # a stale L0^H factor would pass THIS one instead
    assert np.linalg.norm(L1 @ y - rhs) / np.linalg.norm(rhs) < 1.0e-10
    assert factor.metrics["dual_factor_active"] is True
    assert factor.metrics["adjoint_refactor_count"] == 1
    assert factor.metrics["adjoint_healed_defect"] <= 1.0e-8
    factor.destroy()


def test_heal_after_refactorize_transposes_the_current_operator() -> None:
    """A heal that first runs AFTER a refactorize must build K^H of the current operator."""
    import unittest.mock as mock

    n = 6
    L0 = _operator(n)
    L1 = _shifted_same_pattern(L0, -0.5 + 0.9j)
    rng = np.random.default_rng(11)
    rhs = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    with mock.patch(
        "linstabpy.linalg.factorize._run_adjoint_self_test", side_effect=_heal_forcing_self_test()
    ):
        factor = factorize(L0)
        factor.refactorize(L1)  # no adjoint use yet: the heal has not run
        x = factor.solve_adjoint(rhs)  # heal runs now, on the refactorized operator

    resid_new = np.linalg.norm(L1.conj().T @ x - rhs) / np.linalg.norm(rhs)
    resid_old = np.linalg.norm(L0.conj().T @ x - rhs) / np.linalg.norm(rhs)
    assert resid_new < 1.0e-10, resid_new
    assert resid_old > 1.0e-3, resid_old
    assert factor.metrics["dual_factor_active"] is True
    assert factor.metrics["adjoint_refactor_count"] == 0  # heal came after the refactorize
    factor.destroy()


def test_heal_and_refactorize_leave_the_callers_petsc_operator_untouched() -> None:
    """The heal transposes the caller's operator to build K^H.

    With a PETSc Mat, ``hermitianTranspose()`` without ``out`` transposes
    IN PLACE and returns the same object.
    The caller's K silently became K^H, and the refactorize path then
    destroyed it.
    Both the first heal and a refactorize must leave the caller's matrices
    exactly as they were.
    """
    import unittest.mock as mock

    from petsc4py import PETSc

    from linstabpy.utils.petsc_core import scipy_to_petsc

    n = 6
    L0 = _operator(n)
    L1 = _shifted_same_pattern(L0, 0.3 - 0.2j)
    K0 = scipy_to_petsc(L0)
    K1 = scipy_to_petsc(L1)
    K0_ref = K0.copy()
    K1_ref = K1.copy()
    rng = np.random.default_rng(3)
    rhs = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def unchanged(K: PETSc.Mat, K_ref: PETSc.Mat) -> bool:
        D = K.copy()
        D.axpy(-1.0, K_ref)
        d = D.norm()
        D.destroy()
        return d == 0.0

    with mock.patch(
        "linstabpy.linalg.factorize._run_adjoint_self_test", side_effect=_heal_forcing_self_test()
    ):
        factor = factorize(K0)
        y0 = factor.solve_adjoint(rhs)  # heal engages: builds K0^H
        assert factor.metrics["dual_factor_active"] is True
        assert unchanged(K0, K0_ref), "the heal transposed the caller's operator in place"
        factor.refactorize(K1)
        y1 = factor.solve_adjoint(rhs)
        assert unchanged(K1, K1_ref), "refactorize transposed the caller's new operator in place"
        assert unchanged(K0, K0_ref)

    assert np.linalg.norm(L0.conj().T @ y0 - rhs) / np.linalg.norm(rhs) < 1.0e-10
    assert np.linalg.norm(L1.conj().T @ y1 - rhs) / np.linalg.norm(rhs) < 1.0e-10
    factor.destroy()
    # the caller still owns its matrices: destroying them must not double-free
    for K in (K0, K1, K0_ref, K1_ref):
        K.destroy()
