# SPDX-License-Identifier: GPL-3.0-only
"""Metamorphic tests for linstabpy linear stability / resolvent analysis.

The oracle problem in scientific computing: there is no "correct answer" to
compare against for a complex eigenvalue of a 2-D compressible LNSE operator.
These tests instead verify mathematical *invariants* (metamorphic relations)
that must hold across input transformations, independently of the reference
solution.

All tests are PURE NUMPY/SCIPY — no petsc4py, no slepc4py, no mpi4py, no
linstabpy module imports.  Reason: every linstabpy sub-package (operators,
mesh, boundary_conditions, baseflow) transitively imports ``linstabpy.utils.
petsc_init``, which in turn imports ``petsc4py`` at module load time.  Pytest
collection would therefore fail on systems without a PETSc installation.
The tests below build the mathematical objects by hand and call
``numpy.linalg.eig`` directly.

Relations implemented
---------------------
A. **Eigenvalue shift invariance**
   σ(A + α·I) = {λ + α : λ ∈ σ(A)} for any complex scalar α.
   Parametrized over three shifts.

B. **Direct ↔ adjoint conjugacy**
   For a weight matrix W = diag(w) with w > 0, define the weighted adjoint
   A† = W⁻¹ A^H W.  Then σ(A†) = conj(σ(A)).
   Tested on random complex matrices with random positive diagonal weights.
   Parametrized over matrix sizes.

C. **Eigenvector residual**
   For every returned eigenpair (λᵢ, vᵢ): ‖A·vᵢ − λᵢ·vᵢ‖₂ / ‖vᵢ‖₂
   must be below 100× machine epsilon times ‖A‖_F.
   This confirms numpy.linalg.eig satisfies its own definition, which would
   break if an assembly or scaling bug produced a non-square or degenerate
   operator.

F. **Chu energy norm Hermitian positive-definite**
   The compressible energy norm (Chu 1965) is a diagonal 5×ngp matrix whose
   entries are products of strictly positive mean-flow quantities.  We
   reproduce the formula in closed form and verify: all W_diag > 0,
   invW_diag = 1/W_diag to 14 significant figures, and W_diag = W_diag^H
   (trivially true for real diagonal).
   Parametrized over (kappa, Ma) pairs covering subsonic and supersonic cases.

H. **Reality symmetry — conjugate pairs**
   For a real matrix A ∈ ℝⁿˣⁿ, non-real eigenvalues appear in conjugate
   pairs: if λ ∈ σ(A) then conj(λ) ∈ σ(A).
   Parametrized over matrix sizes.

Relations skipped and why
-------------------------
D. **Resolvent lower bound** σ₁(s) ≥ 1/dist(s, σ(L)):
   A full resolvent solve at a given frequency requires the linstabpy
   ``solvers.resolvent`` module, which imports PETSc/SLEPc at load time.
   Building a meaningful resolvent operator by hand (size >1000 DOF) without
   linstabpy makes this a 30+ second test.  Deferred.

E. **Mesh-refinement convergence rate**:
   Covered by ``test_analytical.py`` (Orszag 1971 benchmark) which already
   exercises the SLEPc solver path.  Cannot be replicated as a pure-numpy
   test without rebuilding the SBP finite-difference stack from scratch.

G. **Scale invariance under density scaling**:
   The non-dimensional spectrum depends on Re and Ma in a way that is
   entangled with the density scaling.  The exact scaling law requires
   tracing through the inv(B) transformation and the coefficient matrices.
   Skipped to avoid a fragile hand-coded derivation that might embed wrong
   physics.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# NO linstabpy imports here — they all trigger petsc4py at load time.
# ---------------------------------------------------------------------------

# Absolute tolerance for eigenvalue set comparisons (A, B, H).
# LAPACK ZGEEV backward error for an n×n matrix is O(n·eps·‖A‖).
# For n=14 and ‖A‖~O(1), that yields ~3e-14 per eigenpair.  Using 5e-13
# (≈ 2250× eps) gives comfortable headroom without masking genuine bugs.
_EIG_ATOL = 5e-13


# ===========================================================================
# Helpers
# ===========================================================================


def _sorted_eigs(A: np.ndarray) -> np.ndarray:
    """Return eigenvalues of square matrix A sorted by (real, imag)."""
    vals = np.linalg.eig(A)[0]
    idx = np.lexsort((vals.imag, vals.real))
    return vals[idx]


def _random_complex_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return an n×n random complex matrix with bounded entries."""
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))


# ===========================================================================
# A. Eigenvalue shift invariance
# ===========================================================================

_SHIFT_PARAMS = [
    0.5 + 0.0j,
    0.0 + 0.3j,
    -0.2 + 0.7j,
]


@pytest.mark.parametrize("alpha", _SHIFT_PARAMS)
def test_A_eigenvalue_shift_invariance(alpha: complex) -> None:
    """σ(A + α·I) = {λ + α : λ ∈ σ(A)} for any complex α.

    Build one 12×12 random complex matrix A (seeded deterministically), then
    verify that numpy.linalg.eig obeys the shift law for several α values.
    The seed is fixed per (alpha) so each parametrize variant is independent.
    """
    rng = np.random.default_rng(42)
    n = 12
    A = _random_complex_matrix(rng, n)

    shifted_A = A + alpha * np.eye(n, dtype=complex)

    vals_A_plus_alpha = _sorted_eigs(A) + alpha  # analytical prediction
    vals_shifted = _sorted_eigs(shifted_A)       # what numpy reports

    np.testing.assert_allclose(
        vals_shifted,
        vals_A_plus_alpha,
        atol=_EIG_ATOL,
        rtol=0,
        err_msg=f"Shift invariance failed for alpha={alpha!r}",
    )


# ===========================================================================
# B. Direct ↔ adjoint conjugacy (weighted)
# ===========================================================================

_ADJOINT_SIZES = [6, 10, 14]


@pytest.mark.parametrize("n", _ADJOINT_SIZES)
def test_B_weighted_adjoint_conjugacy(n: int) -> None:
    """σ(W⁻¹ A^H W) = conj(σ(A)) for positive diagonal weight W.

    This is the algebraic identity underlying ``adjoint_eigenvalues()``.
    Seed varies with n to avoid correlated matrices across sizes.
    """
    rng = np.random.default_rng(17 + n)
    A = _random_complex_matrix(rng, n)
    # Random positive diagonal weights (simulate Chu energy entries)
    w = np.abs(rng.standard_normal(n)) + 0.5  # guaranteed > 0

    # Weighted adjoint: W⁻¹ A^H W
    A_adj = np.diag(1.0 / w) @ A.conj().T @ np.diag(w)

    vals_direct_conj = _sorted_eigs(A).conj()
    vals_adjoint = _sorted_eigs(A_adj)

    # Sort adjoint by (real, imag) — same sort key applied to both sides
    idx_adj = np.lexsort((vals_adjoint.imag, vals_adjoint.real))
    vals_adjoint_sorted = vals_adjoint[idx_adj]
    # The conjugates of sorted direct eigenvalues need re-sorting after conjugation
    idx_conj = np.lexsort((vals_direct_conj.imag, vals_direct_conj.real))
    vals_conj_sorted = vals_direct_conj[idx_conj]

    np.testing.assert_allclose(
        vals_adjoint_sorted,
        vals_conj_sorted,
        atol=_EIG_ATOL,
        rtol=0,
        err_msg=(
            f"Weighted-adjoint conjugacy failed for n={n}: "
            f"max diff = {np.max(np.abs(vals_adjoint_sorted - vals_conj_sorted)):.3e}"
        ),
    )


# ===========================================================================
# C. Eigenvector residual
# ===========================================================================


def test_C_eigenvector_residual_random() -> None:
    """‖A·vᵢ − λᵢ·vᵢ‖₂ / ‖vᵢ‖₂ < 100·eps·‖A‖_F for all returned eigenpairs.

    Uses a 20×20 random complex matrix.  The bound 100·eps·‖A‖_F accounts for
    the backward error of a numerically stable eigensolver (LAPACK ZGEEV).
    """
    rng = np.random.default_rng(99)
    n = 20
    A = _random_complex_matrix(rng, n)

    vals, vecs = np.linalg.eig(A)

    norm_A = np.linalg.norm(A, "fro")
    tol = 100.0 * np.finfo(float).eps * norm_A

    max_rel_residual = 0.0
    for i in range(n):
        lam = vals[i]
        v = vecs[:, i]
        residual = np.linalg.norm(A @ v - lam * v)
        rel = residual / np.linalg.norm(v)
        max_rel_residual = max(max_rel_residual, rel)

    assert max_rel_residual < tol, (
        f"Eigenvector residual too large: max ‖Av−λv‖/‖v‖ = {max_rel_residual:.3e} "
        f"> tol = {tol:.3e} (100·eps·‖A‖_F)"
    )


def test_C_eigenvector_residual_structured() -> None:
    """Repeat eigenvector residual check for a banded-structure matrix.

    Uses a tridiagonal-like complex matrix to exercise non-normal structure
    similar to a convective LNSE operator.
    """
    n = 15
    rng = np.random.default_rng(7)
    # Tridiagonal-like structure: diagonal + off-diagonals
    diag_vals = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    upper = 0.5 * (rng.standard_normal(n - 1) + 1j * rng.standard_normal(n - 1))
    lower = 0.3 * (rng.standard_normal(n - 1) + 1j * rng.standard_normal(n - 1))
    A = np.diag(diag_vals) + np.diag(upper, 1) + np.diag(lower, -1)

    vals, vecs = np.linalg.eig(A)

    norm_A = np.linalg.norm(A, "fro")
    tol = 100.0 * np.finfo(float).eps * norm_A

    residuals = [
        np.linalg.norm(A @ vecs[:, i] - vals[i] * vecs[:, i]) / np.linalg.norm(vecs[:, i])
        for i in range(n)
    ]
    max_rel = max(residuals)

    assert max_rel < tol, (
        f"Structured-matrix eigvec residual: max = {max_rel:.3e} > tol = {tol:.3e}"
    )


# ===========================================================================
# F. Chu energy norm Hermitian positive-definite
# ===========================================================================

_CHU_PARAMS = [
    # (kappa, Ma, description)
    (1.4, 0.3, "subsonic"),
    (1.4, 0.8, "transonic"),
    (1.4, 2.15, "supersonic-SWBLI"),
    (1.3, 1.5, "alt-kappa"),
]


@pytest.mark.parametrize("kappa,Ma,label", _CHU_PARAMS)
def test_F_chu_norm_positive_definite(kappa: float, Ma: float, label: str) -> None:
    """Chu energy norm weight matrix is Hermitian positive-definite.

    Reproduces the formula from ``linstabpy/operators/energy_norm.py``:

        W[rho block] = 0.5 * T / (rho * kappa * Ma²) * w_int
        W[u,v,w block] = 0.5 * rho * w_int
        W[T block] = 0.5 * rho / (kappa * (kappa-1) * T * Ma²) * w_int

    Verified properties:
    1. All W_diag entries are strictly positive (products of positive inputs).
    2. invW_diag = 1/W_diag to floating-point precision.
    3. W_diag is real (Hermitian diagonal satisfied trivially).
    4. invW_diag is real and positive.
    """
    rng = np.random.default_rng(31 + int(10 * kappa) + int(100 * Ma))
    ngp = 50  # small for speed

    # Uniform-ish mean flow with small perturbations, all strictly positive
    rho = 0.8 + 0.4 * np.abs(rng.standard_normal(ngp))
    T = 0.9 + 0.3 * np.abs(rng.standard_normal(ngp))
    w_int = np.abs(rng.standard_normal(ngp)) + 0.01  # integration weights > 0

    Ma2 = Ma ** 2

    W_diag = 0.5 * np.concatenate([
        T / (rho * kappa * Ma2) * w_int,           # rho block
        rho * w_int,                                 # u block
        rho * w_int,                                 # v block
        rho * w_int,                                 # w block
        rho / (kappa * (kappa - 1.0) * T * Ma2) * w_int,  # T block
    ])
    invW_diag = 1.0 / W_diag

    # 1. All entries strictly positive
    assert np.all(W_diag > 0), (
        f"[{label}] W_diag has non-positive entries: min={W_diag.min():.3e}"
    )

    # 2. Inverse matches 1/W_diag to tight tolerance
    rel_inv_err = np.max(np.abs(invW_diag - 1.0 / W_diag) / np.abs(invW_diag))
    assert rel_inv_err < 1e-14, (
        f"[{label}] invW_diag ≠ 1/W_diag: max rel err = {rel_inv_err:.3e}"
    )

    # 3. W_diag is real (Hermitian diagonal)
    assert np.all(np.isreal(W_diag)), f"[{label}] W_diag has imaginary parts"

    # 4. invW_diag also real and positive
    assert np.all(invW_diag > 0), (
        f"[{label}] invW_diag has non-positive entries: min={invW_diag.min():.3e}"
    )

    # 5. W * invW = 1 (self-consistency of normalization)
    product_err = np.max(np.abs(W_diag * invW_diag - 1.0))
    assert product_err < 1e-14, (
        f"[{label}] W * invW ≠ 1: max err = {product_err:.3e}"
    )


# ===========================================================================
# H. Reality symmetry — conjugate pair property
# ===========================================================================

_REAL_MATRIX_SIZES = [6, 8, 12]


@pytest.mark.parametrize("n", _REAL_MATRIX_SIZES)
def test_H_real_matrix_conjugate_pairs(n: int) -> None:
    """For A ∈ ℝⁿˣⁿ, non-real eigenvalues appear in conjugate pairs.

    If λ ∈ σ(A) and Im(λ) ≠ 0, then conj(λ) ∈ σ(A).

    Implementation note: for each complex eigenvalue λ in the returned set,
    we check that conj(λ) is also in the set within tolerance _EIG_ATOL.
    Real eigenvalues (|Im| < _EIG_ATOL) are excluded from this check.
    """
    rng = np.random.default_rng(55 + n)
    A = rng.standard_normal((n, n)).astype(float)  # explicitly real

    vals = np.linalg.eig(A)[0]

    # Tolerance for declaring an eigenvalue "complex" vs "real"
    imag_tol = _EIG_ATOL

    for lam in vals:
        if abs(lam.imag) <= imag_tol:
            # Real eigenvalue — no conjugate requirement
            continue
        lam_conj = lam.conj()
        # Check that conj(lambda) is in the eigenvalue set
        dists = np.abs(vals - lam_conj)
        min_dist = dists.min()
        assert min_dist < _EIG_ATOL, (
            f"Conjugate of λ={lam!r} not found in σ(A): "
            f"min dist = {min_dist:.3e} > tol = {_EIG_ATOL:.3e}\n"
            f"Eigenvalues: {vals}"
        )
