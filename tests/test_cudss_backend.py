# SPDX-License-Identifier: GPL-3.0-only
"""Tests for the cuDSS GPU sparse-direct backend.

Two tiers.  The first tier runs everywhere: configuration guards, input
validation, equilibration arithmetic and the probe-family plumbing need
neither a GPU nor PETSc (all imports here are lazy on PETSc by design).
The second tier needs a working cuDSS stack (nvmath wheels plus an NVIDIA
GPU) and skips with an announced reason when it is absent; it holds the
differential oracles against MUMPS and SciPy.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linstabpy.linalg import backends
from linstabpy.linalg.cudss_factor import CudssSolver, cudss_available, equilibrate

# The private helpers are the units under test; importing them here is the point.
from linstabpy.linalg.factorize import (
    _run_adjoint_self_test,
    factorize,
)
from linstabpy.resolvent.operator import (
    _make_production_probe_family,
)

_NVMATH_INSTALLED = importlib.util.find_spec("nvmath") is not None
_HAS_PETSC = (
    importlib.util.find_spec("mpi4py") is not None
    and importlib.util.find_spec("petsc4py") is not None
)

requires_petsc = pytest.mark.skipif(
    not _HAS_PETSC, reason="SKIPPED: missing PETSc stack (mpi4py + petsc4py)"
)


def _test_operator(n: int = 300, seed: int = 7) -> sp.csr_matrix:
    """Build a well-conditioned complex nonsymmetric sparse operator.

    Returns
    -------
    scipy.sparse.csr_matrix
        A diagonally dominant operator with a full diagonal.
    """
    rng = np.random.RandomState(seed)
    density = min(5.0 / n, 1.0)
    real = sp.random(n, n, density=density, random_state=rng, format="csr")
    imag = sp.random(n, n, density=density, random_state=rng, format="csr")
    base = real + 1j * imag
    return (base + sp.eye(n, dtype=complex, format="csr") * (4.0 + 0.5j)).tocsr()


def _cudss_solver_cfg(**overrides: object) -> dict[str, dict[str, object]]:
    linsolve: dict[str, object] = {
        "type": "iterative",
        "ksp_type": "fgmres",
        "pc_type": "lu",
        "ksp_pc_side": "right",
        "factor_backend": "cudss",
        "ksp_rtol": 1.0e-12,
        "ksp_max_it": 200,
    }
    linsolve.update(overrides)
    return {"linsolve": linsolve}


# ---------------------------------------------------------------------------
# Tier 1: no GPU needed
# ---------------------------------------------------------------------------


@requires_petsc
def test_apply_factor_backend_refuses_cudss() -> None:
    """The PETSc options seam must reject cudss with a pointer to the real route."""
    from petsc4py import PETSc

    A = PETSc.Mat().createAIJ(size=(1, 1), comm=PETSc.COMM_SELF)
    A.setValue(0, 0, 1.0 + 0.0j)
    A.assemble()
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    try:
        ksp.setOperators(A)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        with pytest.raises(ValueError, match="not a PETSc factor type"):
            backends.apply_factor_backend(pc, "cudss")
    finally:
        ksp.destroy()
        A.destroy()


@requires_petsc
def test_direct_solver_type_with_cudss_raises() -> None:
    """The direct path (PREONLY+PCLU) must refuse cudss at setup, not hang later."""
    operator = sp.eye(3, dtype=complex, format="csr") * (2.0 + 1.0j)
    with pytest.raises(ValueError, match="not a PETSc factor type"):
        factorize(
            operator,
            solver_cfg={"linsolve": {"type": "mumps", "factor_backend": "cudss"}},
        )


def test_cudss_unavailable_without_nvmath() -> None:
    if _NVMATH_INSTALLED:
        pytest.skip("SKIPPED: nvmath installed; the missing-wheel path is not testable here")
    assert cudss_available() is False
    assert backends.factor_backend_available("cudss") is False


def test_cudss_solver_validates_before_importing_nvmath() -> None:
    """Shape and precision checks run before the optional import, so they work anywhere."""
    rectangular = sp.csr_matrix(np.ones((2, 3), dtype=complex))
    with pytest.raises(ValueError, match="square"):
        CudssSolver(rectangular)
    square = sp.eye(2, dtype=complex, format="csr")
    with pytest.raises(ValueError, match="precision"):
        CudssSolver(square, precision="half")


def test_cudss_solver_missing_nvmath_names_the_extra() -> None:
    if _NVMATH_INSTALLED:
        pytest.skip("SKIPPED: nvmath installed; the missing-wheel path is not testable here")
    square = sp.eye(2, dtype=complex, format="csr")
    with pytest.raises(RuntimeError, match=r"linstabpy\[cudss\]"):
        CudssSolver(square)


def test_equilibrate_reconstructs_and_bounds_rows() -> None:
    kmat = _test_operator(n=60, seed=3)
    scaled, r, c = equilibrate(kmat)
    reference = (sp.diags(r) @ kmat @ sp.diags(c)).tocsr()
    np.testing.assert_allclose(np.abs(scaled - reference).max(), 0.0, atol=1.0e-15)
    row_inf = np.asarray(abs(scaled).max(axis=1).todense()).ravel()
    assert row_inf.max() <= 1.0 + 1.0e-12


def test_probe_family_masked_and_normalized() -> None:
    n = 40
    masked_rows = np.zeros(n, dtype=bool)
    masked_rows[5:15] = True
    B_mat = sp.diags(masked_rows.astype(float)).tocsr()
    family = _make_production_probe_family(B_mat, lambda v: v)
    assert family is not None
    rng = np.random.RandomState(0)
    probe = family(rng)
    assert probe.shape == (n,)
    np.testing.assert_allclose(np.linalg.norm(probe), 1.0, rtol=1.0e-12)
    # Exact-zero check by construction: the diagonal mask multiplies these
    # rows by 0.0, so any nonzero here is a wiring bug, not round-off.
    assert not np.any(probe[~masked_rows])
    assert _make_production_probe_family(None, lambda v: v) is None


def test_adjoint_self_test_draws_from_probe_family() -> None:
    """With a probe family the self-test must use it for every probe pair."""
    n = 10
    calls: list[int] = []

    def family(rng: np.random.RandomState) -> np.ndarray:
        calls.append(1)
        return rng.randn(n) + 1j * rng.randn(n)

    def identity(v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=complex)

    defect = _run_adjoint_self_test(identity, identity, n, k=3, probe_family=family)
    assert defect < 1.0e-14
    assert len(calls) == 6  # two probes per pair, three pairs


# ---------------------------------------------------------------------------
# Tier 2: needs the cuDSS stack (nvmath wheels + NVIDIA GPU)
# ---------------------------------------------------------------------------

needs_cudss = pytest.mark.skipif(
    not (_HAS_PETSC and cudss_available()),
    reason="SKIPPED: missing cuDSS stack (nvmath wheels + NVIDIA GPU + PETSc)",
)


@needs_cudss
def test_cudss_forward_matches_mumps() -> None:
    """Differential oracle: the cuDSS+FGMRES solve must match a MUMPS direct solve."""
    if not backends.factor_backend_available("mumps"):
        pytest.skip("SKIPPED: missing MUMPS factor backend")
    kmat = _test_operator()
    rng = np.random.RandomState(11)
    rhs = rng.randn(kmat.shape[0]) + 1j * rng.randn(kmat.shape[0])

    reference = factorize(kmat, solver_cfg={"linsolve": {"type": "mumps"}})
    subject = factorize(kmat, solver_cfg=_cudss_solver_cfg())
    try:
        assert subject.backend == "cudss-fgmres"
        x_ref = reference.solve(rhs)
        x_gpu = subject.solve(rhs)
        np.testing.assert_allclose(x_gpu, x_ref, rtol=1.0e-9, atol=1.0e-12)
    finally:
        subject.destroy()
        reference.destroy()


@needs_cudss
def test_cudss_adjoint_matches_scipy_dual_factor() -> None:
    """The adjoint must ride a dedicated K^H factor and match a SciPy K^H solve."""
    kmat = _test_operator(n=200, seed=9)
    rng = np.random.RandomState(13)
    rhs = rng.randn(kmat.shape[0]) + 1j * rng.randn(kmat.shape[0])

    subject = factorize(kmat, solver_cfg=_cudss_solver_cfg())
    try:
        x_adj = subject.solve_adjoint(rhs)
        x_ref = spla.spsolve(kmat.conj().T.tocsc(), rhs)
        np.testing.assert_allclose(x_adj, x_ref, rtol=1.0e-8, atol=1.0e-10)
        assert subject.metrics is not None
        assert subject.metrics["dual_factor_active"] is True
    finally:
        subject.destroy()


@needs_cudss
def test_cudss_refactorize_equals_fresh_factor() -> None:
    """New values on the kept pattern must reproduce a fresh factorization."""
    kmat = _test_operator(n=200, seed=5)
    shifted = (kmat - 0.3j * sp.eye(kmat.shape[0], dtype=complex, format="csr")).tocsr()
    rng = np.random.RandomState(17)
    rhs = rng.randn(kmat.shape[0]) + 1j * rng.randn(kmat.shape[0])

    walked = factorize(kmat, solver_cfg=_cudss_solver_cfg())
    fresh = factorize(shifted, solver_cfg=_cudss_solver_cfg())
    try:
        walked.refactorize(shifted)
        x_walked = walked.solve(rhs)
        x_fresh = fresh.solve(rhs)
        np.testing.assert_allclose(x_walked, x_fresh, rtol=1.0e-10, atol=1.0e-13)
        assert walked.metrics is not None
        assert walked.metrics["refactor_count"] == 1
    finally:
        fresh.destroy()
        walked.destroy()


@needs_cudss
def test_cudss_single_precision_needs_probe_family() -> None:
    """FP32 without the production probe family must fail loudly at the adjoint gate."""
    kmat = _test_operator(n=200, seed=21)
    rng = np.random.RandomState(23)
    rhs = rng.randn(kmat.shape[0]) + 1j * rng.randn(kmat.shape[0])

    subject = factorize(kmat, solver_cfg=_cudss_solver_cfg(factor_precision="single"))
    try:
        with pytest.raises(RuntimeError, match="probe_family"):
            subject.solve_adjoint(rhs)
    finally:
        subject.destroy()


@needs_cudss
def test_cudss_single_precision_masked_matches_double() -> None:
    """FP32 on the production (masked) family must match the FP64 solution."""
    kmat = _test_operator(n=200, seed=25)
    n = kmat.shape[0]
    mask = np.zeros(n)
    mask[: n // 2] = 1.0
    B_mat = sp.diags(mask).tocsr()
    family = _make_production_probe_family(B_mat, lambda v: v)
    assert family is not None
    rng = np.random.RandomState(27)
    rhs = mask * (rng.randn(n) + 1j * rng.randn(n))
    rhs /= np.linalg.norm(rhs)

    double = factorize(kmat, solver_cfg=_cudss_solver_cfg())
    single = factorize(kmat, solver_cfg=_cudss_solver_cfg(factor_precision="single"))
    single.probe_family = family
    try:
        x_double = double.solve(rhs)
        x_single = single.solve(rhs)
        np.testing.assert_allclose(x_single, x_double, rtol=1.0e-8, atol=1.0e-10)
        x_adj_single = single.solve_adjoint(rhs)
        x_adj_ref = spla.spsolve(kmat.conj().T.tocsc(), rhs)
        np.testing.assert_allclose(x_adj_single, x_adj_ref, rtol=1.0e-7, atol=1.0e-9)
    finally:
        single.destroy()
        double.destroy()
