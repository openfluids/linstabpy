# SPDX-License-Identifier: GPL-3.0-only
"""MPI regression guard for collective refactorize pattern rejection.

The nonzero-pattern comparison in ``Factorization.refactorize`` is local to
PETSc-owned rows.
Under MPI, a structural change present on only one rank must still make
every rank take the same raise/fallback branch.
That must happen before the collective ``KSP.setOperators()/setUp()`` path.
The serial suite stays useful (size 1 reduces to the same logic).
The check is also meaningful under ``mpiexec``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from petsc4py import PETSc

from linstabpy.linalg.factorize import factorize
from linstabpy.solvers.petsc_utils import scipy_to_petsc

_PATTERN_ERROR = "refactorize requires an identical nonzero pattern"


def _solver_cfg() -> dict[str, object]:
    return {
        "petsc_matrix_mode": "distributed",
        "linsolve": {"type": "mumps", "factor_backend": "mumps"},
    }


def _operator(
    n: int, *, diagonal_shift: complex = 0.0j, rank0_extra: bool = False
) -> sp.csc_matrix:
    diagonal = 4.0 + 0.1j + diagonal_shift + 0.01 * np.arange(n)
    lower = -0.25 + 0.02j + 0.001 * np.arange(n - 1)
    upper = -0.15 - 0.03j - 0.001j * np.arange(n - 1)
    mat = sp.diags([lower, diagonal, upper], [-1, 0, 1], format="lil", dtype=complex)
    if rank0_extra:
        # PETSc's default row partition gives rank 0 row 0.  Column
        # n-1 is outside the tridiagonal pattern for n >= 8, so this is a
        # structural mismatch localized to rank 0's local CSR rows.
        mat[0, n - 1] = 0.125 - 0.25j
    return mat.tocsc()


def _local_pattern(mat: PETSc.Mat) -> tuple[np.ndarray, np.ndarray]:
    indptr, indices, _ = mat.getValuesCSR()
    return np.asarray(indptr, dtype=np.int64).copy(), np.asarray(indices, dtype=np.int64).copy()


def _patterns_equal(left: PETSc.Mat, right: PETSc.Mat) -> bool:
    left_indptr, left_indices = _local_pattern(left)
    right_indptr, right_indices = _local_pattern(right)
    return bool(
        np.array_equal(left_indptr, right_indptr) and np.array_equal(left_indices, right_indices)
    )


def test_refactorize_pattern_guard_raises_collectively_for_rank_local_mismatch() -> None:
    comm = PETSc.COMM_WORLD
    mpi_comm = comm.tompi4py()
    size = comm.getSize()
    n = max(8, 4 * size)
    cfg = _solver_cfg()

    base = scipy_to_petsc(_operator(n), cfg)
    same_pattern = scipy_to_petsc(_operator(n, diagonal_shift=0.35 + 0.07j), cfg)
    different_pattern = scipy_to_petsc(
        _operator(n, diagonal_shift=0.55 - 0.03j, rank0_extra=True), cfg
    )
    fac = None
    try:
        if size > 1:
            assert "mpi" in base.getType().lower()

        fac = factorize(base, solver_cfg=cfg)
        assert fac.backend == "MUMPS"

        # Same nonzero pattern, different values: all ranks must pass through
        # refactorize without raising.
        assert fac._mat is not None
        same_pattern_ok = _patterns_equal(fac._mat, same_pattern)
        same_pattern_failures = mpi_comm.allreduce(int(not same_pattern_ok), op=MPI.SUM)
        assert same_pattern_failures == 0
        fac.refactorize(same_pattern)

        assert fac._mat is not None
        r0, r1 = fac._mat.getOwnershipRange()
        owns_rank0_row = int(r0 <= 0 < r1)
        owner_count = mpi_comm.allreduce(owns_rank0_row, op=MPI.SUM)
        assert owner_count == 1

        local_mismatch = not _patterns_equal(fac._mat, different_pattern)
        mismatch_count = mpi_comm.allreduce(int(local_mismatch), op=MPI.SUM)
        wrong_localization = mpi_comm.allreduce(
            int(local_mismatch != bool(owns_rank0_row)), op=MPI.SUM
        )
        assert mismatch_count == 1
        assert wrong_localization == 0

        did_raise = False
        wrong_error = False
        try:
            fac.refactorize(different_pattern)
        except ValueError as exc:
            did_raise = _PATTERN_ERROR in str(exc)
            wrong_error = not did_raise

        # This collective assertion is the regression guard: with the LOR in
        # refactorize(), every rank reports the same ValueError even though only
        # rank 0 has a local CSR-pattern mismatch.  Without the collective guard,
        # non-owning ranks would either report did_raise=False or hang in the
        # collective PETSc setup path.
        raise_count = mpi_comm.allreduce(int(did_raise), op=MPI.SUM)
        wrong_error_count = mpi_comm.allreduce(int(wrong_error), op=MPI.SUM)
        assert wrong_error_count == 0
        assert raise_count == size
    finally:
        if fac is not None:
            fac.destroy()
        different_pattern.destroy()
        same_pattern.destroy()
        base.destroy()
