# SPDX-License-Identifier: GPL-3.0-only
"""Nonzero-pattern guard of the in-place refactorize path.

A serial cuDSS factorization always refactorizes by a short route.
That route extracts the CSR triple once, compares the index arrays, and
then writes the values into the same Mat.
It keeps the symbolic factorization of the first operator.
A second operator with another nonzero pattern must therefore raise.
The guard uses NumPy only, so these tests need no PETSc and no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.linalg.factorize import hermitian_transpose_plan, require_same_pattern

_PATTERN_ERROR = "refactorize requires an identical nonzero pattern"


def _operator(n: int, *, diagonal_shift: complex = 0.0j, extra: bool = False) -> sp.csr_matrix:
    diagonal = 4.0 + 0.1j + diagonal_shift + 0.01 * np.arange(n)
    lower = -0.25 + 0.02j + 0.001 * np.arange(n - 1)
    upper = -0.15 - 0.03j - 0.001j * np.arange(n - 1)
    mat = sp.diags([lower, diagonal, upper], [-1, 0, 1], format="lil", dtype=complex)
    if extra:
        # Column n-1 is outside the tridiagonal pattern for n >= 8. This entry
        # is therefore a structural change and not only a value change.
        mat[0, n - 1] = 0.125 - 0.25j
    csr = mat.tocsr()
    csr.sort_indices()
    return csr


def _pattern(mat: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(mat.indptr, dtype=np.int64), np.asarray(mat.indices, dtype=np.int64)


def test_same_pattern_with_new_values_passes_the_guard() -> None:
    base = _operator(12)
    same_pattern = _operator(12, diagonal_shift=0.35 + 0.07j)

    require_same_pattern(_pattern(base), same_pattern.indptr, same_pattern.indices)


def test_extra_nonzero_raises_the_pattern_error() -> None:
    base = _operator(12)
    different_pattern = _operator(12, diagonal_shift=0.55 - 0.03j, extra=True)

    with pytest.raises(ValueError, match=_PATTERN_ERROR):
        require_same_pattern(_pattern(base), different_pattern.indptr, different_pattern.indices)


def test_a_shorter_row_pointer_raises_the_pattern_error() -> None:
    base = _operator(12)
    dropped = base.copy()
    dropped[0, 1] = 0.0
    dropped.eliminate_zeros()

    with pytest.raises(ValueError, match=_PATTERN_ERROR):
        require_same_pattern(_pattern(base), dropped.indptr, dropped.indices)


def test_transpose_plan_matches_the_sparse_conjugate_transpose() -> None:
    base = _operator(12, extra=True)
    perm, indptr_h, indices_h = hermitian_transpose_plan(base.indptr, base.indices, 12)
    shifted = _operator(12, diagonal_shift=0.35 + 0.07j, extra=True)

    planned = sp.csr_matrix((np.conj(shifted.data[perm]), indices_h, indptr_h), shape=(12, 12))
    expected = shifted.conj().T.tocsr()
    expected.sort_indices()

    np.testing.assert_array_equal(planned.indptr, expected.indptr)
    np.testing.assert_array_equal(planned.indices, expected.indices)
    np.testing.assert_array_equal(planned.data, expected.data)
