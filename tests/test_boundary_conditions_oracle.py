#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Independent structural oracle for Dirichlet BC application.

``apply_dirichlet`` enforces ``q = 0`` at boundary DOFs by zeroing the
corresponding rows AND columns of the operator and placing ``penalty`` on the
diagonal. The DOF for variable ``v`` at grid point ``g`` is ``var*ngp + g`` with
the documented variable order ``[rho, u, v, w, T] = 0..4``.

This oracle re-encodes that contract from the documentation (not from the
production maps) and reconstructs the expected post-BC matrix entry by entry,
then asserts ``apply_dirichlet`` reproduces it exactly. A wrong variable index,
a transposed DOF layout (``g*5 + var``), or row-only (no column) zeroing diverges
from the reconstruction and fails here -- a check independent of the production
index maps and CSR row/col plumbing.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from linstabpy.boundary_conditions.dirichlet import apply_dirichlet

# Documented contract, re-stated independently of the production maps.
_VAR_ORDER = {"r": 0, "u": 1, "v": 2, "w": 3, "T": 4}
_BORDER_KEY = {"l": "li", "r": "ri", "b": "bi", "t": "ti", "m": "mi"}


def _expected_dofs(idx: dict, borders: str, variables: str, ngp: int) -> np.ndarray:
    dofs: list[int] = []
    for b in borders:
        grid_pts = idx[_BORDER_KEY[b]]
        for v in variables:
            dofs.extend((_VAR_ORDER[v] * ngp + grid_pts).tolist())
    return np.unique(np.asarray(dofs, dtype=int))


def _expected_matrix(L0: np.ndarray, bc: np.ndarray, penalty: float) -> np.ndarray:
    n = L0.shape[0]
    out = L0.copy()
    bc_mask = np.zeros(n, dtype=bool)
    bc_mask[bc] = True
    out[bc_mask, :] = 0.0  # zero BC rows
    out[:, bc_mask] = 0.0  # zero BC columns
    out[bc, bc] = penalty  # diagonal penalty on constrained DOFs
    return out


def _run_dirichlet_oracle(
    ngp: int = 9,
    n_border: int = 3,
    borders: str = "lrbt",
    variables: str = "ruvwT",
    penalty: float = 1e4,
    seed: int = 20260601,
) -> dict:
    rng = np.random.default_rng(seed)
    # distinct boundary grid points per border (overlap across borders allowed)
    idx = {
        key: np.sort(rng.choice(ngp, size=n_border, replace=False)) for key in _BORDER_KEY.values()
    }

    dense = rng.normal(size=(5 * ngp, 5 * ngp)) + 1j * rng.normal(size=(5 * ngp, 5 * ngp))
    L0 = sps.csc_matrix(dense)

    bc_expected = _expected_dofs(idx, borders, variables, ngp)
    M_expected = _expected_matrix(dense, bc_expected, penalty)

    L0_bc, bc_indices = apply_dirichlet(L0.copy(), idx, borders, variables, ngp, penalty=penalty)
    M_got = np.asarray(L0_bc.todense())

    dof_match = bool(
        bc_indices.shape == bc_expected.shape and np.array_equal(bc_indices, bc_expected)
    )
    max_abs = float(np.max(np.abs(M_got - M_expected)))
    return {
        "ngp": ngp,
        "dof_indices_match": dof_match,
        "max_abs_matrix_diff": max_abs,
        "pass": bool(dof_match and max_abs == 0.0),
    }


def test_dirichlet_matches_independent_structural_oracle():
    # Independent structural oracle: post-BC matrix reconstructed from the
    # documented DOF contract, not from the production index maps.
    report = _run_dirichlet_oracle()
    assert report["pass"], report


def test_dirichlet_oracle_variable_and_border_subsets():
    for borders, variables, seed in (
        ("b", "uvwT", 1),  # classic no-slip + isothermal wall on the bottom
        ("lr", "r", 2),
        ("t", "ruvwT", 3),
        ("lrbt", "uvw", 4),
    ):
        report = _run_dirichlet_oracle(
            ngp=11, n_border=4, borders=borders, variables=variables, seed=seed
        )
        assert report["pass"], (borders, variables, report)
