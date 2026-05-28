# SPDX-License-Identifier: GPL-3.0-only
"""Dirichlet boundary condition enforcement for the LNSE operator.

Translates ``BC_Dirichlet.m`` from the MATLAB LinStab2D codebase.  Works
with scipy.sparse matrices (the operator is assembled in scipy before
optional conversion to PETSc).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover - typing-only optional dependency
    from petsc4py import PETSc


def _zero_rows_csr(mat: sp.csr_matrix, rows: np.ndarray) -> None:
    """Zero rows in a CSR matrix by clearing stored data values."""
    starts = mat.indptr[rows]
    ends = mat.indptr[rows + 1]
    idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    if idx.size:
        mat.data[idx] = 0


def _zero_cols_csc(mat: sp.csc_matrix, cols: np.ndarray) -> None:
    """Zero columns in a CSC matrix by clearing stored data values."""
    starts = mat.indptr[cols]
    ends = mat.indptr[cols + 1]
    idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    if idx.size:
        mat.data[idx] = 0


def _set_diagonal_entries_csc(
    mat: sp.csc_matrix,
    indices: np.ndarray,
    value: float | complex,
) -> sp.csc_matrix:
    """Set selected diagonal entries, reusing existing CSC slots when present."""
    indices = np.unique(np.asarray(indices, dtype=int))
    if indices.size == 0:
        return mat

    missing: list[int] = []
    for col in indices:
        start = mat.indptr[col]
        stop = mat.indptr[col + 1]
        rows = mat.indices[start:stop]
        hit = np.flatnonzero(rows == col)
        if hit.size:
            mat.data[start + int(hit[0])] = value
        else:
            missing.append(int(col))

    if missing:
        missing_idx = np.asarray(missing, dtype=int)
        mat = mat + sp.csc_matrix(
            (np.full(missing_idx.size, value), (missing_idx, missing_idx)),
            shape=mat.shape,
        )
    return mat


# Border character -> idx dict key
_BORDER_MAP: dict[str, str] = {"l": "li", "r": "ri", "b": "bi", "t": "ti", "m": "mi"}

# Variable character -> block index (0-based)
_VARIABLE_MAP: dict[str, int] = {"r": 0, "u": 1, "v": 2, "w": 3, "T": 4}


def apply_dirichlet(
    L0: sp.csc_matrix,
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    ngp: int,
    penalty: float = 1e4,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Apply Dirichlet boundary conditions to the LNSE operator.

    For every combination of *border* and *variable*, the corresponding
    rows and columns in *L0* are zeroed and the diagonal entries are set
    to *penalty*, enforcing :math:`q = 0` at those DOFs.

    Parameters
    ----------
    L0 : sparse matrix, shape (5*ngp, 5*ngp)
        The LNSE operator.
    idx : dict
        Boundary index dict with keys ``'bi'``, ``'ti'``, ``'li'``, ``'ri'``,
        and ``'mi'`` (mask boundary), containing 0-based indices into the
        grid point array (length ``ngp``).
    borders : str
        Which borders to apply BCs on.  Characters from ``'lrbtm'``:
        ``l`` = left, ``r`` = right, ``b`` = bottom, ``t`` = top,
        ``m`` = mask/inner boundary.
    variables : str
        Which variables to apply BCs for.  Characters from ``'ruvwT'``:
        ``r`` = rho, ``u`` = u, ``v`` = v, ``w`` = w, ``T`` = T.
    ngp : int
        Number of grid points.
    penalty : float, optional
        Penalty value placed on the diagonal (default ``1e4``).

    Returns
    -------
    L0 : csc_matrix
        Modified operator with BC rows/columns zeroed and diagonal set
        to *penalty*.
    bc_indices : ndarray of int
        Sorted 1-D array of DOF indices where BCs were applied.

    Raises
    ------
    ValueError
        If an unrecognised border or variable character is encountered.
    """
    # ------------------------------------------------------------------
    # 1. Collect DOF indices for all (border, variable) pairs
    # ------------------------------------------------------------------
    dof_list: list[np.ndarray] = []

    for b in borders:
        border_key = _BORDER_MAP.get(b)
        if border_key is None:
            raise ValueError(
                f"Unknown border character '{b}'; expected one of 'l', 'r', 'b', 't', 'm'"
            )
        grid_pts = idx[border_key]

        for v in variables:
            var_idx = _VARIABLE_MAP.get(v)
            if var_idx is None:
                raise ValueError(
                    f"Unknown variable character '{v}'; expected one of 'r', 'u', 'v', 'w', 'T'"
                )
            dof_list.append(var_idx * ngp + grid_pts)

    if len(dof_list) == 0:
        return L0.tocsc(), np.array([], dtype=int)

    bc_indices = np.unique(np.concatenate(dof_list))

    # ------------------------------------------------------------------
    # 2. Zero rows and columns, then set diagonal to penalty
    # ------------------------------------------------------------------
    # Direct data-array manipulation avoids SparseEfficiencyWarning
    # that fancy indexing on CSC/CSR would trigger.
    L0 = L0.tocsr()
    _zero_rows_csr(L0, bc_indices)
    L0 = L0.tocsc()
    _zero_cols_csc(L0, bc_indices)

    L0 = _set_diagonal_entries_csc(L0, bc_indices, penalty)
    L0.eliminate_zeros()

    return L0, bc_indices


def apply_dirichlet_petsc(
    L0: "PETSc.Mat",
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    ngp: int,
    penalty: float = 1e4,
) -> tuple["PETSc.Mat", np.ndarray]:
    """Apply Dirichlet boundary conditions directly to a PETSc matrix.

    This mirrors :func:`apply_dirichlet` for solver-facing PETSc matrices by
    zeroing the selected rows and columns and placing ``penalty`` on the
    diagonal entries of the constrained degrees of freedom.
    """
    from petsc4py import PETSc

    bc_indices = _collect_dirichlet_indices(idx, borders, variables, ngp)
    if bc_indices.size == 0:
        return L0, bc_indices

    if L0.getSize()[0] != L0.getSize()[1]:
        raise ValueError("L0 must be square")
    if bc_indices.min() < 0 or bc_indices.max() >= L0.getSize()[0]:
        raise ValueError("Dirichlet indices are out of bounds for the PETSc matrix")

    L0.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    L0.zeroRowsColumns(np.asarray(bc_indices, dtype=np.int32), diag=complex(penalty))
    L0.assemble()
    return L0, bc_indices


def _collect_dirichlet_indices(
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    ngp: int,
) -> np.ndarray:
    """Collect 1-D boundary DOF indices for a set of border/variable keys."""
    dof_list: list[np.ndarray] = []

    for b in borders:
        border_key = _BORDER_MAP.get(b)
        if border_key is None:
            raise ValueError(
                f"Unknown border character '{b}'; expected one of 'l', 'r', 'b', 't', 'm'"
            )
        grid_pts = idx[border_key]

        for v in variables:
            var_idx = _VARIABLE_MAP.get(v)
            if var_idx is None:
                raise ValueError(
                    f"Unknown variable character '{v}'; expected one of 'r', 'u', 'v', 'w', 'T'"
                )
            dof_list.append(var_idx * ngp + grid_pts)

    if len(dof_list) == 0:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(dof_list))


def apply_dirichlet_spatial(
    Lc: sp.csc_matrix,
    R: sp.csc_matrix,
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    ngp: int,
    penalty: float = 1e4,
) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """Apply Dirichlet boundary conditions to doubled spatial system matrices.

    For each boundary DOF, rows/columns in *Lc* are zeroed and diagonal set
    to *penalty*.  For *R*, rows/columns are zeroed but **no** penalty is
    placed on the diagonal — adding penalty to both matrices creates spurious
    eigenvalues at alpha = 1.  This matches MATLAB ``BC_Dirichlet_Spatial.m``.
    """
    base_indices = _collect_dirichlet_indices(idx, borders, variables, ngp)
    if base_indices.size == 0:
        return Lc, R, np.array([], dtype=int)

    n = ngp * 5
    bc_indices = np.sort(np.unique(np.concatenate([base_indices, base_indices + n])))

    def _zero_rows_cols(mat: sp.csc_matrix) -> sp.csc_matrix:
        mat = mat.tocsr()
        _zero_rows_csr(mat, bc_indices)
        mat = mat.tocsc()
        _zero_cols_csc(mat, bc_indices)
        mat.eliminate_zeros()
        return mat

    # Lc: zero rows/cols + penalty on diagonal
    Lc = _zero_rows_cols(Lc)
    Lc = _set_diagonal_entries_csc(Lc, bc_indices, penalty)
    Lc.eliminate_zeros()

    # R: zero rows/cols only (no penalty)
    R = _zero_rows_cols(R)

    return Lc, R, bc_indices


def apply_dirichlet_spatial_petsc(
    Lc: "PETSc.Mat",
    R: "PETSc.Mat",
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    ngp: int,
    penalty: float = 1e4,
) -> tuple["PETSc.Mat", "PETSc.Mat", np.ndarray]:
    """Apply Dirichlet BCs directly to doubled spatial PETSc matrices."""
    from petsc4py import PETSc

    base_indices = _collect_dirichlet_indices(idx, borders, variables, ngp)
    if base_indices.size == 0:
        return Lc, R, np.array([], dtype=int)

    if Lc.getSize()[0] != Lc.getSize()[1] or R.getSize()[0] != R.getSize()[1]:
        raise ValueError("Lc and R must be square")
    if Lc.getSize() != R.getSize():
        raise ValueError("Lc and R must have identical shapes")

    n = ngp * 5
    bc_indices = np.sort(np.unique(np.concatenate([base_indices, base_indices + n])))
    if bc_indices.min() < 0 or bc_indices.max() >= Lc.getSize()[0]:
        raise ValueError("Dirichlet indices are out of bounds for the spatial PETSc matrices")

    bc_idx = np.asarray(bc_indices, dtype=np.int32)
    Lc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    R.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    Lc.zeroRowsColumns(bc_idx, diag=complex(penalty))
    R.zeroRowsColumns(bc_idx, diag=0.0)
    Lc.assemble()
    R.assemble()
    return Lc, R, bc_indices


def build_dirichlet_restriction_matrices(
    n_dofs: int,
    dirichlet_indices: np.ndarray,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Build reduced-system restriction/expansion maps (MATLAB ``SetDirBC``).

    Returns matrices:
    - ``T_cont``: removes Dirichlet rows from a full vector/matrix
    - ``T_exp``: injects reduced vectors back into full-space free dofs
    plus the sorted free-dof index array.
    """
    free_dofs = dirichlet_free_dofs(n_dofs, dirichlet_indices)

    # T_cont = I with Dirichlet rows removed (shape n_free x n_full)
    t_cont = sp.csr_matrix(
        (
            np.ones(free_dofs.size, dtype=float),
            (np.arange(free_dofs.size, dtype=int), free_dofs),
        ),
        shape=(free_dofs.size, n_dofs),
    )
    # T_exp = I with Dirichlet columns removed (shape n_full x n_free)
    t_exp = sp.csr_matrix(
        (
            np.ones(free_dofs.size, dtype=float),
            (free_dofs, np.arange(free_dofs.size, dtype=int)),
        ),
        shape=(n_dofs, free_dofs.size),
    )
    return t_cont, t_exp, free_dofs


def dirichlet_free_dofs(n_dofs: int, dirichlet_indices: np.ndarray) -> np.ndarray:
    """Return sorted unconstrained DOF indices after Dirichlet elimination."""
    if n_dofs < 0:
        raise ValueError("n_dofs must be non-negative")

    index_set = np.unique(np.asarray(dirichlet_indices, dtype=int))
    if index_set.size:
        if index_set.min() < 0 or index_set.max() >= n_dofs:
            raise ValueError("dirichlet_indices must be valid 0-based dof indices")

    all_dofs = np.arange(n_dofs, dtype=int)
    return np.setdiff1d(all_dofs, index_set, assume_unique=True)


def apply_dirichlet_identity_pair(
    lhs: sp.csc_matrix,
    rhs: sp.csc_matrix,
    index_set: np.ndarray,
) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """Apply identity-row Dirichlet BC to two matrices (MATLAB ``SetDirichletBC``).

    For each index in ``index_set``:
    - zero corresponding rows in ``lhs`` and ``rhs``
    - set diagonal entry to 1 in both matrices
    """
    bc_indices = np.unique(np.asarray(index_set, dtype=int))
    if bc_indices.size == 0:
        return lhs.tocsc(), rhs.tocsc()

    n = lhs.shape[0]
    if lhs.shape[0] != lhs.shape[1] or rhs.shape[0] != rhs.shape[1]:
        raise ValueError("lhs and rhs must be square matrices")
    if rhs.shape != lhs.shape:
        raise ValueError("lhs and rhs must have the same shape")
    if bc_indices.min() < 0 or bc_indices.max() >= n:
        raise ValueError("index_set contains out-of-bounds dof indices")

    lhs = lhs.tocsr()
    rhs = rhs.tocsr()
    _zero_rows_csr(lhs, bc_indices)
    _zero_rows_csr(rhs, bc_indices)

    ident = sp.csc_matrix(
        (np.ones(bc_indices.size, dtype=float), (bc_indices, bc_indices)),
        shape=lhs.shape,
    )
    lhs = lhs.tocsc() + ident
    rhs = rhs.tocsc() + ident

    lhs.eliminate_zeros()
    rhs.eliminate_zeros()
    return lhs, rhs
