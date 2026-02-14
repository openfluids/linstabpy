"""Dirichlet boundary condition enforcement for the LNSE operator.

Translates ``BC_Dirichlet.m`` from the MATLAB LinStab2D codebase.  Works
with scipy.sparse matrices (the operator is assembled in scipy before
optional conversion to PETSc).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


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
    # Strategy:
    #   - Convert to lil_matrix for efficient row zeroing.
    #   - Convert to csc for efficient column zeroing.
    #   - Set diagonal entries via lil_matrix.
    L0 = L0.tolil()
    L0[bc_indices] = 0  # zero rows (fancy indexing on lil)
    L0 = L0.tocsc()
    L0[:, bc_indices] = 0  # zero columns (efficient for csc)

    # Set diagonal entries to penalty
    L0 = L0.tolil()
    for i in bc_indices:
        L0[i, i] = penalty
    L0 = L0.tocsc()

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

    For each boundary DOF, rows/columns are zeroed and diagonal entries set to
    ``penalty`` in both ``Lc`` and ``R`` at both copies of the doubled system.
    """
    base_indices = _collect_dirichlet_indices(idx, borders, variables, ngp)
    if base_indices.size == 0:
        return Lc, R, np.array([], dtype=int)

    n = ngp * 5
    bc_indices = np.sort(np.unique(np.concatenate([base_indices, base_indices + n])))

    def _apply_bc(mat: sp.csc_matrix) -> sp.csc_matrix:
        mat = mat.tolil()
        mat[bc_indices] = 0
        mat = mat.tocsc()
        mat[:, bc_indices] = 0
        mat = mat.tolil()
        for i in bc_indices:
            mat[i, i] = penalty
        return mat.tocsc()

    return _apply_bc(Lc), _apply_bc(R), bc_indices
