"""Neumann boundary-condition enforcement for the LNSE operator.

Translates ``BC_Neumann.m`` from the MATLAB LinStab2D codebase.
This applies derivative-normal operator rows to boundary DOFs in
curvilinear coordinates while preserving sparse structure.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .dirichlet import _BORDER_MAP, _VARIABLE_MAP, _zero_rows_csr
from ..mesh.mesh import Mesh


_NEUMANN_BORDER_SIGNS: dict[str, float] = {
    "l": -1.0,
    "r": 1.0,
    "b": 1.0,
    "t": -1.0,
}


def _collect_neumann_indices(
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    ngp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect boundary dof indices, mesh indices, and normals for Neumann BC.

    Returns:
    - index_set: unique BC dofs
    - index_mesh: unique mesh point ids for those dofs
    - border_tags: border character associated with each unique BC dof
    """
    raw_index_set: list[np.ndarray] = []
    raw_index_mesh: list[np.ndarray] = []
    raw_borders: list[np.ndarray] = []

    for b in borders:
        if b not in _NEUMANN_BORDER_SIGNS:
            raise ValueError(
                "Neumann B.C. are currently only supported for b, t, l and r."
            )

        border_key = _BORDER_MAP.get(b)
        if border_key is None:
            raise ValueError(
                f"Unknown border character '{b}'; expected one of 'l', 'r', 'b', 't'"
            )
        boundary_points = idx[border_key]
        boundary_count = boundary_points.size
        if boundary_count == 0:
            continue

        for v in variables:
            v = "r" if v == "r" else v
            variable_offset = _VARIABLE_MAP.get(v)
            if variable_offset is None:
                raise ValueError(
                    f"Unknown variable character '{v}'; expected one of 'r', 'u', 'v', 'w', 'T'"
                )

            dof_idx = variable_offset * ngp + boundary_points
            raw_index_set.append(dof_idx)
            raw_index_mesh.append(boundary_points)
            raw_borders.append(np.full(boundary_count, b, dtype="<U1"))

    if not raw_index_set:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype="<U1"),
        )

    raw_index_set_array = np.concatenate(raw_index_set)
    raw_index_mesh_array = np.concatenate(raw_index_mesh)
    raw_borders_array = np.concatenate(raw_borders)

    index_set, unique_pos = np.unique(raw_index_set_array, return_index=True)
    index_mesh = raw_index_mesh_array[unique_pos]
    border_tags = raw_borders_array[unique_pos]
    return index_set, index_mesh, border_tags


def apply_neumann(
    L0: sp.csc_matrix,
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    penalty: float = 1e4,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Apply Neumann boundary conditions to the LNSE operator.

    For each selected border, the corresponding row is replaced by a
    curvilinear normal-derivative row (scaled by ``penalty``) applied to the
    same variable block.
    """
    ngp = mesh.ngp
    index_set, index_mesh, border_tags = _collect_neumann_indices(
        idx, borders, variables, ngp
    )

    if index_set.size == 0:
        return L0.tocsc(), np.array([], dtype=int)

    x_flat = mesh.X.ravel(order="F")[mesh.used_ind]
    y_flat = mesh.Y.ravel(order="F")[mesh.used_ind]

    gradX_x = mesh.Dx @ x_flat
    gradX_y = mesh.Dy @ x_flat
    gradY_x = mesh.Dx @ y_flat
    gradY_y = mesh.Dy @ y_flat

    L0 = L0.tocsr()
    _zero_rows_csr(L0, index_set)
    L0 = L0.tocsc()

    row_ids: list[np.ndarray] = []
    col_ids: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for bc_dof, imesh, border in zip(index_set, index_mesh, border_tags):
        variable_offset = bc_dof // ngp

        sign = _NEUMANN_BORDER_SIGNS[str(border)]
        if border in ("l", "r"):
            row_scale_x = sign * gradY_y[imesh]
            row_scale_y = sign * (-gradY_x[imesh])
        else:
            row_scale_x = sign * (-gradX_y[imesh])
            row_scale_y = sign * gradX_x[imesh]

        row = mesh.Dx.getrow(imesh) * row_scale_x
        row = row + mesh.Dy.getrow(imesh) * row_scale_y
        row = row.tocsr() * penalty
        if row.nnz == 0:
            continue
        row = row.tocoo()
        row_ids.append(np.full(row.nnz, int(bc_dof), dtype=int))
        col_ids.append(row.col.astype(int) + variable_offset * ngp)
        vals.append(row.data)

    if vals:
        bc_rows = np.concatenate(row_ids)
        bc_cols = np.concatenate(col_ids)
        bc_vals = np.concatenate(vals)
        L0 = L0 + sp.csc_matrix((bc_vals, (bc_rows, bc_cols)), shape=L0.shape)

    L0.eliminate_zeros()
    return L0, index_set
