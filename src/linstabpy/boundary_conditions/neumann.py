# SPDX-License-Identifier: GPL-3.0-only
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
    *,
    exclude_corners: bool = False,
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
        boundary_points = np.asarray(idx[border_key], dtype=int)
        if exclude_corners and boundary_points.size:
            if b in ("l", "r"):
                corner_points = np.concatenate(
                    (np.asarray(idx["bi"], dtype=int), np.asarray(idx["ti"], dtype=int))
                )
            else:
                corner_points = np.concatenate(
                    (np.asarray(idx["li"], dtype=int), np.asarray(idx["ri"], dtype=int))
                )
            boundary_points = np.setdiff1d(
                boundary_points,
                np.unique(corner_points),
                assume_unique=False,
            )
        boundary_count = boundary_points.size
        if boundary_count == 0:
            continue

        for v in variables:
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

    order = np.argsort(raw_index_set_array, kind="mergesort")
    sorted_index_set = raw_index_set_array[order]
    sorted_borders = raw_borders_array[order]
    group_starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_index_set)) + 1))
    group_stops = np.concatenate((group_starts[1:], [sorted_index_set.size]))
    ambiguous_borders: dict[int, tuple[str, ...]] = {}
    for start, stop in zip(group_starts, group_stops):
        border_group = np.unique(sorted_borders[start:stop])
        if border_group.size > 1:
            ambiguous_borders[int(sorted_index_set[start])] = tuple(str(b) for b in border_group)
    if ambiguous_borders:
        preview = ", ".join(
            f"dof {dof}: borders={''.join(borders)}"
            for dof, borders in list(sorted(ambiguous_borders.items()))[:5]
        )
        raise ValueError(
            "Ambiguous Neumann BC at shared corner DOFs; a single rule cannot span multiple "
            f"normal directions for the same variable ({preview}). Split the boundary recipe "
            "or exclude corners from overlapping Neumann borders."
        )

    index_set, unique_pos = np.unique(raw_index_set_array, return_index=True)
    index_mesh = raw_index_mesh_array[unique_pos]
    border_tags = raw_borders_array[unique_pos]
    return index_set, index_mesh, border_tags


def _build_neumann_rows(
    mesh: Mesh,
    index_set: np.ndarray,
    index_mesh: np.ndarray,
    border_tags: np.ndarray,
    ngp: int,
    penalty: float,
    curvilinear_normals: bool = False,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Build replacement sparse rows for Neumann boundary conditions."""
    if index_set.size == 0:
        return []

    row_entries: list[tuple[int, np.ndarray, np.ndarray]] = []

    if curvilinear_normals and mesh.J11 is not None:
        for bc_dof, imesh, border in zip(index_set, index_mesh, border_tags):
            variable_offset = int(bc_dof // ngp)
            imesh = int(imesh)
            border = str(border)
            sign = _NEUMANN_BORDER_SIGNS[border]
            if border in ("l", "r"):
                row_scale_x = sign * mesh.J22[imesh]
                row_scale_y = sign * (-mesh.J21[imesh])
            else:
                row_scale_x = sign * (-mesh.J12[imesh])
                row_scale_y = sign * mesh.J11[imesh]
            row = mesh.Dx.getrow(imesh) * row_scale_x
            row = row + mesh.Dy.getrow(imesh) * row_scale_y
            row = row.tocsr() * penalty
            if row.nnz == 0:
                continue
            row = row.tocoo()
            cols = row.col.astype(int) + variable_offset * ngp
            vals = np.asarray(row.data, dtype=complex)
            row_entries.append((int(bc_dof), cols, vals))
    else:
        x_flat = mesh.X.ravel(order="F")[mesh.used_ind]
        y_flat = mesh.Y.ravel(order="F")[mesh.used_ind]
        gradX_x = mesh.Dx @ x_flat
        gradX_y = mesh.Dy @ x_flat
        gradY_x = mesh.Dx @ y_flat
        gradY_y = mesh.Dy @ y_flat
        for bc_dof, imesh, border in zip(index_set, index_mesh, border_tags):
            variable_offset = int(bc_dof // ngp)
            imesh = int(imesh)
            border = str(border)
            sign = _NEUMANN_BORDER_SIGNS[border]
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
            cols = row.col.astype(int) + variable_offset * ngp
            vals = np.asarray(row.data, dtype=complex)
            row_entries.append((int(bc_dof), cols, vals))

    return row_entries


def apply_neumann(
    L0: sp.csc_matrix,
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    penalty: float = 1e4,
    *,
    exclude_corners: bool = False,
    curvilinear_normals: bool = False,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Apply Neumann boundary conditions to the LNSE operator.

    For each selected border, the corresponding row is replaced by a
    curvilinear normal-derivative row (scaled by ``penalty``) applied to the
    same variable block.
    """
    ngp = mesh.ngp
    index_set, index_mesh, border_tags = _collect_neumann_indices(
        idx, borders, variables, ngp, exclude_corners=exclude_corners
    )

    if index_set.size == 0:
        return L0.tocsc(), np.array([], dtype=int)

    L0 = L0.tocsr()
    _zero_rows_csr(L0, index_set)
    L0 = L0.tocsc()

    row_entries = _build_neumann_rows(
        mesh,
        index_set,
        index_mesh,
        border_tags,
        ngp,
        penalty,
        curvilinear_normals=curvilinear_normals,
    )

    if row_entries:
        bc_rows = np.concatenate(
            [np.full(cols.size, row_id, dtype=int) for row_id, cols, _ in row_entries]
        )
        bc_cols = np.concatenate([cols for _, cols, _ in row_entries])
        bc_vals = np.concatenate([vals for _, _, vals in row_entries])
        L0 = L0 + sp.csc_matrix((bc_vals, (bc_rows, bc_cols)), shape=L0.shape)

    L0.eliminate_zeros()
    return L0, index_set


def apply_neumann_petsc(
    L0: "PETSc.Mat",
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    borders: str,
    variables: str,
    penalty: float = 1e4,
    *,
    exclude_corners: bool = False,
    curvilinear_normals: bool = False,
) -> tuple["PETSc.Mat", np.ndarray]:
    """Apply Neumann boundary conditions directly to a PETSc matrix."""
    from petsc4py import PETSc

    ngp = mesh.ngp
    index_set, index_mesh, border_tags = _collect_neumann_indices(
        idx, borders, variables, ngp, exclude_corners=exclude_corners
    )

    if index_set.size == 0:
        return L0, np.array([], dtype=int)
    if L0.getSize()[0] != L0.getSize()[1]:
        raise ValueError("L0 must be square")
    if index_set.min() < 0 or index_set.max() >= L0.getSize()[0]:
        raise ValueError("Neumann indices are out of bounds for the PETSc matrix")

    L0.zeroRows(np.asarray(index_set, dtype=np.int32), diag=0.0)
    # Assemble after zeroRows before switching to INSERT_VALUES.
    L0.assemble()
    L0.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    row_entries = _build_neumann_rows(
        mesh,
        index_set,
        index_mesh,
        border_tags,
        ngp,
        penalty,
        curvilinear_normals=curvilinear_normals,
    )
    for row_id, cols, vals in row_entries:
        L0.setValues(
            int(row_id),
            np.asarray(cols, dtype=np.int32),
            vals,
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
    L0.assemble()
    return L0, index_set
