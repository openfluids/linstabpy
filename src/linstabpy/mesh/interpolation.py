"""Sparse interpolation operators on tensor-product grids."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix, csc_matrix


def interp_mat_2d(
    x_src: ArrayLike,
    y_src: ArrayLike,
    X_dst: ArrayLike,
    Y_dst: ArrayLike,
    method: str = "linear",
) -> csc_matrix:
    """Build a sparse interpolation matrix with Fortran-order flattening."""
    x_nodes = np.asarray(x_src, dtype=float).reshape(-1)
    y_nodes = np.asarray(y_src, dtype=float).reshape(-1)
    X = np.asarray(X_dst, dtype=float)
    Y = np.asarray(Y_dst, dtype=float)

    if x_nodes.size == 0 or y_nodes.size == 0:
        raise ValueError("x_src and y_src must be non-empty")
    if X.shape != Y.shape:
        raise ValueError("X_dst and Y_dst must have the same shape")

    nx = x_nodes.size
    ny = y_nodes.size
    n_src = nx * ny
    points_dst = np.column_stack((X.ravel(order="F"), Y.ravel(order="F")))
    n_dst = points_dst.shape[0]

    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    basis = np.zeros((nx, ny), dtype=float)

    for col in range(n_src):
        basis.fill(0.0)
        j = col % nx
        i = col // nx
        basis[j, i] = 1.0

        interp = RegularGridInterpolator(
            (x_nodes, y_nodes),
            basis,
            method=method,
            bounds_error=False,
            fill_value=0.0,
        )
        column = interp(points_dst)
        nz = np.flatnonzero(column)
        if nz.size == 0:
            continue
        row_chunks.append(nz.astype(int))
        col_chunks.append(np.full(nz.size, col, dtype=int))
        data_chunks.append(column[nz].astype(np.complex128))

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        data = np.concatenate(data_chunks)
    else:
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        data = np.array([], dtype=np.complex128)

    return coo_matrix((data, (rows, cols)), shape=(n_dst, n_src), dtype=np.complex128).tocsc()
