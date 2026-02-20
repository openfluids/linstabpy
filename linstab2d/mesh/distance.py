"""Distance helpers for structured grids."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist


def min_distance(X: ArrayLike, Y: ArrayLike, indices: ArrayLike) -> NDArray[np.float64]:
    """Return minimum Euclidean distance to selected Fortran-order reference points."""
    x_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(Y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("X and Y must have the same shape")

    points = np.column_stack((x_arr.ravel(order="F"), y_arr.ravel(order="F")))
    n_points = points.shape[0]
    idx = np.asarray(indices, dtype=int).reshape(-1)

    if idx.size == 0:
        return np.full(x_arr.shape, np.inf, dtype=float)
    if np.any(idx < 0) or np.any(idx >= n_points):
        raise IndexError("indices contain out-of-bounds entries")

    refs = points[idx]
    distances = cdist(points, refs)
    return distances.min(axis=1).reshape(x_arr.shape, order="F")
