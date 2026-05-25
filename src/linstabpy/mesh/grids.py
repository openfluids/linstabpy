# SPDX-License-Identifier: GPL-3.0-only
"""Piecewise grid generation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _mvg(y: ArrayLike) -> NDArray[np.float64]:
    """Return a 5-point moving average with preserved endpoints."""
    values = np.asarray(y, dtype=float).reshape(-1)
    n = values.size
    if n <= 1:
        return values.copy()

    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    if n == 2:
        smoothed[1] = values[1]
        return smoothed

    smoothed[1] = float(np.mean(values[:3]))
    if n > 4:
        for i in range(2, n - 2):
            smoothed[i] = float(np.mean(values[i - 2 : i + 3]))
    if n > 3:
        smoothed[-2] = float(np.mean(values[-3:]))
    if n == 3:
        smoothed[1] = float(np.mean(values))
    smoothed[-1] = values[-1]
    return smoothed


def segmspace(x_segm: ArrayLike, nx_segm: ArrayLike, nx_smooth: int = 0) -> NDArray[np.float64]:
    """Return a smoothed piecewise-uniform grid."""
    x_nodes = np.asarray(x_segm, dtype=float).reshape(-1)
    counts = np.asarray(nx_segm, dtype=int).reshape(-1)
    smooth_passes = int(nx_smooth)

    if x_nodes.size != counts.size + 1:
        raise ValueError("x_segm must have one more entry than nx_segm")
    if counts.size == 0 or np.any(counts <= 0):
        raise ValueError("nx_segm must contain positive integers")
    if smooth_passes < 0:
        raise ValueError("nx_smooth must be non-negative")

    n_segm = x_nodes.size - 1
    total = int(np.sum(counts))

    if n_segm == 1:
        grid = np.linspace(x_nodes[0], x_nodes[1], counts[0], dtype=float)
    else:
        dz = np.diff(x_nodes) / counts
        cumsum = np.cumsum(counts)
        grid = np.zeros(total, dtype=float)

        grid[: counts[0]] = np.linspace(
            x_nodes[0],
            x_nodes[1] - dz[0] / 2.0,
            counts[0],
            dtype=float,
        )

        for si in range(1, n_segm - 1):
            start = int(cumsum[si - 1])
            npts = int(counts[si])
            grid[start : start + npts] = np.linspace(
                x_nodes[si] + dz[si] / 2.0,
                x_nodes[si + 1] - dz[si] / 2.0,
                npts,
                dtype=float,
            )

        si = n_segm - 1
        start = int(cumsum[si - 1])
        npts = int(counts[si])
        grid[start : start + npts] = np.linspace(
            x_nodes[si] + dz[si] / 2.0,
            x_nodes[si + 1],
            npts,
            dtype=float,
        )

    for _ in range(smooth_passes):
        grid[1:-1] = _mvg(grid[1:-1])
    return grid
