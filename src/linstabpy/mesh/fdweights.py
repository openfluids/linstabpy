"""Finite-difference weights for arbitrary nodes."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fdweights(xi: float, x: ArrayLike, m: int) -> NDArray[np.float64]:
    """Return Fornberg finite-difference weights for derivative order ``m``."""
    x_nodes = np.asarray(x, dtype=float).reshape(-1)
    if x_nodes.size == 0:
        return np.zeros((0,), dtype=float)

    order = int(m)
    p = x_nodes.size - 1
    shifted = x_nodes - float(xi)

    @lru_cache(maxsize=None)
    def weight(q: int, j: int, k: int) -> float:
        if q < 0 or q > j:
            return 0.0
        if q == 0 and j == 0:
            return 1.0
        if k < j:
            numerator = shifted[j] * weight(q, j - 1, k) - q * weight(q - 1, j - 1, k)
            return float(numerator / (shifted[j] - shifted[k]))
        beta_num = float(np.prod(shifted[j - 1] - shifted[: j - 1])) if j > 1 else 1.0
        beta_den = float(np.prod(shifted[j] - shifted[:j]))
        beta = beta_num / beta_den
        return float(
            beta
            * (
                q * weight(q - 1, j - 1, j - 1)
                - shifted[j - 1] * weight(q, j - 1, j - 1)
            )
        )

    return np.array([weight(order, p, k) for k in range(p + 1)], dtype=float)
