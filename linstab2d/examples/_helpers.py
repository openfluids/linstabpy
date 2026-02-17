"""Shared helpers for example scripts.

Environment-variable parsing, grid reconstruction, and other utilities
used across multiple examples.
"""

from __future__ import annotations

import os

import numpy as np


def env_bool(name: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """Read an environment variable as a positive integer."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        value = int(v)
    except (TypeError, ValueError):
        print(f"Invalid {name}={v!r}; using {default}.")
        return default
    return max(1, value)


def env_float(name: str, default: float) -> float:
    """Read an environment variable as a float."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        print(f"Invalid {name}={v!r}; using {default}.")
        return default


def env_complex(name: str, default: complex) -> complex:
    """Read an environment variable as a complex number."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return complex(v)
    except (TypeError, ValueError):
        print(f"Invalid {name}={v!r}; using {default}.")
        return default


def to_grid(
    values: np.ndarray,
    used_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Scatter a DOF vector back onto the full 2D grid (NaN-filled)."""
    flat = np.full(int(np.prod(shape)), np.nan, dtype=complex)
    flat[np.asarray(used_indices)] = np.asarray(values)
    return flat.reshape(shape, order="F")
