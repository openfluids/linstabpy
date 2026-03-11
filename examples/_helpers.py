"""Shared helpers for example scripts.

Environment-variable parsing, grid reconstruction, and other utilities
used across multiple examples.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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


def load_jsonc(path: str | Path) -> dict:
    """Load a JSONC file (JSON with // and /* */ comments)."""
    text = Path(path).read_text(encoding="utf-8")
    cleaned = _strip_jsonc_comments(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSONC object must be a dict: {path}")
    return data


def _strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving string literals."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    in_line_comment = False
    in_block_comment = False
    escape = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def to_grid(
    values: np.ndarray,
    used_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Scatter a DOF vector back onto the full 2D grid (NaN-filled)."""
    flat = np.full(int(np.prod(shape)), np.nan, dtype=complex)
    flat[np.asarray(used_indices)] = np.asarray(values)
    return flat.reshape(shape, order="F")


def leading_mode_real_frequency(
    omega: np.ndarray,
    imag_shift: float = 0.0,
) -> complex:
    """Return the real part of the least-damped mode plus an imag shift."""
    values = np.asarray(omega, dtype=complex).reshape(-1)
    if values.size == 0:
        raise ValueError("omega must contain at least one eigenvalue")
    lead = values[int(np.argmax(np.imag(values)))]
    return complex(float(np.real(lead)), float(imag_shift))
