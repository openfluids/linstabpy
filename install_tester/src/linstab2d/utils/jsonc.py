"""JSONC (JSON with comments) loader and config-value parsers.

Provides a single ``load_jsonc()`` entry point used by examples,
validation cases, and results pipelines to read ``.jsonc`` parameter
files that drive all linstab2d runs.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonc(path: str | Path) -> dict:
    """Load a JSONC file (JSON with ``//`` and ``/* */`` comments)."""
    text = Path(path).read_text(encoding="utf-8")
    cleaned = _strip_jsonc_comments(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSONC object must be a dict: {path}")
    return data


def parse_complex(value) -> complex:
    """Convert a JSONC-friendly value to a Python complex number.

    Accepts:
    - ``[real, imag]`` list  (the canonical JSONC representation)
    - a plain number (int or float, treated as real)
    - a string parseable by ``complex()``
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected [real, imag] list, got length {len(value)}")
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, str):
        return complex(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to complex")


def parse_complex_list(value) -> list[complex]:
    """Convert a list of JSONC-friendly values to a list of complex numbers.

    Each element can be anything accepted by :func:`parse_complex`.
    """
    return [parse_complex(v) for v in value]


def parse_int_list(value) -> list[int]:
    """Convert a list of JSONC-friendly values to a list of positive ints."""
    return [max(1, int(v)) for v in value]


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _strip_jsonc_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments while preserving string literals."""
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
