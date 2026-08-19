# SPDX-License-Identifier: GPL-3.0-only
"""Load YAML case files and parse list values."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

# YAML 1.2 needs a signed exponent (1.0e+5). Case files use 1e5 and 1.0e5.
_SCI_FLOAT = re.compile(
    r"^[-+]?(?:[0-9][0-9_]*)?(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"
)


class _CaseLoader(yaml.SafeLoader):
    """Safe YAML loader that also accepts unsigned scientific floats."""


_CaseLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _SCI_FLOAT,
    list("-+0123456789."),
)


def load_config(path: str | Path) -> dict:
    """Load a YAML case file. Suffix must be .yaml or .yml."""
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Case file must use a .yaml suffix: {src}")
    data = yaml.load(src.read_text(encoding="utf-8"), Loader=_CaseLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {src}")
    return data


def parse_complex(value) -> complex:
    """Convert a YAML-friendly value to a Python complex number.

    Accepts:

    - ``[real, imag]`` list.
    - a plain number (int or float, treated as real).
    - a string parseable by ``complex()``.
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
    """Convert a list of YAML-friendly values to a list of complex numbers."""
    return [parse_complex(v) for v in value]


def parse_int_list(value) -> list[int]:
    """Convert a list of YAML-friendly values to a list of positive ints."""
    return [max(1, int(v)) for v in value]
