# SPDX-License-Identifier: GPL-3.0-only
"""Pytest: symbolically verify Python coefficients against MATLAB source.

The two ``.m`` files are optional off-repo fixtures staged at
``martini_examples/``.  When absent the test SKIPS with an informative
message; it never passes vacuously.
"""

from __future__ import annotations

import pytest

from matlab_source_parity import (
    TOTAL,
    audit,
)
from tests._paths import martini_export, require_existing

_CYL_PATH = martini_export("getCoeffsLaminar.m")
_CART_PATH = martini_export("getCoeffsLaminar_Cartesian.m")

require_existing(_CYL_PATH, "MATLAB getCoeffsLaminar reference (martini_examples/)")
require_existing(_CART_PATH, "MATLAB getCoeffsLaminar_Cartesian reference (martini_examples/)")

# Run both audits once at collection time so both test functions share the result.
_RESULTS = audit(_CYL_PATH, _CART_PATH)


def _assert_clean(coord: str) -> None:
    result = _RESULTS[coord]
    if result.structural or result.parse_failures:
        lines = [
            f"{coord}: {len(result.structural)} structural diff(s),"
            f" {len(result.parse_failures)} parse failure(s)"
            f" out of {TOTAL} total"
        ]
        if result.structural:
            worst_key, worst_diff = max(result.structural.items(), key=lambda kv: len(str(kv[1])))
            lines.append(f"  worst: {worst_key}  delta = {worst_diff}")
            for k, v in sorted(result.structural.items()):
                lines.append(f"  {k}: matlab_minus_python = {v}")
        if result.parse_failures:
            lines.append("  parse failures:")
            for k, v in sorted(result.parse_failures.items()):
                lines.append(f"    {k}: {v}")
        pytest.fail("\n".join(lines))


def test_cylindrical_coefficients_match_matlab() -> None:
    """All 175 cylindrical coefficients match the MATLAB source (zero structural diffs)."""
    _assert_clean("cylindrical")


def test_cartesian_coefficients_match_matlab() -> None:
    """All 175 cartesian coefficients match the MATLAB source (zero structural diffs)."""
    _assert_clean("cartesian")
