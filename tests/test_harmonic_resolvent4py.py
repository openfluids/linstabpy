# SPDX-License-Identifier: GPL-3.0-only
"""Tests for harmonic resolvent solver via resolvent4py."""

from __future__ import annotations

import pytest


def test_harmonic_resolvent_import() -> None:
    """Test that the resolver4py caller can be imported without resolvent4py."""
    # This should not raise; imports are lazy
    from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains

    assert callable(harmonic_resolvent_gains)


def test_harmonic_resolvent_missing_resolvent4py() -> None:
    """Test that calling without resolvent4py raises ImportError with clear message."""
    import sys

    # Temporarily hide resolvent4py
    hidden = {}
    for key in list(sys.modules.keys()):
        if "resolvent4py" in key or "petsc4py" in key:
            hidden[key] = sys.modules.pop(key)

    try:
        from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains

        coo_fnames = [("rows_00.dat", "cols_00.dat", "vals_00.dat")]
        perts_freqs = [-1.0, 0.0, 1.0]

        with pytest.raises(ImportError, match="resolvent4py"):
            harmonic_resolvent_gains(
                coo_fnames,
                10,
                perts_freqs=perts_freqs,
            )
    finally:
        # Restore modules
        sys.modules.update(hidden)


@pytest.mark.skipif(
    True,
    reason="requires resolvent4py + petsc4py (run on c9)",
)
def test_harmonic_resolvent_end_to_end() -> None:
    """End-to-end harmonic resolvent test.

    This placeholder exists to show the test structure. Full testing requires
    resolvent4py and a complex PETSc build, which only exists on c9.
    """
    pytest.skip("requires resolvent4py + petsc4py (run on c9)")
