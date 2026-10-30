# SPDX-License-Identifier: GPL-3.0-only
"""Regression tests for Martini MATLAB HDF5 extraction."""

from __future__ import annotations

import pytest

from tests._matlab_parity import matlab_parity

# Every test in this module compares against MATLAB-derived reference data, which
# this repository does not ship. Deselected by default; see tests/_matlab_parity.py.
pytestmark = matlab_parity

from examples.linstab2d.matlab_hdf5 import extract_file
from tests._matlab_parity import matlab_parity, require_matlab_reference
from tests._paths import martini_export


# Raw-MATLAB-export extraction check. The martini_examples/ bundle is deliberately
# not committed, so this is opt-in: deselected by default, and a FAILURE rather than
# a skip once LINSTABPY_MATLAB_PARITY=1 asks for it.
def test_extract_blasius_results_exposes_resolvent_fields() -> None:
    path = martini_export("example_1_Blasius_results.mat")
    require_matlab_reference(path, description="the raw Martini MATLAB export")

    flat, inventory = extract_file(path)

    # save_martini2024_bundle.m stores the struct in a variable named ``results``.
    assert "results__results_Blasius__gains" in flat
    assert "results__results_Blasius__responses" in flat
    assert "results__results_Blasius__forces" in flat
    assert flat["results__results_Blasius__gains"].shape == (1, 3)
    assert flat["results__results_Blasius__responses"].dtype.kind == "c"
    assert any(field.path == "results/results_Blasius/responses" for field in inventory)


# Raw-MATLAB-export extraction check. The martini_examples/ bundle is deliberately
# not committed, so this is opt-in: deselected by default, and a FAILURE rather than
# a skip once LINSTABPY_MATLAB_PARITY=1 asks for it.
