"""Regression tests for Martini MATLAB HDF5 extraction."""

from __future__ import annotations

import pytest

pytest.importorskip("h5py")

from validation.martini2024.tools.matlab_hdf5 import extract_file

from tests._paths import martini_export


def test_extract_blasius_results_exposes_resolvent_fields() -> None:
    path = martini_export("example_1_Blasius_results.mat")
    if not path.exists():
        pytest.skip(f"Missing Martini export: {path}")

    flat, inventory = extract_file(path)

    assert "results_Blasius__gains" in flat
    assert "results_Blasius__responses" in flat
    assert "results_Blasius__forces" in flat
    assert flat["results_Blasius__gains"].shape == (1, 3)
    assert flat["results_Blasius__responses"].dtype.kind == "c"
    assert any(field.path == "results_Blasius/responses" for field in inventory)


def test_extract_blasius_logs_decodes_strings_and_scalars() -> None:
    path = martini_export("example_1_Blasius_logs.mat")
    if not path.exists():
        pytest.skip(f"Missing Martini export: {path}")

    flat, _ = extract_file(path)

    steps = [str(step) for step in flat["timing__step"].reshape(-1)]
    assert any("resolvent" in step.lower() for step in steps)
    assert float(flat["timing_total"].reshape(-1)[0]) > 0.0
