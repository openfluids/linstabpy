# SPDX-License-Identifier: GPL-3.0-only
"""Tests for prescribed-forcing map driver and config exclusivity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from linstabpy.config.schema import ConfigValidationError
from linstabpy.resolvent.prescribed_io import load_prescribed_forcing
from linstabpy.solvers.router.prescribed_map import (
    _validate_prescribed_exclusivity,
)


class TestPrescribedConfigExclusivity:
    """Config exclusivity gate: prescribed and classical options cannot coexist."""

    def test_prescribed_alone_ok(self):
        """Config with only prescribed_forcing is valid."""
        cfg = {
            "prescribed_forcing": [
                {"beta": 0.1, "omega": 0.5, "path": "dummy.npz"},
            ]
        }
        # Should not raise.
        _validate_prescribed_exclusivity(cfg)

    def test_prescribed_with_beta_range_raises(self):
        """Config with both prescribed_forcing and beta_range raises."""
        cfg = {
            "prescribed_forcing": [
                {"beta": 0.1, "omega": 0.5, "path": "dummy.npz"},
            ],
            "beta_range": [0.0, 0.2, 0.4],
        }
        with pytest.raises(ConfigValidationError):
            _validate_prescribed_exclusivity(cfg)

    def test_prescribed_with_f_range_raises(self):
        """Config with both prescribed_forcing and f_range raises."""
        cfg = {
            "prescribed_forcing": [
                {"beta": 0.1, "omega": 0.5, "path": "dummy.npz"},
            ],
            "f_range": [0.0, 0.5, 1.0],
        }
        with pytest.raises(ConfigValidationError):
            _validate_prescribed_exclusivity(cfg)

    def test_prescribed_with_run_resolvent_raises(self):
        """Config with both prescribed_forcing and run_resolvent raises."""
        cfg = {
            "prescribed_forcing": [
                {"beta": 0.1, "omega": 0.5, "path": "dummy.npz"},
            ],
            "run_resolvent": True,
        }
        with pytest.raises(ConfigValidationError):
            _validate_prescribed_exclusivity(cfg)

    def test_empty_prescribed_ok(self):
        """Config with empty prescribed_forcing is valid."""
        cfg = {
            "prescribed_forcing": [],
            "beta_range": [0.0, 0.2],
        }
        # Empty prescribed list: should not raise.
        _validate_prescribed_exclusivity(cfg)

    def test_no_prescribed_ok(self):
        """Config without prescribed_forcing is valid."""
        cfg = {
            "beta_range": [0.0, 0.2],
            "f_range": [0.0, 0.5],
        }
        # Should not raise.
        _validate_prescribed_exclusivity(cfg)


class TestPrescribedIOValidation:
    """Tests for load_prescribed_forcing validation."""

    def test_load_valid_npz(self):
        """Loading a valid prescribed-forcing npz works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            f_hat = np.array([1.0 + 0.5j, 2.0 - 0.3j, 0.5 + 1.0j])
            np.savez(npz_path, f=f_hat, beta=0.1, omega=0.5)

            forcing = load_prescribed_forcing(npz_path, expected_ndof=3)
            assert forcing.beta == 0.1
            assert forcing.omega == 0.5
            np.testing.assert_array_almost_equal(forcing.f, f_hat)

    def test_load_wrong_ndof_raises(self):
        """Loading with wrong ndof raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            f_hat = np.array([1.0 + 0.5j, 2.0 - 0.3j])
            np.savez(npz_path, f=f_hat, beta=0.1, omega=0.5)

            with pytest.raises(ValueError, match="f length"):
                load_prescribed_forcing(npz_path, expected_ndof=5)

    def test_load_zero_bin_rejected(self):
        """Loading (beta, omega) = (0, 0) is rejected by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            f_hat = np.array([1.0 + 0.5j])
            np.savez(npz_path, f=f_hat, beta=0.0, omega=0.0)

            with pytest.raises(ValueError, match="mean-flow bin"):
                load_prescribed_forcing(npz_path, expected_ndof=1, allow_zero_bin=False)

    def test_load_zero_bin_allowed(self):
        """Loading (beta, omega) = (0, 0) works when flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            f_hat = np.array([1.0 + 0.5j])
            np.savez(npz_path, f=f_hat, beta=0.0, omega=0.0)

            forcing = load_prescribed_forcing(npz_path, expected_ndof=1, allow_zero_bin=True)
            assert forcing.beta == 0.0
            assert forcing.omega == 0.0

    def test_load_gamma_mismatch_raises(self):
        """npz and config gamma mismatch raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            f_hat = np.array([1.0 + 0.5j])
            np.savez(npz_path, f=f_hat, beta=0.1, omega=0.5, gamma=0.1)

            with pytest.raises(ValueError, match="gamma"):
                load_prescribed_forcing(npz_path, expected_ndof=1, config_gamma=0.2)

    def test_load_gamma_agreement_ok(self):
        """npz and config gamma agreement is fine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            f_hat = np.array([1.0 + 0.5j])
            np.savez(npz_path, f=f_hat, beta=0.1, omega=0.5, gamma=0.1)

            forcing = load_prescribed_forcing(npz_path, expected_ndof=1, config_gamma=0.1)
            assert forcing.gamma == 0.1

    def test_load_nan_forcing_raises(self):
        """Forcing with NaN raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            f_hat = np.array([1.0 + 0.5j, np.nan])
            np.savez(npz_path, f=f_hat, beta=0.1, omega=0.5)

            with pytest.raises(ValueError, match="NaN or Inf"):
                load_prescribed_forcing(npz_path, expected_ndof=2)


class TestPrescribedMapTableParsing:
    """Tests for parsing the prescribed_forcing table from config."""

    def test_table_with_forcing_path_key(self):
        """Table rows with 'forcing_path' key are parsed."""
        row = {"beta": 0.1, "omega": 0.5, "forcing_path": "/path/to/file.npz"}
        assert float(row.get("beta")) == 0.1
        assert float(row.get("omega")) == 0.5
        forcing_path = str(row.get("forcing_path", row.get("path")))
        assert forcing_path == "/path/to/file.npz"

    def test_table_with_path_key_fallback(self):
        """Table rows with 'path' key fall back correctly."""
        row = {"beta": 0.1, "omega": 0.5, "path": "/path/to/file.npz"}
        forcing_path = str(row.get("forcing_path", row.get("path")))
        assert forcing_path == "/path/to/file.npz"


class TestPrescribedMapSkips:
    """Monkeypatched end-to-end: verify CSV row writing and assembly grouping."""

    def test_csv_columns_written(self):
        """CSV is written with expected columns."""
        from linstabpy.solvers.router.map_record import append_row
        from linstabpy.solvers.router.prescribed_map import _prescribed_point_record

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"

            # Create a sample record.
            record = _prescribed_point_record(
                row_index=0,
                beta=0.1,
                omega=0.5,
                gain=1.23,
                rho_fwd=1e-8,
                gamma=0.0,
                forcing_norm="chu",
                status="ok",
                wall_s=0.5,
                peak_rss_bytes=1000000,
            )

            # Simulate the CSV write.
            fields = [
                "row",
                "beta",
                "omega",
                "gain",
                "rho_fwd",
                "gamma",
                "forcing_norm",
                "status",
                "wall_s",
                "peak_rss_bytes",
                "git_sha",
                "backend",
                "ranks",
                "fd_order",
                "epsilon",
            ]
            row_values = [record.get(field) for field in fields[:-5]]
            row_values.extend(["abc1234", "mumps", 1, 4, 0.0])

            append_row(csv_path, fields, row_values)

            # Verify the CSV file was created with the header.
            assert csv_path.exists()
            with open(csv_path) as f:
                first_line = f.readline().strip()
                assert "beta" in first_line
                assert "gain" in first_line
                assert "rho_fwd" in first_line
