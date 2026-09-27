# SPDX-License-Identifier: GPL-3.0-only
"""Tests for prescribed-forcing map driver and config exclusivity."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from linstabpy.config.schema import ConfigValidationError
from linstabpy.resolvent.prescribed_io import load_prescribed_forcing
from linstabpy.solvers.router.prescribed_map import (
    _require_row_matches_forcing,
    _validate_prescribed_exclusivity,
)


class TestPrescribedConfigSchema:
    """Schema validation for prescribed_forcing config."""

    def test_validate_config_prescribed_forcing_valid_list(self) -> None:
        """A minimal config with prescribed_forcing as list of dicts passes."""
        from linstabpy.config.schema import validate_config

        cfg = {
            "prescribed_forcing": [
                {"beta": 0.1, "omega": 0.5, "path": "dummy.npz"},
            ]
        }
        assert validate_config(cfg) is None

    def test_validate_config_prescribed_forcing_string_raises(self) -> None:
        """prescribed_forcing as string (not list) raises ConfigValidationError."""
        from linstabpy.config.schema import validate_config

        cfg = {
            "prescribed_forcing": "path/to/file.npz",
        }
        with pytest.raises(ConfigValidationError, match="prescribed_forcing.*expected list"):
            validate_config(cfg)

    def test_validate_config_prescribed_forcing_non_dict_row_raises(self) -> None:
        """prescribed_forcing list containing non-dict raises ConfigValidationError."""
        from linstabpy.config.schema import validate_config

        cfg = {
            "prescribed_forcing": [
                {"beta": 0.1, "omega": 0.5, "path": "dummy.npz"},
                "not_a_dict",
            ]
        }
        with pytest.raises(ConfigValidationError, match="prescribed_forcing.*expected dict"):
            validate_config(cfg)


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
        """Loading with a mismatched ndof raises."""
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
        """Npz and config gamma mismatch raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            f_hat = np.array([1.0 + 0.5j])
            np.savez(npz_path, f=f_hat, beta=0.1, omega=0.5, gamma=0.1)

            with pytest.raises(ValueError, match="gamma"):
                load_prescribed_forcing(npz_path, expected_ndof=1, config_gamma=0.2)

    def test_load_gamma_agreement_ok(self):
        """Npz and config gamma agreement is fine."""
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
            with Path(csv_path).open() as f:
                first_line = f.readline().strip()
                assert "beta" in first_line
                assert "gain" in first_line
                assert "rho_fwd" in first_line


def test_require_row_matches_forcing_beta_mismatch_raises() -> None:
    """A table beta that differs from the npz beyond 1e-9 relative raises."""
    with pytest.raises(ValueError, match="table beta"):
        _require_row_matches_forcing(
            row_index=0,
            forcing_path="t.npz",
            beta_row=1.0,
            omega_row=1.0,
            beta_npz=1.0 + 1e-6,
            omega_npz=1.0,
        )


def test_require_row_matches_forcing_omega_mismatch_raises() -> None:
    """A table omega that differs from the npz beyond 1e-9 relative raises."""
    with pytest.raises(ValueError, match="table omega"):
        _require_row_matches_forcing(
            row_index=0,
            forcing_path="t.npz",
            beta_row=1.0,
            omega_row=1.0,
            beta_npz=1.0,
            omega_npz=1.0 + 1e-6,
        )


def test_require_row_matches_forcing_exact_match_returns_none() -> None:
    """Matching table and npz (beta, omega) return None."""
    result = _require_row_matches_forcing(
        row_index=0,
        forcing_path="t.npz",
        beta_row=1.0,
        omega_row=1.0,
        beta_npz=1.0,
        omega_npz=1.0,
    )
    assert result is None


def test_write_prescribed_csv_empty_records_no_file(tmp_path: Path) -> None:
    """Calling _write_prescribed_csv with empty records creates no file."""
    from linstabpy.solvers.router.prescribed_map import _write_prescribed_csv

    csv_path = tmp_path / "gains.csv"
    _write_prescribed_csv(csv_path, [], {})
    assert not csv_path.exists()


def test_write_prescribed_csv_one_record_has_header(tmp_path: Path) -> None:
    """_write_prescribed_csv writes CSV with provenance columns for one record."""
    from linstabpy.solvers.router.prescribed_map import _write_prescribed_csv

    csv_path = tmp_path / "gains.csv"
    record = {
        "row": 0,
        "beta": 0.0,
        "omega": 60.0,
        "gain": 1.0e-4,
        "rho_fwd": 1.0e-12,
        "gamma": 0.0,
        "forcing_norm": "l2",
        "status": "ok",
        "wall_s": 0.1,
        "peak_rss_bytes": 1000,
    }
    _write_prescribed_csv(csv_path, [record], {"fd_order": 4})
    assert csv_path.exists()

    with Path(csv_path).open() as f:
        header = f.readline().strip()
        assert "git_sha" in header
        assert "backend" in header
        assert "fd_order" in header
        assert "beta" in header
        assert "gain" in header


def test_prescribed_map_empty_forcing_table_raises() -> None:
    """Empty prescribed_forcing list raises ValueError before mesh work."""
    from linstabpy.solvers.router.prescribed_map import prescribed_map

    cfg = {"prescribed_forcing": []}
    base_flow = object()
    with pytest.raises(ValueError, match="empty or missing"):
        prescribed_map(base_flow, cfg=cfg)


def test_prescribed_map_missing_forcing_config_raises() -> None:
    """Missing prescribed_forcing key raises ValueError."""
    from linstabpy.solvers.router.prescribed_map import prescribed_map

    cfg = {}
    base_flow = object()
    with pytest.raises(ValueError, match="empty or missing"):
        prescribed_map(base_flow, cfg=cfg)


def test_prescribed_map_path_key_alias_records_failure(tmp_path: Path) -> None:
    """Row with 'path' key and mismatched npz beta records failed status."""
    from examples.linstab2d.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.solvers.router.prescribed_map import prescribed_map

    Re = 6.0e5
    Ma = 0.3
    Pr = 0.7
    kappa = 1.4
    T_0 = 273.15

    nx = 12
    ny = 8
    dx = 0.01
    xrange = (5 * dx, 1.25 + dx)
    yrange = (-1.0, 1.0)
    fd_order = 4

    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order, alpha_filter="none")

    Y = cmesh.Y.copy()
    y_i = 0.01
    y_max = 0.05
    a = y_i * y_max / (y_max - 2 * y_i)
    b = 1 + 2 * a / y_max
    Y_phys = a * (1 + Y) / (b - Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    bf = load_blasius(mesh.X, Y_phys, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    x_pts = np.asarray(mesh.X).ravel(order="F")[mesh.used_ind]
    y_pts = np.asarray(mesh.Y).ravel(order="F")[mesh.used_ind]
    weight = np.exp(-(((x_pts - 0.65) / 0.08) ** 2 + ((y_pts - 0.01) / 0.01) ** 2))
    f_hat = np.tile(weight.astype(complex), 5)
    norm = np.linalg.norm(f_hat)
    if norm > 0.0:
        f_hat /= norm

    npz_path = tmp_path / "forcing.npz"
    np.savez(npz_path, f=f_hat, beta=0.001, omega=60.0)

    cfg = {
        "mesh": mesh,
        "Re": Re,
        "Ma": Ma,
        "Pr": Pr,
        "kappa": kappa,
        "T_0": T_0,
        "fd_order": fd_order,
        "model": "2D",
        "bc_recipe": [
            {
                "type": "dirichlet",
                "borders": "lrbt",
                "variables": "uvwT",
            }
        ],
        "bc_penalty": 1.0e4,
        "prescribed_forcing": [
            {
                "beta": 0.0,
                "omega": 60.0,
                "path": str(npz_path),
            }
        ],
        "forcing_norm": "l2",
    }

    result = prescribed_map(bf, cfg=cfg)

    assert len(result.point_records) == 1
    record = result.point_records[0]
    assert record["status"] == "failed"
    assert "exception" in record
    assert "table beta" in record["exception"]


def test_prescribed_map_row_without_path_key_raises() -> None:
    """Row with neither forcing_path nor path key raises ValueError."""
    from examples.linstab2d.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.solvers.router.prescribed_map import prescribed_map

    Re = 6.0e5
    Ma = 0.3
    Pr = 0.7
    kappa = 1.4
    T_0 = 273.15

    nx = 12
    ny = 8
    dx = 0.01
    xrange = (5 * dx, 1.25 + dx)
    yrange = (-1.0, 1.0)
    fd_order = 4

    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order, alpha_filter="none")

    Y = cmesh.Y.copy()
    y_i = 0.01
    y_max = 0.05
    a = y_i * y_max / (y_max - 2 * y_i)
    b = 1 + 2 * a / y_max
    Y_phys = a * (1 + Y) / (b - Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    bf = load_blasius(mesh.X, Y_phys, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    cfg = {
        "mesh": mesh,
        "Re": Re,
        "Ma": Ma,
        "Pr": Pr,
        "kappa": kappa,
        "T_0": T_0,
        "fd_order": fd_order,
        "model": "2D",
        "bc_recipe": [
            {
                "type": "dirichlet",
                "borders": "lrbt",
                "variables": "uvwT",
            }
        ],
        "bc_penalty": 1.0e4,
        "prescribed_forcing": [
            {
                "beta": 0.0,
                "omega": 60.0,
            }
        ],
        "forcing_norm": "l2",
    }

    with pytest.raises(ValueError, match="missing 'forcing_path'"):
        prescribed_map(bf, cfg=cfg)


def test_prescribed_map_solve_failure_records_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solve exception yields a failed record that carries the exception text."""
    import importlib

    from examples.linstab2d.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.solvers.router.prescribed_map import prescribed_map

    Re = 6.0e5
    Ma = 0.3
    Pr = 0.7
    kappa = 1.4
    T_0 = 273.15

    nx = 12
    ny = 8
    dx = 0.01
    xrange = (5 * dx, 1.25 + dx)
    yrange = (-1.0, 1.0)
    fd_order = 4

    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order, alpha_filter="none")

    Y = cmesh.Y.copy()
    y_i = 0.01
    y_max = 0.05
    a = y_i * y_max / (y_max - 2 * y_i)
    b = 1 + 2 * a / y_max
    Y_phys = a * (1 + Y) / (b - Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    bf = load_blasius(mesh.X, Y_phys, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    x_pts = np.asarray(mesh.X).ravel(order="F")[mesh.used_ind]
    y_pts = np.asarray(mesh.Y).ravel(order="F")[mesh.used_ind]
    weight = np.exp(-(((x_pts - 0.65) / 0.08) ** 2 + ((y_pts - 0.01) / 0.01) ** 2))
    f_hat = np.tile(weight.astype(complex), 5)
    norm = np.linalg.norm(f_hat)
    if norm > 0.0:
        f_hat /= norm

    npz_path = tmp_path / "forcing.npz"
    np.savez(npz_path, f=f_hat, beta=0.0, omega=60.0)

    cfg = {
        "mesh": mesh,
        "Re": Re,
        "Ma": Ma,
        "Pr": Pr,
        "kappa": kappa,
        "T_0": T_0,
        "fd_order": fd_order,
        "model": "2D",
        "bc_recipe": [
            {
                "type": "dirichlet",
                "borders": "lrbt",
                "variables": "uvwT",
            }
        ],
        "bc_penalty": 1.0e4,
        "prescribed_forcing": [
            {
                "beta": 0.0,
                "omega": 60.0,
                "forcing_path": str(npz_path),
            }
        ],
        "forcing_norm": "l2",
    }

    def raiser(*args: object, **kwargs: object) -> NoReturn:
        message = "injected solve failure"
        raise RuntimeError(message)

    mod = importlib.import_module("linstabpy.solvers.router.prescribed_map")
    monkeypatch.setattr(mod, "prescribed_forcing_response", raiser)

    result = prescribed_map(bf, cfg=cfg)

    assert len(result.point_records) == 1
    record = result.point_records[0]
    assert record["status"] == "failed"
    assert "exception" in record
    assert "injected solve failure" in record["exception"]


def test_prescribed_map_config_gamma_mismatch_records_failure(tmp_path: Path) -> None:
    """Config gamma differing from npz gamma records a failed row."""
    from examples.linstab2d.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.solvers.router.prescribed_map import prescribed_map

    Re = 6.0e5
    Ma = 0.3
    Pr = 0.7
    kappa = 1.4
    T_0 = 273.15

    nx = 12
    ny = 8
    dx = 0.01
    xrange = (5 * dx, 1.25 + dx)
    yrange = (-1.0, 1.0)
    fd_order = 4

    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order, alpha_filter="none")

    Y = cmesh.Y.copy()
    y_i = 0.01
    y_max = 0.05
    a = y_i * y_max / (y_max - 2 * y_i)
    b = 1 + 2 * a / y_max
    Y_phys = a * (1 + Y) / (b - Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    bf = load_blasius(mesh.X, Y_phys, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    x_pts = np.asarray(mesh.X).ravel(order="F")[mesh.used_ind]
    y_pts = np.asarray(mesh.Y).ravel(order="F")[mesh.used_ind]
    weight = np.exp(-(((x_pts - 0.65) / 0.08) ** 2 + ((y_pts - 0.01) / 0.01) ** 2))
    f_hat = np.tile(weight.astype(complex), 5)
    norm = np.linalg.norm(f_hat)
    if norm > 0.0:
        f_hat /= norm

    npz_path = tmp_path / "forcing.npz"
    np.savez(npz_path, f=f_hat, beta=0.001, omega=60.0, gamma=0.1)

    cfg = {
        "mesh": mesh,
        "Re": Re,
        "Ma": Ma,
        "Pr": Pr,
        "kappa": kappa,
        "T_0": T_0,
        "fd_order": fd_order,
        "model": "2D",
        "bc_recipe": [
            {
                "type": "dirichlet",
                "borders": "lrbt",
                "variables": "uvwT",
            }
        ],
        "bc_penalty": 1.0e4,
        "prescribed_forcing": [
            {
                "beta": 0.001,
                "omega": 60.0,
                "forcing_path": str(npz_path),
            }
        ],
        "forcing_norm": "l2",
        "resolvent_discount_gamma": 0.0,
    }

    result = prescribed_map(bf, cfg=cfg)

    assert len(result.point_records) == 1
    record = result.point_records[0]
    assert record["status"] == "failed"
    assert "exception" in record
    assert "gamma" in record["exception"]
