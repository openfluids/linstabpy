# SPDX-License-Identifier: GPL-3.0-only
"""RAM auto-selector tests."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.linalg.backends import factor_backend_available
from linstabpy.linalg.factorize import factorize
from linstabpy.linalg.shifted import shifted_operator
from linstabpy.solvers.router import stable_sweep
from linstabpy.solvers.router.ram_autoselect import (
    OperatingPoint,
    SymbolicProbeResult,
    _symbolic_analysis_probe,
    ram_autoselect,
)


def _pattern(n: int = 4) -> sp.csc_matrix:
    diagonal = 4.0 + 0.1j * np.arange(n)
    lower = -0.2 + 0.03j * np.arange(n - 1)
    upper = 0.1 - 0.02j * np.arange(n - 1)
    return sp.diags([lower, diagonal, upper], [-1, 0, 1], format="csc", dtype=complex)


def _operator() -> sp.csc_matrix:
    diagonal = np.array([2.1 + 0.1j, 2.8 - 0.2j, 3.5 + 0.3j, 4.4 - 0.1j, 5.2 + 0.2j])
    lower = np.array([0.02 - 0.01j, -0.03 + 0.02j, 0.01 + 0.02j, -0.02j])
    return sp.diags([lower, diagonal, 0.4 * lower], [-1, 0, 1], format="csc", dtype=complex)


def _fake_probe(fill_count: int, calls: list[tuple[tuple[int, int], int]] | None = None):
    def probe(pattern: sp.spmatrix, _cfg: dict[str, Any]) -> SymbolicProbeResult:
        if calls is not None:
            calls.append((tuple(pattern.shape), fill_count))
        return SymbolicProbeResult(
            fill_count=fill_count,
            bytes_per_nonzero=32.0,
            factor_bytes=1,  # intentionally inconsistent: selector must use fill_count*bytes/nonzero
            nnz=int(pattern.nnz),
            reorder="colamd",
        )

    return probe


def _selection_cfg(fill_count: int, budget: int, calls: list[tuple[tuple[int, int], int]] | None = None) -> dict[str, Any]:
    return {
        "available_ram_bytes": budget,
        "memory_fraction": 1.0,
        "operator_value_multiplier": 1.0,
        "symbolic_probe": _fake_probe(fill_count, calls),
        "compression_levels": (1.0e-4,),
        "compression_memory_ratios": (0.05,),
        "precision_candidates": ("double",),
        "restart": 5,
        "ranks": 1,
    }


def _sweep_cfg(n: int) -> dict[str, Any]:
    eye = np.eye(n, dtype=complex)
    return {
        "sigma_tol": 1.0e-7,
        "inv_norm_est": 1.0,
        "actual_solve_tol": 1.0e-10,
        "fgmres_rtol": 1.0e-11,
        "fgmres_maxiter": n,
        "n_modes": 1,
        "oversample": n,
        "seeds": (7, 8, 9),
        "probes_a": (eye[0], eye[-1]),
        "probes_b": (eye[1], eye[-2]),
        "W": np.ones(n, dtype=complex),
        "invW": np.ones(n, dtype=complex),
        "linearity_tol": 1.0e-9,
    }


def _mumps_oracle_gain(L0: sp.spmatrix, omega: complex) -> float:
    shifted = shifted_operator(L0, omega)
    fac = factorize(
        shifted,
        env_prefix="LINSTABPY_RESOLVENT",
        solver_cfg={"solver": {"resolvent": {"type": "direct", "factor_backend": "mumps"}}},
    )
    try:
        eye = np.eye(L0.shape[0], dtype=complex)
        dense_h = np.column_stack([fac.solve(eye[:, i]) for i in range(L0.shape[0])])
    finally:
        fac.destroy()
    return float(np.linalg.svd(dense_h, compute_uv=False)[0])


def test_ram_autoselect_is_reproducible_and_probe_fill_drives_decision() -> None:
    pattern = _pattern()
    calls: list[tuple[tuple[int, int], int]] = []
    tight_cfg = _selection_cfg(fill_count=500, budget=6_000, calls=calls)

    point = ram_autoselect(pattern, tight_cfg)
    repeat = ram_autoselect(pattern, tight_cfg)

    assert point == repeat
    assert point.backend == "compressed-sweep"
    assert calls == [((4, 4), 500), ((4, 4), 500)]

    # Same pattern and RAM, but a small empirical fill measurement: direct now fits.
    # The formula prior based on nnz is not allowed to override the probe measurement.
    low_fill = ram_autoselect(pattern, _selection_cfg(fill_count=5, budget=6_000))
    assert low_fill.backend == "direct"

    ample = ram_autoselect(pattern, _selection_cfg(fill_count=500, budget=100_000))
    assert ample.backend == "direct"


def test_real_symbolic_probe_measures_fill_and_drives_default_selection() -> None:
    # Exercise the production SciPy sparse-LU probe directly (no injection): it is
    # the empirical-calibration mechanism (R5) the selector relies on.
    pattern = _pattern(20)
    probe = _symbolic_analysis_probe(pattern, None)
    assert probe.backend == "scipy-splu"
    assert probe.fill_count >= pattern.nnz  # LU fill is at least the original structure
    assert probe.bytes_per_nonzero > 0.0
    assert probe.factor_bytes > 0

    # End-to-end through ram_autoselect with the REAL probe (no symbolic_probe key).
    ample = ram_autoselect(pattern, {"available_ram_bytes": 10**12})
    assert ample.backend == "direct"


def test_ram_autoselect_never_mutates_or_returns_physics_knobs() -> None:
    physics_cfg = {
        "available_ram_bytes": 100_000,
        "memory_fraction": 1.0,
        "symbolic_probe": _fake_probe(5),
        "grid": (64, 48),
        "fd_order": 6,
        "n_modes": 4,
        "beta_range": (0.0, 2.0),
        "frequency_range": (0.1, 0.8),
    }
    physics_before = {key: physics_cfg[key] for key in ("grid", "fd_order", "n_modes", "beta_range", "frequency_range")}

    point = ram_autoselect(_pattern(), physics_cfg)

    assert {key: physics_cfg[key] for key in physics_before} == physics_before
    assert {field.name for field in fields(OperatingPoint)} == {
        "backend",
        "reorder",
        "compression_level",
        "restart",
        "ranks",
        "precision",
    }
    for key in physics_before:
        assert not hasattr(point, key)


def test_direct_operating_point_matches_direct_mumps_oracle() -> None:
    if not factor_backend_available("mumps"):
        pytest.skip("SKIPPED: mumps backend unavailable")
    L0 = _operator()
    omega = 1.0 + 0.0j
    point = ram_autoselect(L0, _selection_cfg(fill_count=5, budget=100_000))
    assert point.backend == "direct"

    cfg = _sweep_cfg(L0.shape[0])
    result = stable_sweep.sweep_stable_regime(L0, [omega], [omega], cfg)
    oracle = _mumps_oracle_gain(L0, omega)

    assert result.gain[0] == pytest.approx(oracle, rel=cfg["sigma_tol"], abs=cfg["sigma_tol"])


def test_compressed_sweep_operating_point_matches_direct_mumps_or_loud_skips() -> None:
    if not factor_backend_available("strumpack"):
        pytest.skip("SKIPPED: strumpack backend unavailable")
    if not factor_backend_available("mumps"):
        pytest.skip("SKIPPED: mumps backend unavailable")
    L0 = _operator()
    omega = 1.0 + 0.0j
    point = ram_autoselect(L0, _selection_cfg(fill_count=500, budget=6_000))
    assert point.backend == "compressed-sweep"

    cfg = _sweep_cfg(L0.shape[0])
    cfg["fgmres_maxiter"] = max(cfg["fgmres_maxiter"], point.restart)
    cfg["compressed_anchor"] = {
        "enabled": True,
        "compression": "blr",
        "rel_tol": point.compression_level,
    }
    try:
        result = stable_sweep.sweep_stable_regime(L0, [omega], [omega], cfg)
    except RuntimeError as exc:
        if "adjoint_of_approx_factor" in str(exc):
            pytest.skip(f"SKIPPED: strumpack certified compressed-sweep unavailable: {exc}")
        raise
    oracle = _mumps_oracle_gain(L0, omega)

    assert result.gain[0] == pytest.approx(oracle, rel=cfg["sigma_tol"], abs=cfg["sigma_tol"])
