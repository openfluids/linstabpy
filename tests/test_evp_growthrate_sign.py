# SPDX-License-Identifier: GPL-3.0-only
"""Characterization test for EVP temporal eigenvalue sign convention.

FINDING: The EVP growth-rate sign convention is verified correct in code.
  The temporal eigensolver computes omega = eig(i*L0), which under the
  shared e^{-i*omega*t} time convention means Im(omega) = Re(lambda) = growth rate.
  Therefore +Im(omega) > 0 indicates temporal instability (growing mode).
  This matches the reference Song/Savarino CSV convention (+omega_i = unstable).
  NO production code change and NO sign flip is required.

This is a verification characterization test, not a redesign. The reference
CSV is private (resolvent/archive/reference_song32deg_eigenspectrum.csv) and
is loaded via environment variable fallback; the test announces skip status
when absent (never silent pass).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.stability import eigenvalues as temporal_eigenvalues


def test_temporal_eigenvalue_sign_is_growth_positive() -> None:
    """Test that positive Im(omega) matches positive growth rate in the convention.

    CONVENTION ANCHOR:
      - Time evolution: exp(-i*omega*t)
      - A mode grows when Re(lambda) > 0, where dq/dt = L0 q (lambda = eig(L0))
      - Under e^{-i*omega*t}: omega = i*lambda, so Im(omega) = Re(lambda) = growth
      - Therefore +Im(omega) = unstable (growing), -Im(omega) = stable (damped)

    This test embeds four eigenvalues in a diagonal matrix:
      - lam[0]: +0.05 real (GROWING) with +0.40 oscillation
      - lam[1,2,3]: negative real (damped) with various oscillations

    The solver must report the leading eigenvalue with Im(omega) > 0, and that
    Im(omega) must equal the real part of the eigenvalue (the growth rate).
    """
    # Four eigenvalues: one growing (Re=+0.05), three damped (Re<0)
    lam = np.array([0.05 + 0.40j, -0.30 + 1.1j, -0.50 - 0.7j, -0.80 + 0.2j], dtype=complex)
    L0 = sp.diags(lam, format="csc", dtype=complex)

    # Call the temporal eigenvalue solver, targeting near the first eigenvalue.
    target = 1j * lam[0]  # Target for shift-invert: i*lambda
    omega, modes = temporal_eigenvalues(L0, n_modes=4, target=target)

    # The leading eigenvalue must have Im(omega) > 0 (unstable).
    assert omega[0].imag > 0, (
        f"Leading eigenvalue Im(omega)={omega[0].imag} must be positive (unstable). Full omega={omega}"
    )

    # The leading Im(omega) must match the growth rate Re(lambda[0]).
    assert np.isclose(omega[0].imag, lam[0].real, atol=1e-8), (
        f"Im(omega[0])={omega[0].imag} must equal growth rate Re(lam[0])={lam[0].real}. Verifies the e^{{-i*omega*t}} convention."
    )

    # The other three damped eigenvalues must have Im(omega) < 0 (stable).
    # Match them by finding the nearest omega to i*lam[k].
    for k in range(1, 4):
        expected_omega = 1j * lam[k]
        distances = np.abs(omega[1:] - expected_omega)
        min_idx = np.argmin(distances)
        closest_omega = omega[1 + min_idx]
        assert closest_omega.imag < 0, (
            f"Damped eigenvalue k={k}: Im(omega)={closest_omega.imag} must be negative (stable). Expected near {expected_omega}"
        )


def test_reference_song32_unstable_band_is_positive_omega_i() -> None:
    """Test that the Song32-deg reference CSV confirms +omega_i = unstable convention.

    REFERENCE LOAD:
      The Song/Savarino reference CSV lives at:
        (1) env var LINSTABPY_SONG32_CSV if set and exists
        (2) resolved relative to the linstabpy repo root: ../archive/reference_song32deg_eigenspectrum.csv
        (3) resolved from this test file: ../../../archive/reference_song32deg_eigenspectrum.csv
      If none found: pytest.skip with announcement (never silent pass).

    VERIFICATION TARGETS:
      The reference CSV documents the Song theta=32deg (globally unstable) case
      under the same e^{-i*omega*t} convention as linstabpy. It provides:
        col1 = lambda/L (dimensionless spanwise wavelength)
        col2 = omega_i (growth rate; +omega_i = unstable)

      This test verifies:
        (a) Non-empty unstable band exists (at least one row with omega_i > 0)
        (b) Unstable band is localised in wavelength: 0.15 < lambda/L < 0.85
        (c) Peak growth ~+0.0558 lies in [0.045, 0.065]
        (d) Peak occurs near lambda/L~0.327, i.e. in [0.30, 0.45]

      These assertions document the reference's sign meaning: when linstabpy
      reports +Im(omega), it must recover this same unstable band.
    """
    # Resolve the reference CSV path.
    csv_path = None

    # (1) Try environment variable
    env_csv = os.environ.get("LINSTABPY_SONG32_CSV")
    if env_csv and Path(env_csv).exists():
        csv_path = Path(env_csv)

    # (2) Try relative to linstabpy repo root (../archive/...)
    if csv_path is None:
        repo_root = Path(__file__).resolve().parents[2]  # linstabpy/ dir
        candidate = repo_root.parent / "archive" / "reference_song32deg_eigenspectrum.csv"
        if candidate.exists():
            csv_path = candidate

    # (3) Try deeper relative search (in case test is reorganized)
    if csv_path is None:
        for i in range(2, 5):
            candidate = (
                Path(__file__).resolve().parents[i]
                / "archive"
                / "reference_song32deg_eigenspectrum.csv"
            )
            if candidate.exists():
                csv_path = candidate
                break

    # If not found, announce skip (never silent pass).
    if csv_path is None:
        pytest.skip(
            "SKIPPED: missing reference CSV (private resolvent/archive/reference_song32deg_eigenspectrum.csv); "
            "set LINSTABPY_SONG32_CSV to enable"
        )

    # Parse the CSV: skip comment lines and blank lines, extract two float columns.
    wavelengths = []  # col1: lambda/L
    growth_rates = []  # col2: omega_i

    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty rows and comment lines.
            if not line or line.startswith("#"):
                continue
            # Split on whitespace.
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lambda_over_L = float(parts[0])
                omega_i = float(parts[1])
                wavelengths.append(lambda_over_L)
                growth_rates.append(omega_i)
            except ValueError:
                continue

    wavelengths = np.array(wavelengths)
    growth_rates = np.array(growth_rates)

    # (a) There must be at least one unstable mode (omega_i > 0).
    unstable_mask = growth_rates > 0
    assert np.any(unstable_mask), (
        "No unstable modes found (all omega_i <= 0). CSV data integrity check failed."
    )

    # (b) Unstable modes must be localised in wavelength.
    unstable_wavelengths = wavelengths[unstable_mask]
    assert np.all(unstable_wavelengths >= 0.15), (
        f"Some unstable modes have lambda/L < 0.15 (outside expected band): {unstable_wavelengths}"
    )
    assert np.all(unstable_wavelengths <= 0.85), (
        f"Some unstable modes have lambda/L > 0.85 (outside expected band): {unstable_wavelengths}"
    )

    # (c) Peak growth should be in [0.045, 0.065] (reference is ~+0.0558).
    max_growth = np.max(growth_rates)
    assert 0.045 <= max_growth <= 0.065, (
        f"Peak growth omega_i={max_growth} outside expected range [0.045, 0.065]"
    )

    # (d) Peak occurs near lambda/L~0.327, i.e. in [0.30, 0.45].
    peak_idx = np.argmax(growth_rates)
    peak_wavelength = wavelengths[peak_idx]
    assert 0.30 <= peak_wavelength <= 0.45, (
        f"Peak growth at lambda/L={peak_wavelength} outside expected range [0.30, 0.45]"
    )

    # All assertions passed: the reference convention is documented and verified.
    # Assertion messages explain that this is the target linstabpy must match
    # when reporting +Im(omega) = unstable under the shared e^{-i*omega*t} convention.
