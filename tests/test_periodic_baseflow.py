# SPDX-License-Identifier: GPL-3.0-only
"""Tests for periodic base flow container."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from linstabpy.baseflow.periodic import PeriodicBaseFlow, load_periodic_baseflow


def test_periodic_baseflow_round_trip(tmp_path: Path) -> None:
    """Round-trip save and load periodic base flow."""
    # Create a small test case: N_h=2, ny=5, nx=4
    N_h = 2
    ny, nx = 5, 4
    omega0 = 2.0 * np.pi  # rad/s
    T_period = 1.0

    # Create synthetic harmonics (random complex)
    qhat = np.random.randn(N_h + 1, 5, ny, nx) + 1j * np.random.randn(N_h + 1, 5, ny, nx)
    # Ensure Qhat[0] is real (mean component)
    qhat[0] = np.real(qhat[0])
    # Set freestream on top row
    qhat[0, 0, -1, :] = 1.0  # rho
    qhat[0, 1, -1, :] = 1.0  # u
    qhat[0, 4, -1, :] = 1.0  # T

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    pbf = PeriodicBaseFlow(
        Qhat=qhat,
        T_period=T_period,
        x=x,
        y=y,
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
        extras={"source": "synthetic", "n_cycles": 3},
    )

    # Save to npz
    npz_path = tmp_path / "pbf.npz"
    np.savez(
        npz_path,
        Qhat=pbf.Qhat,
        T_period=pbf.T_period,
        x=pbf.x,
        y=pbf.y,
        Re=pbf.Re,
        Ma=pbf.Ma,
        Pr=pbf.Pr,
        kappa=pbf.kappa,
        T_0=pbf.T_0,
        source=pbf.extras["source"],
        n_cycles=pbf.extras["n_cycles"],
    )

    # Load back
    pbf2 = load_periodic_baseflow(npz_path)

    assert pbf2.N_h == N_h
    assert pbf2.T_period == T_period
    np.testing.assert_allclose(pbf2.Qhat, qhat)
    np.testing.assert_allclose(pbf2.x, x)
    np.testing.assert_allclose(pbf2.y, y)
    assert pbf2.Re == 1000.0
    assert pbf2.Ma == 0.5
    assert pbf2.Pr == 0.71


def test_periodic_baseflow_reconstruct() -> None:
    """Test Q(t) reconstruction."""
    N_h = 1
    ny, nx = 3, 2
    T_period = 2.0
    omega0 = 2.0 * np.pi / T_period

    # Create a simple test: Qhat[0] = mean, Qhat[1] = harmonic
    qhat = np.zeros((N_h + 1, 5, ny, nx), dtype=complex)

    # Mean: all 1
    qhat[0] = 1.0

    # Harmonic: amplitude 0.1
    qhat[1] = 0.1 * np.ones((5, ny, nx), dtype=complex)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    pbf = PeriodicBaseFlow(
        Qhat=qhat,
        T_period=T_period,
        x=x,
        y=y,
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
    )

    # Reconstruct at t=0
    bf = pbf.reconstruct(0.0)
    # Expected: Q(0) = 1 + 2*Re(0.1*exp(i*omega0*0)) = 1 + 2*0.1 = 1.2
    np.testing.assert_allclose(bf.RHO, 1.2, rtol=1e-14)

    # Reconstruct at t = T_period/4
    bf = pbf.reconstruct(T_period / 4.0)
    # Q(T/4) = 1 + 2*Re(0.1*exp(i*omega0*T/4)) = 1 + 2*0.1*cos(pi/2) = 1
    np.testing.assert_allclose(bf.RHO, 1.0, atol=1e-14)

    # Reconstruct at t = T_period (should equal t=0)
    bf_period = pbf.reconstruct(T_period)
    bf_zero = pbf.reconstruct(0.0)
    np.testing.assert_allclose(bf_period.RHO, bf_zero.RHO, rtol=1e-14)


def test_periodic_baseflow_sample_times() -> None:
    """Test time sampling."""
    T_period = 1.0
    N_h = 1
    qhat = np.ones((N_h + 1, 5, 3, 2), dtype=complex)
    qhat[0] = np.real(qhat[0])

    pbf = PeriodicBaseFlow(
        Qhat=qhat,
        T_period=T_period,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 0.5, 1.0]),
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
    )

    N_t = 10
    times = pbf.sample_times(N_t)

    assert len(times) == N_t
    assert times[0] == 0.0
    np.testing.assert_allclose(times[-1], (N_t - 1) * T_period / N_t)
    np.testing.assert_allclose(np.diff(times), T_period / N_t)


def test_load_periodic_baseflow_missing_key() -> None:
    """Test that missing required keys raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "bad.npz"
        # Missing Qhat
        np.savez(
            npz_path,
            T_period=1.0,
            x=np.array([0, 1]),
            y=np.array([0, 1]),
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )

        with pytest.raises(ValueError, match="Missing required keys"):
            load_periodic_baseflow(npz_path)


def test_load_periodic_baseflow_bad_shape() -> None:
    """Test that mismatched Qhat shape raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "bad_shape.npz"
        # Mismatched ndim
        qhat_bad = np.random.randn(2, 5, 3)  # should be 4D
        np.savez(
            npz_path,
            Qhat=qhat_bad,
            T_period=1.0,
            x=np.array([0, 1]),
            y=np.array([0, 1]),
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )

        with pytest.raises(ValueError, match="ndim"):
            load_periodic_baseflow(npz_path)


def test_load_periodic_baseflow_bad_period() -> None:
    """Test that non-positive T_period raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "bad_period.npz"
        qhat = np.ones((2, 5, 3, 2), dtype=complex)
        qhat[0] = np.real(qhat[0])
        qhat[0, 0, -1, :] = 1.0
        qhat[0, 1, -1, :] = 1.0
        qhat[0, 4, -1, :] = 1.0
        np.savez(
            npz_path,
            Qhat=qhat,
            T_period=-1.0,
            x=np.array([0, 1]),
            y=np.array([0, 1, 2]),
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )

        with pytest.raises(ValueError, match="T_period"):
            load_periodic_baseflow(npz_path)


def test_load_periodic_baseflow_freestream_check() -> None:
    """Test that bad freestream raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "bad_freestream.npz"
        qhat = np.ones((2, 5, 3, 2), dtype=complex)
        qhat[0] = np.real(qhat[0])
        # Bad rho on top row
        qhat[0, 0, -1, :] = 0.5  # outside 0.95-1.05 band
        qhat[0, 1, -1, :] = 1.0
        qhat[0, 4, -1, :] = 1.0

        np.savez(
            npz_path,
            Qhat=qhat,
            T_period=1.0,
            x=np.array([0, 1]),
            y=np.array([0, 1, 2]),
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )

        with pytest.raises(ValueError, match="Freestream"):
            load_periodic_baseflow(npz_path)


def test_load_periodic_baseflow_mesh_mismatch() -> None:
    """Test mesh shape mismatch detection."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "mesh_mismatch.npz"
        qhat = np.ones((2, 5, 3, 2), dtype=complex)
        qhat[0] = np.real(qhat[0])
        qhat[0, 0, -1, :] = 1.0
        qhat[0, 1, -1, :] = 1.0
        qhat[0, 4, -1, :] = 1.0

        np.savez(
            npz_path,
            Qhat=qhat,
            T_period=1.0,
            x=np.array([0, 1]),
            y=np.array([0, 1, 2]),
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )

        with pytest.raises(ValueError, match="Mesh shape mismatch"):
            load_periodic_baseflow(npz_path, mesh_shape=(5, 5))


def test_periodic_baseflow_qhat_ndim_error() -> None:
    """Test that a 3-D Qhat raises ValueError (the contract needs 4-D)."""
    qhat_3d = np.ones((2, 5, 3))
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="got ndim=3"):
        PeriodicBaseFlow(
            Qhat=qhat_3d,
            T_period=1.0,
            x=x,
            y=y,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )


def test_periodic_baseflow_qhat_variables_error() -> None:
    """Test that a 4-variable Qhat raises ValueError (the contract needs 5)."""
    qhat_4vars = np.ones((2, 4, 3, 2), dtype=complex)
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="must have 5 variables"):
        PeriodicBaseFlow(
            Qhat=qhat_4vars,
            T_period=1.0,
            x=x,
            y=y,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )


def test_periodic_baseflow_qhat_nonfinite_error() -> None:
    """Test that non-finite Qhat raises ValueError."""
    qhat = np.ones((2, 5, 3, 2), dtype=complex)
    qhat[0, 0, 0, 0] = np.nan
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        PeriodicBaseFlow(
            Qhat=qhat,
            T_period=1.0,
            x=x,
            y=y,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )


def test_periodic_baseflow_x_shape_error() -> None:
    """Test that x shape mismatch raises ValueError."""
    qhat = np.ones((2, 5, 3, 2), dtype=complex)
    x = np.array([0.0, 1.0, 2.0])  # length 3, should be 2 (nx)
    y = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="expected"):
        PeriodicBaseFlow(
            Qhat=qhat,
            T_period=1.0,
            x=x,
            y=y,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )


def test_periodic_baseflow_y_shape_error() -> None:
    """Test that y shape mismatch raises ValueError."""
    qhat = np.ones((2, 5, 3, 2), dtype=complex)
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.5, 1.0, 1.5])  # length 4, should be 3 (ny)
    with pytest.raises(ValueError, match="expected"):
        PeriodicBaseFlow(
            Qhat=qhat,
            T_period=1.0,
            x=x,
            y=y,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )


def test_periodic_baseflow_qhat0_imaginary_error() -> None:
    """Test that large imaginary part in Qhat[0] raises ValueError."""
    qhat = np.ones((2, 5, 3, 2), dtype=complex)
    qhat[0] = 1.0 + 0.5 * 1j  # Imaginary ratio 0.5, far above the 1e-10 gate.
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="imaginary part"):
        PeriodicBaseFlow(
            Qhat=qhat,
            T_period=1.0,
            x=x,
            y=y,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )


def test_load_periodic_baseflow_qhat_4_variables() -> None:
    """Test load raises ValueError when npz Qhat has 4 variables."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "bad_vars.npz"
        qhat = np.ones((2, 4, 3, 2), dtype=complex)  # 4 vars instead of 5
        qhat[0] = np.real(qhat[0])
        qhat[0, 0, -1, :] = 1.0
        qhat[0, 1, -1, :] = 1.0
        qhat[0, 3, -1, :] = 1.0  # Last var is index 3, not 4
        np.savez(
            npz_path,
            Qhat=qhat,
            T_period=1.0,
            x=np.array([0, 1]),
            y=np.array([0, 1, 2]),
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )
        with pytest.raises(ValueError, match="must have 5 variables"):
            load_periodic_baseflow(npz_path)


def test_load_periodic_baseflow_qhat0_imaginary() -> None:
    """Test load raises ValueError when npz Qhat[0] has large imaginary part."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "bad_imag.npz"
        qhat = np.ones((2, 5, 3, 2), dtype=complex)
        qhat[0] = 1.0 + 0.5 * 1j  # Large imaginary part
        qhat[0, 0, -1, :] = 1.0
        qhat[0, 1, -1, :] = 1.0
        qhat[0, 4, -1, :] = 1.0
        np.savez(
            npz_path,
            Qhat=qhat,
            T_period=1.0,
            x=np.array([0, 1]),
            y=np.array([0, 1, 2]),
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )
        with pytest.raises(ValueError, match="imaginary part"):
            load_periodic_baseflow(npz_path)
