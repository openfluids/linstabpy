# SPDX-License-Identifier: GPL-3.0-only
"""Tests for harmonic operator blocks."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.baseflow.periodic import PeriodicBaseFlow
from linstabpy.operators.harmonic_blocks import (
    HarmonicBlocks,
    assemble_harmonic_blocks,
)


@pytest.fixture
def small_mesh_2d():
    """20x30 mesh on [0,2]x[0,1], order 4."""
    from linstabpy.mesh import Mesh

    return Mesh.create(
        xrange=(0.0, 2.0),
        yrange=(0.0, 1.0),
        nx=20,
        ny=30,
        fd_order=4,
    )


@pytest.fixture
def steady_pbf(small_mesh_2d) -> PeriodicBaseFlow:
    """Steady base flow (only k=0, no oscillation)."""
    ny, nx = small_mesh_2d.Y.shape[0], small_mesh_2d.X.shape[1]

    # Extract 1D coordinate vectors from the 2D grids
    x_1d = small_mesh_2d.X[0, :]
    y_1d = small_mesh_2d.Y[:, 0]

    # Only k=0 (steady)
    qhat = np.zeros((1, 5, ny, nx), dtype=complex)
    # Synthetic steady profile
    qhat[0, 0] = 1.0  # rho
    qhat[0, 1] = 1.0  # u
    qhat[0, 2] = 0.0  # v
    qhat[0, 3] = 0.0  # w
    qhat[0, 4] = 1.0  # T

    return PeriodicBaseFlow(
        Qhat=qhat,
        T_period=1.0,
        x=x_1d,
        y=y_1d,
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
    )


def test_harmonic_blocks_steady(small_mesh_2d, steady_pbf) -> None:
    """Test steady base flow: Ahat_0 == build_lnse_system, higher Ahat_k ~ 0."""
    from linstabpy.operators.assembly import build_lnse_system

    # Build reference LNSE operator at steady state
    bf = steady_pbf.reconstruct(0.0)
    L0_ref, _, _ = build_lnse_system(small_mesh_2d, bf, kx=0.0, model="2D")

    # Assemble harmonic blocks from steady base flow
    blocks = assemble_harmonic_blocks(
        small_mesh_2d,
        steady_pbf,
        N_h=0,
        N_t=4,  # Small N_t for steady
        kx=0.0,
        model="2D",
    )

    assert len(blocks.Ahat) == 1
    assert blocks.N_h == 0
    assert blocks.N_t == 4

    # Ahat[0] should equal L0_ref to high precision
    L0_ref_csr = L0_ref.tocsr()
    diff = L0_ref_csr - blocks.Ahat[0]
    if diff.nnz > 0:
        max_err = np.max(np.abs(diff.data))
        assert max_err < 1e-10, f"Ahat[0] differs from L0_ref by {max_err}"


def test_harmonic_blocks_manufactured(small_mesh_2d) -> None:
    r"""Test manufactured harmonic: Q = Qbar + eps*Qtilde*cos(omega0*t).

    For a linear operator in Q, Ahat[1] should scale linearly with eps.
    Verify the scaling and check O(eps^2) accuracy via central differences.
    """
    from linstabpy.operators.assembly import build_lnse_system

    ny, nx = small_mesh_2d.Y.shape[0], small_mesh_2d.X.shape[1]
    x_1d = small_mesh_2d.X[0, :]
    y_1d = small_mesh_2d.Y[:, 0]

    omega0 = 2.0 * np.pi
    T_period = 1.0

    # Build Qbar (mean)
    qbar = np.zeros((5, ny, nx))
    qbar[0] = 1.0  # rho
    qbar[1] = 1.0  # u
    qbar[2] = 0.0  # v
    qbar[3] = 0.0  # w
    qbar[4] = 1.0  # T

    # Build Qtilde (perturbation, kept small so T stays positive)
    qtilde = np.zeros((5, ny, nx))
    qtilde[0] = 0.02  # small rho perturbation
    qtilde[1] = 0.01  # small u perturbation
    qtilde[2:4] = 0.0
    qtilde[4] = 0.01  # small T perturbation

    def test_one_eps(eps_val: float) -> tuple[np.ndarray, np.ndarray]:
        """Assemble blocks and compute reference via central differences."""
        # Q(t) = Qbar + eps*Qtilde*cos(omega0*t)
        # Harmonics: Qhat[0] = Qbar, Qhat[1] = eps*Qtilde/2
        qhat = np.zeros((2, 5, ny, nx), dtype=complex)
        qhat[0] = qbar
        qhat[1] = (eps_val / 2.0) * qtilde

        pbf = PeriodicBaseFlow(
            Qhat=qhat,
            T_period=T_period,
            x=x_1d,
            y=y_1d,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )

        # Assemble harmonic blocks (k=-1..0..+1)
        N_t = 64
        blocks = assemble_harmonic_blocks(
            small_mesh_2d,
            pbf,
            N_h=1,
            N_t=N_t,
            kx=0.0,
            model="2D",
        )

        # Compute reference: central difference gives dL0/dQ * Qtilde
        qhat_plus = np.zeros((2, 5, ny, nx), dtype=complex)
        qhat_plus[0] = qbar + eps_val * qtilde
        qhat_plus[1] = 0.0

        qhat_minus = np.zeros((2, 5, ny, nx), dtype=complex)
        qhat_minus[0] = qbar - eps_val * qtilde
        qhat_minus[1] = 0.0

        pbf_plus = PeriodicBaseFlow(
            Qhat=qhat_plus,
            T_period=T_period,
            x=x_1d,
            y=y_1d,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )
        pbf_minus = PeriodicBaseFlow(
            Qhat=qhat_minus,
            T_period=T_period,
            x=x_1d,
            y=y_1d,
            Re=1000.0,
            Ma=0.5,
            Pr=0.71,
            kappa=1.4,
            T_0=300.0,
        )

        bf_plus = pbf_plus.reconstruct(0.0)
        bf_minus = pbf_minus.reconstruct(0.0)

        L0_plus, _, _ = build_lnse_system(small_mesh_2d, bf_plus, kx=0.0, model="2D")
        L0_minus, _, _ = build_lnse_system(small_mesh_2d, bf_minus, kx=0.0, model="2D")

        # Central difference: (L0(Qbar + eps*Qtilde) - L0(Qbar - eps*Qtilde)) / (2*eps)
        L0_central = (L0_plus - L0_minus) / (2.0 * eps_val)
        L0_central_csr = L0_central.tocsr()

        # Ahat[1] is at index N_h + 1 = 2 (for N_h=1)
        Ahat_1_pos = blocks.Ahat[2]

        # Ahat[1] should equal (eps/2) * L0_central (linear scaling)
        return Ahat_1_pos, L0_central_csr

    # Test at two eps values
    ahat_eps, ref_eps = test_one_eps(0.05)
    ahat_half, ref_half = test_one_eps(0.025)

    # Check linear scaling: error in Ahat[k] / (eps/2) should fall by factor 4 when eps -> eps/2
    eps1, eps2 = 0.05, 0.025
    scaled_ahat_eps = ahat_eps / (eps1 / 2.0)
    scaled_ahat_half = ahat_half / (eps2 / 2.0)

    # Compare scaled versions to reference
    diff_eps = scaled_ahat_eps - ref_eps
    diff_half = scaled_ahat_half - ref_half

    if diff_eps.nnz > 0:
        norm_diff_eps = np.max(np.abs(diff_eps.data)) if diff_eps.nnz > 0 else 0.0
    else:
        norm_diff_eps = 0.0

    if diff_half.nnz > 0:
        norm_diff_half = np.max(np.abs(diff_half.data)) if diff_half.nnz > 0 else 0.0
    else:
        norm_diff_half = 0.0

    # Both should be small (within discretization error)
    # Check that the ratio falls by a factor of about 4 (O(eps^2)).
    if norm_diff_eps > 1e-12 and norm_diff_half > 1e-12:
        rate = norm_diff_eps / norm_diff_half
        # Should be ~4 for O(eps^2) convergence, but accept wider range due to discretization
        assert 2.5 <= rate <= 6.0, (
            f"Convergence rate {rate} not in [2.5, 6]; expected O(eps^2), first order would give 2"
        )


def test_harmonic_blocks_asymmetry_kx_zero(small_mesh_2d) -> None:
    """Test that conjugation asymmetry is ~0 at kx=0 (where Ahat[-k] = conj(Ahat[k]))."""
    ny, nx = small_mesh_2d.Y.shape[0], small_mesh_2d.X.shape[1]
    x_1d = small_mesh_2d.X[0, :]
    y_1d = small_mesh_2d.Y[:, 0]

    # Simple two-harmonic case
    qhat = np.zeros((2, 5, ny, nx), dtype=complex)
    qhat[0, 0] = 1.0  # rho
    qhat[0, 1] = 1.0  # u
    qhat[0, 2:4] = 0.0
    qhat[0, 4] = 1.0  # T
    qhat[1, 0] = 0.05  # small harmonic
    qhat[1, 1] = 0.02
    qhat[1, 4] = 0.02

    pbf = PeriodicBaseFlow(
        Qhat=qhat,
        T_period=1.0,
        x=x_1d,
        y=y_1d,
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
    )

    blocks = assemble_harmonic_blocks(
        small_mesh_2d,
        pbf,
        N_h=1,
        N_t=8,
        kx=0.0,  # Real operator: Ahat[-k] = conj(Ahat[k])
        model="2D",
    )

    # At kx=0, conjugation asymmetry should be very small (~1e-12)
    assert blocks.conjugation_asymmetry is not None
    assert blocks.conjugation_asymmetry < 1e-12, (
        f"At kx=0, asymmetry should be ~0, got {blocks.conjugation_asymmetry}"
    )


def test_harmonic_blocks_asymmetry_kx_nonzero(small_mesh_2d) -> None:
    """Test that conjugation asymmetry is >1e-3 at kx!=0 (genuinely complex operator)."""
    ny, nx = small_mesh_2d.Y.shape[0], small_mesh_2d.X.shape[1]
    x_1d = small_mesh_2d.X[0, :]
    y_1d = small_mesh_2d.Y[:, 0]

    # Simple two-harmonic case
    qhat = np.zeros((2, 5, ny, nx), dtype=complex)
    qhat[0, 0] = 1.0  # rho
    qhat[0, 1] = 1.0  # u
    qhat[0, 2:4] = 0.0
    qhat[0, 4] = 1.0  # T
    qhat[1, 0] = 0.05
    qhat[1, 1] = 0.02
    qhat[1, 4] = 0.02

    pbf = PeriodicBaseFlow(
        Qhat=qhat,
        T_period=1.0,
        x=x_1d,
        y=y_1d,
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
    )

    blocks = assemble_harmonic_blocks(
        small_mesh_2d,
        pbf,
        N_h=1,
        N_t=8,
        kx=0.5,  # Complex operator: i*kx derivatives make Ahat[-k] != conj(Ahat[k])
        model="2D",
    )

    # At kx!=0, asymmetry should be > 1e-3 (measured ~6.6e-2 at kx=0.5 on this mesh)
    assert blocks.conjugation_asymmetry is not None
    assert blocks.conjugation_asymmetry > 1e-3, (
        f"At kx=0.5, asymmetry should be >1e-3, got {blocks.conjugation_asymmetry}"
    )


def test_harmonic_blocks_tail_ratio(small_mesh_2d, steady_pbf) -> None:
    """Test tail_ratio computation when N_t allows it."""
    # With N_t = 32 and N_h = 5, we can compute tail_ratio
    blocks = assemble_harmonic_blocks(
        small_mesh_2d,
        steady_pbf,
        N_h=5,
        N_t=32,
        kx=0.0,
        model="2D",
    )

    assert blocks.tail_ratio is not None
    # Measured 4.22e-11; bound at 5e-10 (10x margin).
    assert np.isfinite(blocks.tail_ratio) and 0.0 <= blocks.tail_ratio <= 5e-10


def test_harmonic_blocks_stable_under_N_t_doubling(small_mesh_2d) -> None:
    """Doubling N_t leaves every kept Ahat_k unchanged to the measured floor.

    Same manufactured base as test_harmonic_blocks_manufactured, N_h=2.
    A misaligned union pattern or FFT bin would move Ahat_k by O(1).
    """
    ny, nx = small_mesh_2d.Y.shape[0], small_mesh_2d.X.shape[1]
    x_1d = small_mesh_2d.X[0, :]
    y_1d = small_mesh_2d.Y[:, 0]

    # Build mean
    qbar = np.zeros((5, ny, nx))
    qbar[0] = 1.0
    qbar[1] = 1.0
    qbar[4] = 1.0

    # Build perturbation
    qtilde = np.zeros((5, ny, nx))
    qtilde[0] = 0.02
    qtilde[1] = 0.01
    qtilde[4] = 0.01

    # Manufactured periodic base
    eps_val = 0.05
    qhat = np.zeros((2, 5, ny, nx), dtype=complex)
    qhat[0] = qbar
    qhat[1] = (eps_val / 2.0) * qtilde

    pbf = PeriodicBaseFlow(
        Qhat=qhat,
        T_period=1.0,
        x=x_1d,
        y=y_1d,
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
    )

    # Assemble at N_t=32 and N_t=64 with N_h=2
    N_h = 2
    blocks_32 = assemble_harmonic_blocks(
        small_mesh_2d,
        pbf,
        N_h=N_h,
        N_t=32,
        kx=0.0,
        model="2D",
    )
    blocks_64 = assemble_harmonic_blocks(
        small_mesh_2d,
        pbf,
        N_h=N_h,
        N_t=64,
        kx=0.0,
        model="2D",
    )

    # Per-k relative change: ||Ahat_k(64) - Ahat_k(32)||_F / ||Ahat_k(64)||_F
    # Measured: max 1.96e-10; tolerance 2e-9 (10x margin).
    for k_idx in range(len(blocks_64.Ahat)):
        A_64 = blocks_64.Ahat[k_idx]
        A_32 = blocks_32.Ahat[k_idx]
        # Frobenius norms from the stored entries; complex-safe (uses |z|).
        diff = A_64 - A_32
        norm_diff = float(np.linalg.norm(diff.data)) if diff.nnz else 0.0
        norm_A64 = float(np.linalg.norm(A_64.data))
        rel_err = norm_diff / norm_A64 if norm_A64 > 0.0 else 0.0
        assert rel_err < 2e-9, f"k={k_idx - N_h}: rel_change {rel_err:.4e} exceeds 2e-9"


def test_harmonic_blocks_rejects_aliasing_N_t(small_mesh_2d, steady_pbf) -> None:
    """Test that aliasing guard rejects N_t <= 2*N_h."""
    # N_h=3, N_t=6: extracting k=-3..3 from 6-point FFT aliases (N_t must be > 2*N_h)
    with pytest.raises(ValueError, match="N_t=.*must be.*2.*N_h"):
        assemble_harmonic_blocks(
            small_mesh_2d,
            steady_pbf,
            N_h=3,
            N_t=6,
            kx=0.0,
            model="2D",
        )


def test_harmonic_blocks_ahat_length_mismatch() -> None:
    """Test that len(Ahat) != 2*N_h + 1 raises ValueError."""
    # N_h=1 needs 3 blocks (k=-1, 0, +1); give only 2.
    A_list = [sp.csr_matrix(np.eye(3)), sp.csr_matrix(np.eye(3))]
    with pytest.raises(ValueError, match="must have"):
        HarmonicBlocks(
            Ahat=A_list,
            N_h=1,
            N_t=4,
        )


def test_harmonic_blocks_odd_N_t_error(small_mesh_2d, steady_pbf) -> None:
    """Test that odd N_t raises ValueError."""
    # N_t=5 is odd; the assembler rejects it before any sampling.
    with pytest.raises(ValueError, match="must be even"):
        assemble_harmonic_blocks(
            small_mesh_2d,
            steady_pbf,
            N_h=1,
            N_t=5,
            kx=0.0,
            model="2D",
        )


def test_write_resolvent4py_coo_petsc_missing(tmp_path) -> None:
    """Test write_resolvent4py_coo raises ImportError when petsc4py missing."""
    import importlib.util

    if importlib.util.find_spec("petsc4py") is not None:
        pytest.skip("petsc4py installed; the guard needs its absence")

    # Create minimal HarmonicBlocks with one identity block
    A_list = [sp.csr_matrix(np.eye(3))]
    blocks = HarmonicBlocks(
        Ahat=A_list,
        N_h=0,
        N_t=2,
    )

    from linstabpy.operators.harmonic_blocks import write_resolvent4py_coo

    with pytest.raises(ImportError, match="petsc4py not found"):
        write_resolvent4py_coo(blocks, str(tmp_path))


def test_write_resolvent4py_vecs_petsc_missing(tmp_path) -> None:
    """Test write_resolvent4py_vecs raises ImportError when petsc4py missing."""
    import importlib.util

    if importlib.util.find_spec("petsc4py") is not None:
        pytest.skip("petsc4py installed; the guard needs its absence")

    vecs = [np.zeros(3, dtype=complex)]

    from linstabpy.operators.harmonic_blocks import write_resolvent4py_vecs

    with pytest.raises(ImportError, match="petsc4py not found"):
        write_resolvent4py_vecs(vecs, str(tmp_path))


@pytest.mark.skipif(
    importlib.util.find_spec("petsc4py") is None,
    reason="petsc4py absent; the first guard fires before this one",
)
def test_write_resolvent4py_coo_resolvent4py_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test write_resolvent4py_coo raises ImportError when resolvent4py missing.

    petsc4py is present but resolvent4py is poisoned in sys.modules.
    """
    import sys

    # Poison resolvent4py in module cache to force ImportError.
    monkeypatch.setitem(sys.modules, "resolvent4py", None)

    # Create minimal HarmonicBlocks with one identity block.
    A_list = [sp.csr_matrix(np.eye(3))]
    blocks = HarmonicBlocks(
        Ahat=A_list,
        N_h=0,
        N_t=2,
    )

    from linstabpy.operators.harmonic_blocks import write_resolvent4py_coo

    with pytest.raises(ImportError, match="resolvent4py not found"):
        write_resolvent4py_coo(blocks, str(tmp_path))


@pytest.mark.skipif(
    importlib.util.find_spec("petsc4py") is None,
    reason="petsc4py absent; the first guard fires before this one",
)
def test_write_resolvent4py_vecs_resolvent4py_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test write_resolvent4py_vecs raises ImportError when resolvent4py missing.

    petsc4py is present but resolvent4py is poisoned in sys.modules.
    """
    import sys

    # Poison resolvent4py in module cache to force ImportError.
    monkeypatch.setitem(sys.modules, "resolvent4py", None)

    vecs = [np.zeros(3, dtype=complex)]

    from linstabpy.operators.harmonic_blocks import write_resolvent4py_vecs

    with pytest.raises(ImportError, match="resolvent4py not found"):
        write_resolvent4py_vecs(vecs, str(tmp_path))


def test_harmonic_blocks_dissipation_needs_q_diss(small_mesh_2d, steady_pbf) -> None:
    """A positive epsilon on a mesh without Q_diss is refused, not skipped."""
    assert small_mesh_2d.Q_diss_x is None
    with pytest.raises(ValueError, match="carries no Q_diss"):
        assemble_harmonic_blocks(small_mesh_2d, steady_pbf, N_h=0, N_t=4, dissipation_epsilon=0.1)


def test_harmonic_blocks_match_full_fft() -> None:
    """The per-instant Fourier sum equals np.fft.fft of the stacked operators."""
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system

    # 8 x 8 points (the fd_order=4 SBP minimum) keep the dense stack at 8 x 320 x 320.
    small_mesh_2d = Mesh.create(xrange=(0.0, 2.0), yrange=(0.0, 1.0), nx=8, ny=8, fd_order=4)

    ny, nx = small_mesh_2d.Y.shape[0], small_mesh_2d.X.shape[1]
    qhat = np.zeros((3, 5, ny, nx), dtype=complex)
    qhat[0, 0], qhat[0, 1], qhat[0, 4] = 1.0, 1.0, 1.0
    yy = small_mesh_2d.Y
    qhat[1, 1] = 0.05 * (1.0 + 0.5j) * yy
    qhat[2, 2] = 0.02j * yy**2
    pbf = PeriodicBaseFlow(
        Qhat=qhat,
        T_period=2.0,
        x=small_mesh_2d.X[0, :],
        y=small_mesh_2d.Y[:, 0],
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
    )
    N_h, N_t, kx = 2, 8, 0.7
    blocks = assemble_harmonic_blocks(small_mesh_2d, pbf, N_h=N_h, N_t=N_t, kx=kx)

    dense = np.stack(
        [
            build_lnse_system(small_mesh_2d, pbf.reconstruct(t), kx=kx)[0].toarray()
            for t in pbf.sample_times(N_t)
        ]
    )
    ref = np.fft.fft(dense, axis=0) / N_t
    scale = np.max(np.abs(ref[0]))
    for k in range(-N_h, N_h + 1):
        err = np.max(np.abs(blocks.Ahat[k + N_h].toarray() - ref[k % N_t]))
        assert err <= 1e-12 * scale, f"k={k}: max error {err:.2e} against fft"
    tail = np.linalg.norm(ref[N_h + 1]) / np.linalg.norm(ref[0])
    assert blocks.tail_ratio == pytest.approx(tail, rel=1e-10)
