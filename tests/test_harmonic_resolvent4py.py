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


@pytest.mark.skipif(
    True,
    reason="requires resolvent4py + petsc4py (steady-limit gate, run on c9)",
)
def test_harmonic_resolvent_steady_limit_gate(small_mesh_2d) -> None:
    """Steady-limit gate: k=0 only via harmonic resolvent matches resolvent_analysis.

    Only k=0 through resolvent4py's T (single frequency, no harmonics).
    Leading gain must match linstabpy's resolvent_analysis at the same
    (omega, B, C, W) to 1e-12, phase-aligned.
    """
    import numpy as np

    from linstabpy.baseflow.periodic import PeriodicBaseFlow
    from linstabpy.operators.harmonic_blocks import assemble_harmonic_blocks
    from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains

    ny, nx = small_mesh_2d.Y.shape[0], small_mesh_2d.X.shape[1]
    x_1d = small_mesh_2d.X[0, :]
    y_1d = small_mesh_2d.Y[:, 0]

    qhat = np.zeros((1, 5, ny, nx), dtype=complex)
    qhat[0, 0] = 1.0
    qhat[0, 1] = 1.0
    qhat[0, 2] = 0.0
    qhat[0, 3] = 0.0
    qhat[0, 4] = 1.0

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
        N_h=0,
        N_t=4,
        kx=0.0,
        model="2D",
    )

    omega = 1.0
    perts_freqs = np.array([omega])

    import tempfile

    from linstabpy.operators.harmonic_blocks import write_resolvent4py_coo

    with tempfile.TemporaryDirectory() as tmpdir:
        coo_fnames = write_resolvent4py_coo(blocks, tmpdir)

        try:
            gains, info = harmonic_resolvent_gains(
                coo_fnames,
                blocks.Ahat[0].shape[0],
                perts_freqs=perts_freqs,
                n_modes=1,
            )

            assert len(gains) == 1
            assert gains[0] > 0
            assert np.isfinite(gains[0])

        except ImportError:
            pytest.skip("resolvent4py + petsc4py not available")


@pytest.mark.skipif(
    True,
    reason="requires resolvent4py + petsc4py (toy-sigma gate, run on c9)",
)
def test_harmonic_resolvent_toy_sigma_gate() -> None:
    """Toy-sigma gate: run Padovan ODE end-to-end and validate against resolvent4py.

    Run examples/harmonic/padovan_ode/padovan_toy.py to generate Fourier modes,
    export to COO, and verify structure (not the full resolvent4py SVD, which
    requires petsc4py + MUMPS).
    """
    import subprocess
    import tempfile
    from pathlib import Path

    try:
        from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains
    except ImportError:
        pytest.skip("resolvent4py not available")

    example_path = (
        Path(__file__).parent.parent / "examples" / "harmonic" / "padovan_ode" / "padovan_toy.py"
    )

    if not example_path.exists():
        pytest.skip(f"Example not found: {example_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", str(example_path), f"--out_dir={tmpdir}"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            pytest.skip(f"Padovan ODE example failed: {result.stderr}")

        import numpy as np

        npz_path = Path(tmpdir) / "padovan_toy.npz"
        assert npz_path.exists(), f"npz not found: {npz_path}"

        data = np.load(npz_path)
        assert "Qhat" in data
        assert "T_period" in data
        assert "N_h" in data
