# SPDX-License-Identifier: GPL-3.0-only
"""Tests for harmonic resolvent solver via resolvent4py."""

from __future__ import annotations

import sys
from importlib.util import find_spec

import numpy as np
import pytest

# Conditional gate: skip @_requires_stack tests where the stack is missing.
# petsc4py and slepc4py are only on c9; resolvent4py is a pip-installed extra.
_HAS_STACK = (
    find_spec("petsc4py") is not None
    and find_spec("slepc4py") is not None
    and find_spec("resolvent4py") is not None
    and find_spec("mpi4py") is not None
)
_requires_stack = pytest.mark.skipif(
    not _HAS_STACK,
    reason="SKIPPED: requires petsc4py/slepc4py/mpi4py/resolvent4py (c9)",
)


def test_harmonic_resolvent_import() -> None:
    """Test that the module imports even when resolvent4py is absent."""
    # This should not raise; imports are lazy
    from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains

    assert callable(harmonic_resolvent_gains)


def test_harmonic_resolvent_missing_resolvent4py(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that calling without resolvent4py raises ImportError with clear message."""
    # Mock resolvent4py as missing by setting it to None in sys.modules.
    # A None entry forces ImportError on any import attempt.
    monkeypatch.setitem(sys.modules, "resolvent4py", None)

    from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains

    coo_fnames = [("rows_00.dat", "cols_00.dat", "vals_00.dat")]
    perts_freqs = np.array([-1.0, 0.0, 1.0])

    with pytest.raises(ImportError, match="resolvent4py"):
        harmonic_resolvent_gains(
            coo_fnames,
            10,
            perts_freqs=perts_freqs,
        )


@pytest.fixture
def small_mesh_2d():
    """20x30 mesh on [0,2]x[0,1], order 4.

    Returns
    -------
    Mesh
        Rectilinear test mesh with fourth-order stencils.
    """
    from linstabpy.mesh import Mesh

    return Mesh.create(
        xrange=(0.0, 2.0),
        yrange=(0.0, 1.0),
        nx=20,
        ny=30,
        fd_order=4,
    )


@_requires_stack
def test_harmonic_resolvent_coo_round_trip() -> None:
    """COO write/read round-trip: the reader rebuilds the exact Toeplitz operator.

    Write seeded random 3x3 blocks for k=-1..1 and read them back
    through resolvent4py. Compare the dense operator entry by entry
    against the block-Toeplitz reference M_{jk} = Ahat_{j-k}, zero
    when |j-k| > N_h. An entry mismatch falsifies the writer, the
    reader, or the file order.
    """
    import tempfile

    import scipy.sparse as sp

    from linstabpy.operators.harmonic_blocks import (
        HarmonicBlocks,
        write_resolvent4py_coo,
    )

    rng = np.random.default_rng(7)
    n = 3
    N_h = 1
    ahat = [
        sp.csr_matrix(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        for _ in range(2 * N_h + 1)
    ]
    blocks = HarmonicBlocks(Ahat=ahat, N_h=N_h, N_t=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        coo_fnames = write_resolvent4py_coo(blocks, tmpdir)

        from resolvent4py import compute_local_size, read_harmonic_balanced_matrix

        # Smallest ladder that exercises every stored block: N_p = N_h.
        N_p = N_h
        N_full = (2 * N_p + 1) * n
        nl = compute_local_size(n)
        Nl = compute_local_size(N_full)

        M = read_harmonic_balanced_matrix(
            coo_fnames,
            real_bflow=False,
            block_sizes=((nl, n), (nl, n)),
            full_sizes=((Nl, N_full), (Nl, N_full)),
        )

        idx = list(range(N_full))
        dense = np.array(M.getValues(idx, idx))

        reference = np.zeros((N_full, N_full), dtype=complex)
        for j in range(-N_p, N_p + 1):
            for k in range(-N_p, N_p + 1):
                if abs(j - k) <= N_h:
                    jb = j + N_p
                    kb = k + N_p
                    reference[jb * n : (jb + 1) * n, kb * n : (kb + 1) * n] = ahat[
                        (j - k) + N_h
                    ].toarray()

        err = np.max(np.abs(dense - reference))
        assert err < 1e-13, f"Round-trip mismatch: max entry error {err:.2e}"


@_requires_stack
def test_harmonic_resolvent_steady_limit_gate(small_mesh_2d) -> None:
    """Steady-limit gate: k=0 only via harmonic resolvent matches resolvent_analysis.

    N_h=0 blocks (single frequency only), eps_reg=0 (unperturbed), and
    leading gain must match the classical resolvent_analysis to 1e-12.
    """
    import tempfile

    from linstabpy.baseflow.periodic import PeriodicBaseFlow
    from linstabpy.operators.harmonic_blocks import (
        assemble_harmonic_blocks,
        write_resolvent4py_coo,
    )
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

    with tempfile.TemporaryDirectory() as tmpdir:
        coo_fnames = write_resolvent4py_coo(blocks, tmpdir)

        gains, _info = harmonic_resolvent_gains(
            coo_fnames,
            blocks.Ahat[0].shape[0],
            perts_freqs=perts_freqs,
            n_modes=1,
            eps_reg=0.0,
        )

        # Reference: classical resolvent_analysis on the identical matrix.
        from linstabpy.resolvent.analysis import resolvent_analysis

        L0 = blocks.Ahat[0].tocsc()
        n_dof = L0.shape[0]

        # W_diag of ones is the plain L2 norm the harmonic wrapper uses.
        gains_ref, *_ = resolvent_analysis(
            L0,
            omega,
            n_modes=1,
            W_diag=np.ones(n_dof),
            solver_cfg={"resolvent": {"eigen": {"eps_tol": 1e-12}}},
        )

        # Compare leading gains
        assert len(gains) >= 1, "Harmonic resolvent returned no gains"
        assert len(gains_ref) >= 1, "resolvent_analysis returned no gains"

        rel_err = abs(gains[0] - gains_ref[0]) / (abs(gains_ref[0]) + 1e-16)
        detail = f"harmonic {gains[0]:.6e}, classical {gains_ref[0]:.6e}, rel_err {rel_err:.2e}"
        assert rel_err < 1e-12, f"Steady-limit gain mismatch: {detail}"


@_requires_stack
def test_harmonic_resolvent_toy_dense_reference() -> None:
    """Dense-reference gate: Padovan ODE against an independent numpy SVD.

    Integrate the Padovan ODE and compute FFT harmonics of Q and A.
    Build T densely in numpy with blocks T_{jk} = Ahat_{j-k}.
    Compare leading singular values against harmonic_resolvent_gains.
    """
    import importlib.util
    import tempfile
    from pathlib import Path

    # Import Padovan dynamics
    example_path = (
        Path(__file__).parent.parent / "examples" / "harmonic" / "padovan_ode" / "padovan_toy.py"
    )
    if not example_path.exists():
        pytest.skip(f"Padovan example not found: {example_path}")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("padovan_toy", example_path)
    if spec is None or spec.loader is None:
        pytest.skip("Could not load Padovan example")

    padovan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(padovan)

    # Generate harmonics in memory (avoid file I/O overhead)
    from scipy.integrate import solve_ivp

    params = (1 / 5, 1 / 5, 1 / 5)
    mu, alpha, beta = params
    omega0_theory = np.sqrt(1 - beta**2 * mu / alpha)
    T = 2 * np.pi / omega0_theory

    dt = T / 300
    t_final = 200 * T
    t_eval = np.arange(0, t_final + dt, dt)

    sol = solve_ivp(
        padovan.evaluate_dynamics,
        [0, t_final],
        0.1 * np.ones(3),
        method="RK45",
        t_eval=t_eval,
        args=(params,),
        rtol=1e-12,
        atol=1e-12,
        dense_output=False,
    )

    idx0 = np.argmin(np.abs(sol.t - 198 * T))
    idxf = np.argmin(np.abs(sol.t - 199 * T))
    t_period = sol.t[idx0:idxf] - sol.t[idx0]
    Q = sol.y[:, idx0:idxf]

    N_h = 4
    Qhat = (1.0 / len(t_period)) * np.fft.rfft(Q, axis=-1)[:, : (N_h + 1)]

    As = np.array(
        [
            padovan.evaluate_jacobian(t_period[i], Q[:, i], params).reshape(-1, 1)
            for i in range(len(t_period))
        ]
    )
    As = np.concatenate(As, axis=-1)
    Ashat = (1.0 / len(t_period)) * np.fft.rfft(As, axis=-1)[:, : (N_h + 1)]

    omega0 = 2 * np.pi / (t_period[-1] - t_period[0])

    # Export COO and dQ
    import scipy.sparse as sp

    from linstabpy.operators.harmonic_blocks import (
        HarmonicBlocks,
        write_resolvent4py_coo,
        write_resolvent4py_vecs,
    )
    from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains

    ahat_csr_list = [sp.csr_matrix(Ashat[:, k].reshape((3, 3), order="C")) for k in range(N_h + 1)]
    ahat_two_sided = [
        ahat_csr_list[abs(k)].conj() if k < 0 else ahat_csr_list[k] for k in range(-N_h, N_h + 1)
    ]

    blocks = HarmonicBlocks(Ahat=ahat_two_sided, N_h=N_h, N_t=len(t_period))

    with tempfile.TemporaryDirectory() as tmpdir:
        coo_fnames = write_resolvent4py_coo(blocks, tmpdir)

        # Dense reference: build T in numpy.
        N_p = N_h + 3
        n = 3
        N_full = (2 * N_p + 1) * n

        T_dense = np.zeros((N_full, N_full), dtype=complex)
        for j in range(-N_p, N_p + 1):
            for k in range(-N_p, N_p + 1):
                jb = j + N_p
                kb = k + N_p
                if abs(j - k) <= N_h:
                    # Block (j, k) holds Ahat_{j-k}.
                    idx_diff = (j - k) + N_h
                    A_diff = ahat_two_sided[idx_diff].toarray()
                    T_dense[jb * n : (jb + 1) * n, kb * n : (kb + 1) * n] = A_diff
                # Diagonal: subtract i*omega_k (omega_k = k*omega0)
                if j == k:
                    omega_j = j * omega0
                    T_dense[jb * n : (jb + 1) * n, kb * n : (kb + 1) * n] -= (
                        1j * omega_j * np.eye(n)
                    )

        # Flip sign to -T (resolvent4py convention), then add the same
        # eps regularization the wrapper applies before it factors T.
        # Both sides must carry the identical eps for a 1e-8 comparison.
        eps_reg = 1e-7
        T_dense = -T_dense + eps_reg * np.eye(N_full)

        # dQ vectors: one length-n block per ladder harmonic k=-N_p..N_p.
        # dQ_k = i*k*omega0*Qhat_k. Qhat has no harmonics beyond N_h, so
        # blocks with |k| > N_h are zero. The limit cycle is real, so
        # dQ_{-k} = conj(dQ_k) = conj(i*k*omega0*Qhat_k).
        dq_vecs_dense = []
        for k in range(-N_p, N_p + 1):
            if k == 0 or abs(k) > N_h:
                dq_vecs_dense.append(np.zeros(n, dtype=complex))
            else:
                dq_k = 1j * abs(k) * omega0 * Qhat[:, abs(k)]
                if k < 0:
                    dq_k = np.conj(dq_k)
                dq_vecs_dense.append(dq_k)

        dq_all = np.concatenate(dq_vecs_dense)
        dq_all /= np.linalg.norm(dq_all)

        # Left phase direction w = T^{-H} dQ, normalized, as in the wrapper.
        w_all = np.linalg.solve(T_dense.conj().T, dq_all)
        w_all /= np.linalg.norm(w_all)

        # Toy convention (Padovan 2020): Pd = I - w w* on the forcing side
        # and Pr = I - dQ dQ* on the response side.
        Pd = np.eye(N_full) - np.outer(w_all, w_all.conj())
        Pr = np.eye(N_full) - np.outer(dq_all, dq_all.conj())

        T_inv = np.linalg.inv(T_dense)
        H_dense = Pr @ T_inv @ Pd

        sigma_ref = np.linalg.svd(H_dense, compute_uv=False)

        perts_freqs = np.arange(-N_p, N_p + 1) * omega0
        # resolvent4py seeds its PETSc RNG from numpy's global RNG
        # (randomized_svd.py: setSeed(np.random.randint(...))). Seed it here
        # so the randomized SVD is deterministic. With n_rand=20 the
        # truncation error of the near-degenerate pair (modes 2-3) sits at
        # the 1e-8 gate itself and run-to-run seeds crossed it; n_rand=30
        # (the solver default) keeps that nuisance error below the gate.
        np.random.seed(2024)
        gains, _info = harmonic_resolvent_gains(
            coo_fnames,
            n,
            perts_freqs=perts_freqs,
            n_modes=5,
            n_rand=30,
            eps_reg=eps_reg,
            dq_fnames=write_resolvent4py_vecs(dq_vecs_dense, tmpdir),
        )

        # Compare leading 3 gains
        for i in range(min(3, len(gains))):
            rel_err = abs(gains[i] - sigma_ref[i]) / (abs(sigma_ref[i]) + 1e-16)
            assert rel_err < 1e-8, (
                f"Toy SVD mismatch at mode {i}: "
                f"resolvent4py {gains[i]:.6e}, dense {sigma_ref[i]:.6e}, "
                f"rel_err {rel_err:.2e}"
            )
