# SPDX-License-Identifier: GPL-3.0-only
"""The harmonic resolvent uses exp(-i*omega*t), the convention of the classical resolvent.

The equations under test, with ``omega_j = omega + j*omega0``::

    (gamma - i*omega_j) q_j - sum_k Ahat_k q_{j-k} = f_j,
    A(t) = sum_k Ahat_k exp(-i*k*omega0*t).

resolvent4py assembles the same ladder with ``exp(+i*omega*t)``.
``_harmonic_generator`` maps one onto the other. Block ``p`` of its matrix,
counted from the centre of the ladder, is harmonic ``j = -p``.
"""

from __future__ import annotations

import tempfile
from importlib.util import find_spec

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.integrate import solve_ivp

_HAS_STACK = (
    find_spec("petsc4py") is not None
    and find_spec("slepc4py") is not None
    and find_spec("resolvent4py") is not None
    and find_spec("mpi4py") is not None
)
_requires_stack = pytest.mark.skipif(
    not _HAS_STACK,
    reason="SKIPPED: requires petsc4py/slepc4py/mpi4py/resolvent4py (complex PETSc build)",
)

N_DOF = 3
N_H = 1
N_P = 8
OMEGA0 = 1.3
OMEGA = 0.9


def _periodic_operator() -> tuple[list[np.ndarray], np.ndarray]:
    """Return ``[A_{-1}, A_0, A_1]`` (coefficients of exp(-i*k*omega0*t)) and a forcing.

    The blocks are complex and ``A_{-1}`` is not ``conj(A_1)``, so the
    operator is complex at every instant and a sign error in ``k`` or in
    ``omega`` changes the answer. The mean block is damped enough for the
    Floquet multipliers to be below 4e-3 (checked in the time-integration test).
    """
    rng = np.random.default_rng(3)

    def block() -> np.ndarray:
        return rng.standard_normal((N_DOF, N_DOF)) + 1j * rng.standard_normal((N_DOF, N_DOF))

    a0 = 0.6 * block() - 2.5 * np.eye(N_DOF)
    a1 = 0.4 * block()
    am1 = 0.4 * block()
    forcing = block()[:, 0]
    return [am1, a0, a1], forcing


def _generator_dense(blocks: list[np.ndarray], gamma: float) -> np.ndarray:
    """Return the dense matrix of ``_harmonic_generator`` with eps_reg = 0."""
    from linstabpy.operators.harmonic_blocks import HarmonicBlocks, write_resolvent4py_coo
    from linstabpy.solvers.harmonic_resolvent import _harmonic_generator

    hb = HarmonicBlocks(Ahat=[sp.csr_matrix(b) for b in blocks], N_h=N_H, N_t=4)
    freqs = OMEGA + np.arange(-N_P, N_P + 1) * OMEGA0
    with tempfile.TemporaryDirectory() as tmpdir:
        coo = write_resolvent4py_coo(hb, tmpdir)
        generator = _harmonic_generator(coo, N_DOF, freqs, gamma, 0.0)
        size = (2 * N_P + 1) * N_DOF
        idx = list(range(size))
        return np.array(generator.getValues(idx, idx))


def _reverse_blocks(matrix: np.ndarray) -> np.ndarray:
    """Reorder the block rows and columns of the ladder: block p <-> block -p."""
    nb = 2 * N_P + 1
    order = np.concatenate([np.arange(b * N_DOF, (b + 1) * N_DOF) for b in range(nb)[::-1]])
    return matrix[np.ix_(order, order)]


@_requires_stack
def test_generator_equals_dense_block_formula() -> None:
    """The assembled matrix equals the dense ladder written from the formula.

    Block ``(i, j)`` of the dense ladder holds ``(gamma - i*omega_j) delta_ij - Ahat_{i-j}``.
    The assembled matrix lists the harmonics in reverse order, so the
    comparison reverses the blocks of the dense ladder first.
    """
    blocks, _ = _periodic_operator()
    gamma = 0.2
    nb = 2 * N_P + 1
    dense = np.zeros((nb * N_DOF, nb * N_DOF), dtype=complex)
    for i in range(-N_P, N_P + 1):
        for j in range(-N_P, N_P + 1):
            ib, jb = (i + N_P) * N_DOF, (j + N_P) * N_DOF
            if abs(i - j) <= N_H:
                dense[ib : ib + N_DOF, jb : jb + N_DOF] -= blocks[i - j + N_H]
            if i == j:
                omega_j = OMEGA + j * OMEGA0
                dense[ib : ib + N_DOF, jb : jb + N_DOF] += (gamma - 1j * omega_j) * np.eye(N_DOF)

    assembled = _generator_dense(blocks, gamma)
    err = np.max(np.abs(assembled - _reverse_blocks(dense)))
    assert err < 1e-13, f"assembled ladder differs from the formula by {err:.2e}"


def _late_time_components(
    blocks: list[np.ndarray], forcing: np.ndarray, gamma: float
) -> dict[int, np.ndarray]:
    """Integrate ``dq/dt = (A(t) - gamma) q + f exp(-i*omega*t)`` and return ``q_j``.

    ``q_j`` is the Fourier component of the late-time response at ``omega + j*omega0``.
    One period of the late response, sampled uniformly, gives each component
    exactly when the signal is band limited.
    """
    am1, a0, a1 = blocks
    t_period = 2.0 * np.pi / OMEGA0
    eye = np.eye(N_DOF)

    def a_of_t(t: float) -> np.ndarray:
        return a0 + a1 * np.exp(-1j * OMEGA0 * t) + am1 * np.exp(1j * OMEGA0 * t)

    # The transient decays by the Floquet multiplier per period.
    flo = solve_ivp(
        lambda t, y: ((a_of_t(t) - gamma * eye) @ y.reshape(N_DOF, N_DOF)).ravel(),
        [0.0, t_period],
        np.eye(N_DOF, dtype=complex).ravel(),
        rtol=1e-12,
        atol=1e-14,
    )
    multipliers = np.abs(np.linalg.eigvals(flo.y[:, -1].reshape(N_DOF, N_DOF)))
    assert multipliers.max() < 1e-2, f"Floquet multipliers {multipliers}"

    n_periods = 12
    n_samples = 64
    t_end = n_periods * t_period
    t_eval = t_end - t_period + np.arange(n_samples) * t_period / n_samples
    sol = solve_ivp(
        lambda t, q: (a_of_t(t) - gamma * eye) @ q + forcing * np.exp(-1j * OMEGA * t),
        [0.0, t_end],
        np.zeros(N_DOF, dtype=complex),
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-14,
    )
    return {
        j: (sol.y * np.exp(1j * (OMEGA + j * OMEGA0) * sol.t)).mean(axis=1)
        for j in range(-N_H, N_H + 1)
    }


@_requires_stack
@pytest.mark.parametrize("gamma", [0.0, 0.2])
def test_ladder_response_matches_time_integration(gamma: float) -> None:
    """The harmonic solve reproduces the late-time response of the periodic ODE.

    The time integration is independent of every Fourier and sign convention
    in the code. With exp(+i*omega*t) used for ``A(t)`` or for the response
    the relative difference is 0.86 (measured with the unmapped
    resolvent4py ladder); with the mapping it is 9e-13 at gamma = 0.2.
    """
    blocks, forcing = _periodic_operator()
    expected = _late_time_components(blocks, forcing, gamma)

    generator = _generator_dense(blocks, gamma)
    size = generator.shape[0]
    rhs = np.zeros(size, dtype=complex)
    rhs[N_P * N_DOF : (N_P + 1) * N_DOF] = forcing  # forcing at omega_0 = omega
    x = np.linalg.solve(generator, rhs).reshape(2 * N_P + 1, N_DOF)

    # Block p (from the centre) is harmonic j = -p.
    got = {j: x[-j + N_P] for j in range(-N_H, N_H + 1)}
    scale = max(np.linalg.norm(expected[j]) for j in expected)
    err = max(np.linalg.norm(got[j] - expected[j]) for j in expected) / scale
    assert err < 1e-9, f"relative difference {err:.2e} against time integration"


def _fake_lnse_factory(coupling: np.ndarray):
    """Return a stand-in for ``build_lnse_system`` with ``L0 = B * (U - 1)``."""

    def build(_mesh, bf, **_kwargs):
        return sp.csr_matrix(coupling * (float(bf.U.ravel()[0]) - 1.0)), None, None

    return build


def _one_point_flow(qhat_u: complex):
    from linstabpy.baseflow.periodic import PeriodicBaseFlow

    qhat = np.zeros((2, 5, 1, 1), dtype=complex)
    qhat[0, 0] = qhat[0, 1] = qhat[0, 4] = 1.0
    qhat[1, 1] = qhat_u
    return PeriodicBaseFlow(
        Qhat=qhat,
        T_period=2.0,
        x=np.zeros(1),
        y=np.zeros(1),
        Re=1000.0,
        Ma=0.5,
        Pr=0.71,
        kappa=1.4,
        T_0=300.0,
    )


def test_reconstruct_uses_exp_minus() -> None:
    """``Q(t) = Qhat_0 + 2 Re(Qhat_1 exp(-i*omega0*t))`` for a complex ``Qhat_1``.

    With Qhat_1 = 0.1i the velocity is 1 + 0.2 sin(omega0*t), so it is 1.2 at
    a quarter period. The exp(+i*omega0*t) convention gives 0.8.
    """
    pbf = _one_point_flow(0.1j)
    assert pbf.reconstruct(0.25 * pbf.T_period).U[0, 0] == pytest.approx(1.2, abs=1e-14)


def test_blocks_are_coefficients_of_exp_minus(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Ahat_k`` is the coefficient of exp(-i*k*omega0*t) of a linear operator.

    ``L0(t) = B (U(t) - 1)`` with ``U - 1 = q exp(-i*omega0*t) + conj(q) exp(+i*omega0*t)``
    gives ``Ahat_1 = q B`` and ``Ahat_-1 = conj(q) B``, not the reverse.
    """
    from linstabpy.operators import assembly
    from linstabpy.operators.harmonic_blocks import assemble_harmonic_blocks

    coupling = np.array([[1.0 + 2.0j, 0.5], [0.0, -1.0j]])
    monkeypatch.setattr(assembly, "build_lnse_system", _fake_lnse_factory(coupling))
    q = 0.1 + 0.05j
    blocks = assemble_harmonic_blocks(None, _one_point_flow(q), N_h=1, N_t=8)

    np.testing.assert_allclose(blocks.Ahat[2].toarray(), q * coupling, atol=1e-14)
    np.testing.assert_allclose(blocks.Ahat[0].toarray(), np.conj(q) * coupling, atol=1e-14)
    np.testing.assert_allclose(blocks.Ahat[1].toarray(), np.zeros((2, 2)), atol=1e-14)
