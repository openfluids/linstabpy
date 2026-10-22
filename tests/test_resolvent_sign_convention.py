# SPDX-License-Identifier: GPL-3.0-only
"""The resolvent matrix solves the time-domain problem it claims to solve.

Take ``dq/dt = (L - gamma I) q + f exp(-i*omega*t)``.
Its late-time response is ``q(t) = exp(-i*omega*t) H f``.
Here ``H = [(gamma - i*omega) I - L]^-1``.
A matrix exponential of the augmented system integrates the ODE.
That reference uses no resolvent code.
``docs/resolvent_sign_convention.md`` gives the derivation.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linstabpy.config import ConfigValidationError, validate_config
from linstabpy.config.typed import Config
from linstabpy.linalg.shifted import shifted_operator


def _non_normal_stable_operator() -> sp.csc_matrix:
    # Upper-triangular, so the eigenvalues are the diagonal.  Real parts <= -1.2 and
    # imaginary parts of both signs and unequal size make the spectrum asymmetric in omega.
    rng = np.random.default_rng(11)
    diag = np.array([-1.2 + 2.0j, -1.5 - 0.7j, -2.0 + 3.1j, -2.4 - 4.0j, -3.0 + 0.4j, -3.5 - 1.1j])
    upper = np.triu(rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6)), 1) * 1.5
    return sp.csc_matrix(np.diag(diag) + upper)


def _time_domain_response(
    L: sp.csc_matrix, f: np.ndarray, omega: float, gamma: float
) -> np.ndarray:
    """Integrate the ODE to ``T`` and return ``q(T) exp(i*omega*T)``.

    Returns
    -------
    np.ndarray
        The complex amplitude of the late-time response.
    """
    n = L.shape[0]
    aug = np.zeros((n + 1, n + 1), dtype=complex)
    aug[:n, :n] = L.toarray() - gamma * np.eye(n)
    aug[:n, n] = f
    aug[n, n] = -1j * omega  # s' = -i*omega*s, s(0) = 1, so s(t) = exp(-i*omega*t)
    horizon = 60.0  # slowest decay rate is >= 1.2: transient below exp(-72)
    state = scipy.linalg.expm(aug * horizon) @ np.r_[np.zeros(n, dtype=complex), 1.0]
    return state[:n] * np.exp(1j * omega * horizon)


@pytest.mark.parametrize("gamma", [0.0, 0.4])
@pytest.mark.parametrize("omega", [-3.0, 2.0, 3.1])
def test_default_shifted_operator_matches_time_integration(omega: float, gamma: float) -> None:
    L = _non_normal_stable_operator()
    f = np.random.default_rng(5).standard_normal(6) + 1j * np.random.default_rng(6).standard_normal(
        6
    )

    q_time = _time_domain_response(L, f, omega, gamma)
    q_matrix = spla.spsolve(shifted_operator(L, omega, gamma), f)

    assert np.linalg.norm(q_matrix - q_time) / np.linalg.norm(q_time) < 1e-10


def test_matlab_parity_does_not_match_time_integration() -> None:
    # The old LinStab2D matrix is -H(-omega): it fails the time-domain check.
    L = _non_normal_stable_operator()
    f = np.ones(6, dtype=complex)
    omega, gamma = 3.1, 0.0

    q_time = _time_domain_response(L, f, omega, gamma)
    q_old = spla.spsolve(shifted_operator(L, omega, gamma, matlab_parity=True), f)

    assert np.linalg.norm(q_old - q_time) / np.linalg.norm(q_time) > 0.1
    # What it does compute: -H(-omega).
    q_mirror = -spla.spsolve(shifted_operator(L, -omega, gamma), f)
    np.testing.assert_allclose(q_old, q_mirror, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("matlab_parity", [False, True])
def test_adjoint_solve_follows_the_assembled_matrix(*, matlab_parity: bool) -> None:
    pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
    pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
    factorize = pytest.importorskip("linstabpy.linalg.factorize").factorize

    L = _non_normal_stable_operator()
    K = shifted_operator(L, 1.7, 0.2, matlab_parity=matlab_parity)
    rhs = np.arange(1, 7) + 0.5j * np.arange(6)
    fac = factorize(K, env_prefix="LINSTABPY_RESOLVENT")
    try:
        x_adj = fac.solve_adjoint(rhs)
        x_fwd = fac.solve(rhs)
    finally:
        fac.destroy()

    np.testing.assert_allclose(K.toarray().conj().T @ x_adj, rhs, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(K @ x_fwd, rhs, rtol=1e-10, atol=1e-10)


def test_config_default_is_physical_and_key_is_read_from_solver_resolvent() -> None:
    assert Config.load({}).resolvent.sign_convention == "physical"
    assert not Config.load({}).resolvent.matlab_parity
    cfg = {"solver": {"resolvent_sign_convention": "matlab_parity"}}
    assert Config.load(cfg).resolvent.matlab_parity
    # A bare solver mapping is accepted as well.
    assert Config.load(cfg["solver"]).resolvent.matlab_parity


def test_config_rejects_unknown_sign_convention() -> None:
    bad = {"solver": {"resolvent_sign_convention": "mirror"}}
    with pytest.raises(ValueError, match="sign_convention"):
        Config.load(bad)
    with pytest.raises(ConfigValidationError, match="sign_convention"):
        validate_config(bad)
    validate_config({"solver": {"resolvent_sign_convention": "physical"}})
    validate_config({"solver": {"resolvent_sign_convention": "matlab_parity"}})


def test_sign_key_does_not_shadow_the_linsolve_fallback() -> None:
    # A solver.resolvent block would replace solver.linsolve for the resolvent solve,
    # so the key lives beside it.  The resolvent linear solve keeps reading linsolve.
    base = {"solver": {"linsolve": {"type": "iterative", "ksp_rtol": 1e-6}}}
    flagged = {"solver": {**base["solver"], "resolvent_sign_convention": "matlab_parity"}}
    assert Config.load(base).resolvent.linear.ksp_rtol == pytest.approx(1e-6)
    assert Config.load(flagged).resolvent.linear == Config.load(base).resolvent.linear
