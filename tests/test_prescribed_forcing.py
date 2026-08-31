# SPDX-License-Identifier: GPL-3.0-only
"""Tests for the 1.1 prescribed-forcing resolvent path.

Some tests exercise the real solve, so they need petsc4py and mpi4py.
On a box without them they announce a skip. Other tests never touch a
solver. Input validation raises before ``factorize`` runs, and the npz
loader tests are pure file I/O.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.resolvent.prescribed import prescribed_forcing_response
from linstabpy.resolvent.prescribed_io import PrescribedForcing, load_prescribed_forcing

_HAS_SOLVER_STACK = (
    importlib.util.find_spec("mpi4py") is not None
    and importlib.util.find_spec("petsc4py") is not None
)

# analysis.py imports petsc4py/slepc4py unconditionally at module scope, so
# this import must stay conditional: a bare top-level import would break
# collection of every OTHER test in this file on a box without PETSc.
try:
    from linstabpy.resolvent.analysis import resolvent_analysis
except ImportError:
    resolvent_analysis = None

_requires_solver = pytest.mark.skipif(
    not _HAS_SOLVER_STACK, reason="SKIPPED: requires petsc4py/mpi4py"
)


def _seeded_operator(n: int, seed: int) -> sp.csc_matrix:
    """Build a small, well-conditioned complex sparse operator.

    Returns
    -------
    scipy.sparse.csc_matrix
        A diagonally dominant complex matrix, ``n`` by ``n``.
    """
    rng = np.random.default_rng(seed)
    dense = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.05
    diag = 4.0 + rng.random(n) + 1j * rng.standard_normal(n)
    np.fill_diagonal(dense, diag)
    return sp.csc_matrix(dense)


def _dense_shifted(operator: sp.csc_matrix, omega: float, gamma: float) -> np.ndarray:
    """Return the dense ``L0 - (gamma + i*omega) I`` reference matrix.

    Returns
    -------
    numpy.ndarray
        The dense shifted operator.
    """
    n = operator.shape[0]
    return operator.toarray() - (gamma + 1j * omega) * np.eye(n)


def _dense_leading_svd(
    operator: sp.csc_matrix,
    omega: float,
    gamma: float,
    b_diag: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return the leading singular value and vector of ``inv(L_sigma) @ B``.

    B is diagonal (``b_diag``), C is identity: this is the independent,
    non-PETSc reference gate 2 checks the prescribed path against.

    Returns
    -------
    tuple
        ``(sigma1, v1)``.
    """
    dense_shifted = _dense_shifted(operator, omega, gamma)
    h_dense = np.linalg.solve(dense_shifted, np.diag(b_diag))
    _, sigma, vh = np.linalg.svd(h_dense)
    return float(sigma[0]), vh[0].conj()


@_requires_solver
@pytest.mark.parametrize("gamma", [0.0, 0.3])
def test_prescribed_forcing_matches_dense_solve(gamma: float) -> None:
    """Gate 1: prescribed q_hat and gain match a dense solve to 1e-12."""
    n = 12
    omega = 2.5
    L0 = _seeded_operator(n, seed=1)
    rng = np.random.default_rng(7)
    f_hat = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    w_diag = 0.5 + rng.random(n)

    q_hat, gain, info = prescribed_forcing_response(
        L0, omega, w_diag, f_hat, gamma=gamma, solver_cfg={"resolvent": {"eigen": {}}}
    )

    q_dense = np.linalg.solve(_dense_shifted(L0, omega, gamma), f_hat)
    np.testing.assert_allclose(q_hat, q_dense, atol=1e-12, rtol=1e-12)

    q_w_norm = float(np.sqrt(np.sum(w_diag * np.abs(q_dense) ** 2)))
    f_w_norm = float(np.sqrt(np.sum(w_diag * np.abs(f_hat) ** 2)))
    assert gain == pytest.approx(q_w_norm / f_w_norm, abs=1e-12, rel=1e-12)
    assert info["forcing_norm"] == "chu"
    assert cast("float", info["rho_fwd"]) < 1e-8


@_requires_solver
@pytest.mark.parametrize("forcing_norm", ["chu", "l2"])
def test_prescribed_forcing_bounded_by_leading_svd_gain(forcing_norm: str) -> None:
    """Gate 2a: a random forcing never exceeds the leading SVD gain."""
    omega, gamma = 1.7, 0.0
    L0 = _seeded_operator(10, seed=2)
    b_diag = np.array([1.0, 1.0, 0.0, 1.3, 1.0, 0.7, 1.0, 1.0, 0.0, 1.0])
    # Identity Chu weight: the dense SVD reference above is exact for both
    # forcing_norm branches (with W = I, the W-norm is the plain 2-norm).
    w_diag = np.ones(10)
    sigma1, _ = _dense_leading_svd(L0, omega, gamma, b_diag)

    rng = np.random.default_rng(11)
    f_random = rng.standard_normal(10) + 1j * rng.standard_normal(10)
    f_random /= np.linalg.norm(f_random)

    _, gain_random, _ = prescribed_forcing_response(
        L0,
        omega,
        w_diag,
        f_random,
        gamma=gamma,
        B_mat=sp.diags(b_diag),
        forcing_norm=forcing_norm,
        solver_cfg={"resolvent": {"eigen": {}}},
    )
    assert gain_random <= sigma1 * (1.0 + 1e-10)


@_requires_solver
@pytest.mark.parametrize("forcing_norm", ["chu", "l2"])
def test_prescribed_forcing_roundtrips_leading_svd_vector(forcing_norm: str) -> None:
    """Gate 2b: the SVD's own optimal forcing returns exactly its gain.

    This round-trip test catches energy-losing bugs in the pipeline that
    the inequality test (bounded_by_leading_svd) passes through. A wrong
    mask, wrong norm, or a factor-of-2 error will make gain < sigma1, and
    only this test—which recovers the SVD's own optimal vector—detects
    the loss.
    """
    omega, gamma = 1.7, 0.0
    L0 = _seeded_operator(10, seed=2)
    b_diag = np.array([1.0, 1.0, 0.0, 1.3, 1.0, 0.7, 1.0, 1.0, 0.0, 1.0])
    w_diag = np.ones(10)
    sigma1, v1 = _dense_leading_svd(L0, omega, gamma, b_diag)

    _, gain_v1, _ = prescribed_forcing_response(
        L0,
        omega,
        w_diag,
        v1,
        gamma=gamma,
        B_mat=sp.diags(b_diag),
        forcing_norm=forcing_norm,
        solver_cfg={"resolvent": {"eigen": {}}},
    )
    assert gain_v1 == pytest.approx(sigma1, abs=1e-8, rel=1e-8)


def test_prescribed_forcing_ndof_mismatch_raises() -> None:
    """Gate 3: an f_hat whose length differs from ndof raises early."""
    L0 = _seeded_operator(6, seed=3)
    w_diag = np.ones(6)
    f_hat = np.ones(5, dtype=complex)
    with pytest.raises(ValueError, match="does not match operator ndof"):
        prescribed_forcing_response(L0, 1.0, w_diag, f_hat)


def test_prescribed_forcing_non_finite_f_hat_raises() -> None:
    """Gate 3: a non-finite f_hat raises before any solve."""
    L0 = _seeded_operator(6, seed=3)
    w_diag = np.ones(6)
    f_hat = np.ones(6, dtype=complex)
    f_hat[2] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        prescribed_forcing_response(L0, 1.0, w_diag, f_hat)


def test_prescribed_forcing_w_diag_ndof_mismatch_raises() -> None:
    """A W_diag whose length differs from ndof raises before any solve."""
    L0 = _seeded_operator(6, seed=3)
    f_hat = np.ones(6, dtype=complex)
    with pytest.raises(ValueError, match="does not match operator ndof"):
        prescribed_forcing_response(L0, 1.0, np.ones(4), f_hat)


def test_prescribed_forcing_invalid_forcing_norm_raises() -> None:
    """An unsupported forcing_norm raises before any solve."""
    L0 = _seeded_operator(6, seed=3)
    w_diag = np.ones(6)
    f_hat = np.ones(6, dtype=complex)
    with pytest.raises(ValueError, match="forcing_norm"):
        prescribed_forcing_response(L0, 1.0, w_diag, f_hat, forcing_norm="bogus")


@_requires_solver
def test_prescribed_forcing_matches_resolvent_analysis_leading_mode() -> None:
    """The SVD path's own leading forcing round-trips through prescribed()."""
    assert resolvent_analysis is not None  # guaranteed by _requires_solver

    omega, gamma = 1.3, 0.0
    L0 = _seeded_operator(10, seed=5)
    w_diag = np.ones(10)
    b_diag = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.2, 1.0, 1.0, 0.0, 1.0])
    B_mat = sp.diags(b_diag)

    # residuals=False, return_info=False (both defaults) is the historical
    # three-tuple return; the other three shapes need those flags set.
    gains, responses, forcings = cast(
        "tuple[np.ndarray, np.ndarray, np.ndarray]",
        resolvent_analysis(
            L0,
            omega=omega,
            n_modes=1,
            W_diag=w_diag,
            B_mat=B_mat,
            gamma=gamma,
            solver_cfg={"resolvent": {"eigen": {"eps_tol": 1e-12}}},
        ),
    )

    q_hat, gain, _ = prescribed_forcing_response(
        L0,
        omega,
        w_diag,
        forcings[:, 0],
        gamma=gamma,
        B_mat=B_mat,
        solver_cfg={"resolvent": {"eigen": {}}},
    )

    assert gain == pytest.approx(float(gains[0]), abs=1e-8, rel=1e-8)
    np.testing.assert_allclose(q_hat, gains[0] * responses[:, 0], atol=1e-8, rtol=1e-8)


# --------------------------------------------------------------------------
# prescribed_io.py -- no solver stack needed, pure npz round-trip.
# --------------------------------------------------------------------------


def test_load_prescribed_forcing_round_trips(tmp_path: Path) -> None:
    """A well-formed npz loads into a PrescribedForcing with matching fields."""
    path = tmp_path / "point.npz"
    f = (np.arange(5) + 1j * np.arange(5)).astype(complex)
    np.savez(path, f=f, beta=1.5, omega=0.2, gamma=0.1, nx=5, commit="abc123")

    loaded = load_prescribed_forcing(path, expected_ndof=5)

    assert isinstance(loaded, PrescribedForcing)
    np.testing.assert_array_equal(loaded.f, f)
    assert loaded.beta == pytest.approx(1.5)
    assert loaded.omega == pytest.approx(0.2)
    assert loaded.gamma == pytest.approx(0.1)
    assert loaded.extras["nx"] == 5
    assert loaded.extras["commit"] == "abc123"


def test_load_prescribed_forcing_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    np.savez(path, f=np.ones(3, dtype=complex), beta=1.0)
    with pytest.raises(ValueError, match="missing required key"):
        load_prescribed_forcing(path, expected_ndof=3)


def test_load_prescribed_forcing_ndof_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    np.savez(path, f=np.ones(3, dtype=complex), beta=1.0, omega=0.5)
    with pytest.raises(ValueError, match="does not match expected ndof"):
        load_prescribed_forcing(path, expected_ndof=4)


def test_load_prescribed_forcing_non_finite_raises(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    f = np.ones(3, dtype=complex)
    f[1] = np.inf
    np.savez(path, f=f, beta=1.0, omega=0.5)
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_prescribed_forcing(path, expected_ndof=3)


def test_load_prescribed_forcing_zero_bin_rejected_by_default(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    np.savez(path, f=np.ones(3, dtype=complex), beta=0.0, omega=0.0)
    with pytest.raises(ValueError, match="mean-flow bin"):
        load_prescribed_forcing(path, expected_ndof=3)


def test_load_prescribed_forcing_zero_bin_accepted_when_flagged(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    np.savez(path, f=np.ones(3, dtype=complex), beta=0.0, omega=0.0)
    loaded = load_prescribed_forcing(path, expected_ndof=3, allow_zero_bin=True)
    assert (loaded.beta, loaded.omega) == (0.0, 0.0)


def test_load_prescribed_forcing_gamma_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    np.savez(path, f=np.ones(3, dtype=complex), beta=1.0, omega=0.5, gamma=0.2)
    with pytest.raises(ValueError, match="does not match config gamma"):
        load_prescribed_forcing(path, expected_ndof=3, config_gamma=0.5)


def test_load_prescribed_forcing_gamma_match_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    np.savez(path, f=np.ones(3, dtype=complex), beta=1.0, omega=0.5, gamma=0.2)
    loaded = load_prescribed_forcing(path, expected_ndof=3, config_gamma=0.2)
    assert loaded.gamma == pytest.approx(0.2)
