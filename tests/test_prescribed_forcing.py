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


def _dense_shifted(
    operator: sp.csc_matrix, omega: float, gamma: float, *, matlab_parity: bool = False
) -> np.ndarray:
    """Return the dense ``(gamma - i*omega) I - L0`` reference matrix.

    With ``matlab_parity`` it is the LinStab2D matrix ``L0 - (gamma + i*omega) I``.

    Returns
    -------
    numpy.ndarray
        The dense shifted operator.
    """
    n = operator.shape[0]
    if matlab_parity:
        return operator.toarray() - (gamma + 1j * omega) * np.eye(n)
    return (gamma - 1j * omega) * np.eye(n) - operator.toarray()


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
def test_prescribed_forcing_follows_the_resolvent_sign_setting() -> None:
    """The matlab_parity setting gives the LinStab2D response, as in resolvent_analysis."""
    n, omega, gamma = 12, 2.5, 0.3
    L0 = _seeded_operator(n, seed=1)
    rng = np.random.default_rng(7)
    f_hat = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    cfg = {"resolvent": {"eigen": {}}, "resolvent_sign_convention": "matlab_parity"}

    q_hat, _, _ = prescribed_forcing_response(
        L0, omega, np.ones(n), f_hat, gamma=gamma, solver_cfg=cfg
    )

    q_dense = np.linalg.solve(_dense_shifted(L0, omega, gamma, matlab_parity=True), f_hat)
    np.testing.assert_allclose(q_hat, q_dense, atol=1e-12, rtol=1e-12)


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
    the inequality test (bounded_by_leading_svd) passes through. A mismatched
    mask, a mismatched norm, or a factor-of-2 error will make gain < sigma1, and
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


@_requires_solver
@pytest.mark.parametrize("forcing_norm", ["chu", "l2"])
def test_prescribed_forcing_roundtrips_resolvent_analysis_weighted(
    forcing_norm: str,
) -> None:
    """A non-identity W binds the forcing_norm switch between paths.

    When W_diag is non-uniform, the Chu-weighted and L2 gains differ.
    This test roundtrips both branches through resolvent_analysis to bind
    the forcing_norm parameter across both SVD and prescribed paths.
    """
    assert resolvent_analysis is not None  # guaranteed by _requires_solver

    omega, gamma = 1.3, 0.0
    L0 = _seeded_operator(10, seed=5)
    rng = np.random.default_rng(13)
    w_diag = 0.5 + rng.random(10)
    b_diag = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.2, 1.0, 1.0, 0.0, 1.0])
    B_mat = sp.diags(b_diag)

    gains, responses, forcings = cast(
        "tuple[np.ndarray, np.ndarray, np.ndarray]",
        resolvent_analysis(
            L0,
            omega=omega,
            n_modes=1,
            W_diag=w_diag,
            B_mat=B_mat,
            gamma=gamma,
            solver_cfg={"resolvent": {"eigen": {"eps_tol": 1e-12, "forcing_norm": forcing_norm}}},
        ),
    )

    q_hat, gain, _ = prescribed_forcing_response(
        L0,
        omega,
        w_diag,
        forcings[:, 0],
        gamma=gamma,
        B_mat=B_mat,
        forcing_norm=forcing_norm,
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


# --------------------------------------------------------------------------
# Operator and npz-loader validation branches
# --------------------------------------------------------------------------


def test_prescribed_forcing_numpy_array_operator_raises() -> None:
    """A dense array operator raises TypeError before any solve."""
    with pytest.raises(TypeError, match="must be a 2-D sparse matrix or PETSc Mat"):
        prescribed_forcing_response(np.ones((5, 5)), 1.0, np.ones(5), np.ones(5, dtype=complex))


def test_prescribed_forcing_non_square_operator_raises() -> None:
    """A non-square sparse operator raises ValueError before any solve."""
    with pytest.raises(ValueError, match="must be a square matrix"):
        prescribed_forcing_response(
            sp.csr_matrix(np.ones((3, 4))),
            1.0,
            np.ones(3),
            np.ones(3, dtype=complex),
        )


def test_prescribed_forcing_w_diag_length_mismatch_raises() -> None:
    """A W_diag shorter than the operator size raises ValueError."""
    L0 = _seeded_operator(6, seed=3)
    with pytest.raises(ValueError, match="does not match operator ndof"):
        prescribed_forcing_response(L0, 1.0, np.ones(4), np.ones(6, dtype=complex))


def test_npz_scalar_plain_float_returns_unchanged() -> None:
    """A plain Python float is returned unchanged."""
    from linstabpy.resolvent.prescribed_io import _npz_scalar

    result = _npz_scalar(2.5)
    assert result == 2.5


def test_load_prescribed_forcing_f_shape_ndof_mismatch_raises(tmp_path: Path) -> None:
    """An f vector whose length differs from expected_ndof raises ValueError."""
    path = tmp_path / "point.npz"
    np.savez(path, f=np.ones(5, dtype=complex), beta=1.0, omega=0.5)
    with pytest.raises(ValueError, match="does not match expected ndof"):
        load_prescribed_forcing(path, expected_ndof=7)


def test_load_prescribed_forcing_f_not_1d_raises(tmp_path: Path) -> None:
    """Forcing vector that is not 1-D raises ValueError.

    The external forcing must be stored as a 1-D vector, not a matrix.
    """
    path = tmp_path / "point.npz"
    f = np.ones((3, 2), dtype=complex)
    np.savez(path, f=f, beta=1.0, omega=0.5)
    with pytest.raises(ValueError, match="f must be a 1-D vector"):
        load_prescribed_forcing(path, expected_ndof=6)


def test_prescribed_forcing_f_hat_not_1d_raises() -> None:
    """Non-1-D f_hat raises ValueError.

    The forcing vector must be a 1-D array.
    """
    L0 = _seeded_operator(6, seed=3)
    w_diag = np.ones(6)
    f_hat = np.ones((6, 1), dtype=complex)
    with pytest.raises(ValueError, match="f_hat must be a 1-D vector"):
        prescribed_forcing_response(L0, 1.0, w_diag, f_hat)


def test_prescribed_forcing_w_diag_non_finite_raises() -> None:
    """Non-finite W_diag (NaN or Inf) raises ValueError.

    The Chu weight vector must contain only finite values.
    """
    L0 = _seeded_operator(6, seed=3)
    w_diag = np.ones(6)
    w_diag[2] = np.nan
    f_hat = np.ones(6, dtype=complex)
    with pytest.raises(ValueError, match="W_diag contains NaN or Inf"):
        prescribed_forcing_response(L0, 1.0, w_diag, f_hat)


def test_validate_w_diag_success_returns_float_array() -> None:
    """Valid W_diag passes validation and returns float array.

    This covers the success path return statement in _validate_w_diag.
    """
    from linstabpy.resolvent.prescribed import _validate_w_diag

    w_input = np.array([1.0, 2.5, 0.3, 4.7])
    w_result = _validate_w_diag(w_input, ndof=4)
    assert isinstance(w_result, np.ndarray)
    assert w_result.dtype == np.float64
    np.testing.assert_array_equal(w_result, w_input)


@pytest.mark.skipif(
    importlib.util.find_spec("petsc4py") is None,
    reason="SKIPPED: petsc4py absent; the solve runs on the compute node",
)
def test_prescribed_map_end_to_end_passing_point(tmp_path: Path) -> None:
    """End-to-end resolvent: shrunk mesh, real base flow, prescribed forcing."""
    from examples.linstab2d.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.solvers.router.prescribed_map import prescribed_map
    from linstabpy.utils.petsc_init import init_petsc_from_argv

    init_petsc_from_argv()

    from petsc4py import PETSc

    if PETSc.COMM_WORLD.getSize() > 1:
        message = "test requires serial execution"
        raise RuntimeError(message)

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

    csv_path = tmp_path / "gains.csv"
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
        "prescribed_output_csv": str(csv_path),
        "resolvent": {"eigen": {"forcing_norm": "l2"}},
    }

    result = prescribed_map(bf, cfg=cfg)

    assert len(result.point_records) == 1
    record = result.point_records[0]
    assert record["status"] == "ok"
    assert record["gain"] is not None
    assert record["gain"] > 0
    assert record["rho_fwd"] is not None
    assert record["rho_fwd"] < 1e-8
    assert csv_path.exists()


@pytest.mark.skipif(
    importlib.util.find_spec("petsc4py") is None,
    reason="SKIPPED: petsc4py absent; the solve runs on the compute node",
)
def test_prescribed_map_mismatched_npz_fails(tmp_path: Path) -> None:
    """Mismatched table beta and npz beta causes failed record."""
    from examples.linstab2d.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.solvers.router.prescribed_map import prescribed_map
    from linstabpy.utils.petsc_init import init_petsc_from_argv

    init_petsc_from_argv()

    from petsc4py import PETSc

    if PETSc.COMM_WORLD.getSize() > 1:
        message = "test requires serial execution"
        raise RuntimeError(message)

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

    csv_path = tmp_path / "gains.csv"
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
        "prescribed_output_csv": str(csv_path),
        "resolvent": {"eigen": {"forcing_norm": "l2"}},
    }

    result = prescribed_map(bf, cfg=cfg)

    assert len(result.point_records) == 1
    record = result.point_records[0]
    assert record["status"] == "failed"
    assert "exception" in record
    assert "table beta" in record["exception"]


def test_prescribed_forcing_zero_forcing_raises() -> None:
    """All-zero forcing fails before the factorization, so no PETSc is needed."""
    with pytest.raises(ValueError, match="zero.*norm.*cannot compute gain"):
        prescribed_forcing_response(
            _seeded_operator(6, seed=3),
            1.0,
            np.ones(6),
            np.zeros(6, dtype=complex),
            solver_cfg={"resolvent": {"eigen": {}}},
        )


@pytest.mark.skipif(
    importlib.util.find_spec("petsc4py") is None,
    reason="SKIPPED: petsc4py absent; the solve runs on the compute node",
)
def test_prescribed_forcing_petsc_mat_input_parity() -> None:
    """PETSc Mat and scipy sparse matrix inputs give identical gains."""
    from petsc4py import PETSc

    L0_scipy = _seeded_operator(6, seed=3)
    # createAIJ reads the index triplet as CSR. _seeded_operator returns
    # CSC, whose triplet read as CSR builds the transpose, so convert
    # first. The transpose keeps every singular value but changes the
    # response to a fixed forcing, which is what this test compares.
    L0_csr = L0_scipy.tocsr()
    L0_petsc = PETSc.Mat().createAIJ(
        size=L0_csr.shape,
        csr=(L0_csr.indptr, L0_csr.indices, L0_csr.data),
    )
    L0_petsc.assemble()

    rng = np.random.default_rng(42)
    f_hat = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    w_diag = np.ones(6)

    _, gain_scipy, _ = prescribed_forcing_response(
        L0_scipy,
        1.0,
        w_diag,
        f_hat,
        solver_cfg={"resolvent": {"eigen": {}}},
    )

    _, gain_petsc, _ = prescribed_forcing_response(
        L0_petsc,
        1.0,
        w_diag,
        f_hat,
        solver_cfg={"resolvent": {"eigen": {}}},
    )

    L0_petsc.destroy()

    assert gain_scipy == pytest.approx(gain_petsc, rel=1e-12)


def test_prescribed_forcing_negative_w_diag_raises() -> None:
    """A negative weight would make the W-norm NaN; it is refused."""
    w = np.ones(6)
    w[2] = -1.0
    with pytest.raises(ValueError, match="1 negative entries"):
        prescribed_forcing_response(_seeded_operator(6, seed=3), 1.0, w, np.ones(6, dtype=complex))


def test_load_prescribed_forcing_keeps_array_extra(tmp_path: Path) -> None:
    """A ``mesh`` coordinate array in the file is kept as an array."""
    path = tmp_path / "f.npz"
    mesh_xy = np.linspace(0.0, 1.0, 4)
    np.savez(path, f=np.ones(3, dtype=complex), beta=0.1, omega=0.5, mesh=mesh_xy, nx=4)
    forcing = load_prescribed_forcing(path, expected_ndof=3)
    np.testing.assert_array_equal(forcing.extras["mesh"], mesh_xy)
    assert forcing.extras["nx"] == 4


def test_load_prescribed_forcing_gamma_round_off_is_accepted(tmp_path: Path) -> None:
    """A float64 round-off difference between file and config gamma passes."""
    path = tmp_path / "f.npz"
    np.savez(path, f=np.ones(3, dtype=complex), beta=0.1, omega=0.5, gamma=0.1 + 0.2 - 0.2)
    forcing = load_prescribed_forcing(path, expected_ndof=3, config_gamma=0.1)
    assert forcing.gamma == pytest.approx(0.1, rel=1e-15)


def test_load_prescribed_forcing_float32_gamma_is_rejected(tmp_path: Path) -> None:
    """A gamma stored as float32 differs by about 1e-9 and is reported."""
    path = tmp_path / "f.npz"
    np.savez(path, f=np.ones(3, dtype=complex), beta=0.1, omega=0.5, gamma=np.float32(0.1))
    with pytest.raises(ValueError, match="gamma"):
        load_prescribed_forcing(path, expected_ndof=3, config_gamma=0.1)
