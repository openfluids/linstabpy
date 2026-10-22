# SPDX-License-Identifier: GPL-3.0-only
"""Discounted resolvent oracles on tiny complex operators."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
import scipy.linalg
import scipy.sparse as sp

from linstabpy.resolvent.analysis import resolvent_analysis

_CFG = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}
_CFG_PARITY = {
    "resolvent": {"eigen": {"eps_tol": 1e-12}},
    "resolvent_sign_convention": "matlab_parity",
}
_OMEGA = 3.0


def _weighted_identity(ndof: int) -> np.ndarray:
    return np.ones(ndof)


def _leading_gain(
    L0: sp.spmatrix,
    omega: float,
    *,
    gamma: float = 0.0,
    solver_cfg: dict = _CFG,
) -> float:
    gains, _, _ = resolvent_analysis(
        L0,
        omega=omega,
        n_modes=1,
        W_diag=_weighted_identity(L0.shape[0]),
        solver_cfg=solver_cfg,
        gamma=gamma,
    )
    return float(gains[0])


def _dense_gain_oracle(
    L0: sp.spmatrix,
    omega: float,
    *,
    gamma: float = 0.0,
    matlab_parity: bool = False,
) -> float:
    dense = L0.toarray()
    eye = np.eye(dense.shape[0])
    if matlab_parity:
        shifted = dense - (float(gamma) + 1j * omega) * eye
    else:
        shifted = (float(gamma) - 1j * omega) * eye - dense
    return float(1.0 / scipy.linalg.svdvals(shifted)[-1])


def _near_diagonal_operator() -> sp.csc_matrix:
    # The small off-diagonal entries make this a near-diagonal matrix without
    # changing the independent dense-SVD closed form used by the assertions.
    dense = np.diag(
        np.array(
            [
                -0.2 + 1j * _OMEGA,
                -1.7 - 0.5j,
                -2.5 + 1.2j,
                -4.0 - 2.0j,
                -6.0 + 4.0j,
                -8.0 - 6.0j,
            ],
            dtype=complex,
        )
    )
    dense[0, 2] = 0.03 - 0.01j
    dense[1, 3] = -0.02 + 0.04j
    dense[3, 5] = 0.01 + 0.02j
    return sp.csc_matrix(dense)


def _near_pole_operator() -> sp.csc_matrix:
    # NEAR-pole, deliberately NOT the exact pole. An eigenvalue exactly at
    # 1j*_OMEGA makes K_0 = L0 - 1j*omega*I singular, and the sparse-LU solve of
    # an exactly-singular system returns a particular solution with a tiny
    # residual and a deceptively small finite gain (~0.32) -- it does NOT visibly
    # blow up. Offsetting the eigenvalue by 1e-5 keeps K_0 well-defined but
    # near-singular, so gamma=0 gives a genuine ~1/1e-5 = 1e5 blow-up while the
    # discount tames it -- the honest demonstration of the capability.
    return sp.diags(
        np.array(
            [
                1j * _OMEGA + 1.0e-5j,
                -2.0 - 1.0j,
                -3.0 + 4.0j,
                -5.0 - 3.0j,
                -7.0 + 6.0j,
                -11.0 - 8.0j,
            ],
            dtype=complex,
        ),
        format="csc",
        dtype=complex,
    )


def _stable_diagonal_operator() -> sp.csc_matrix:
    return sp.diags(
        np.array(
            [
                -0.2 + 1j * _OMEGA,
                -1.5 - 0.2j,
                -2.0 + 2.0j,
                -3.5 - 4.0j,
                -5.0 + 5.0j,
                -9.0 - 7.0j,
            ],
            dtype=complex,
        ),
        format="csc",
        dtype=complex,
    )


def test_gamma_zero_gain_matches_independent_dense_svd_oracle():
    L0 = _near_diagonal_operator()

    gain = _leading_gain(L0, _OMEGA, gamma=0.0)
    expected = _dense_gain_oracle(L0, _OMEGA, gamma=0.0)

    assert gain == pytest.approx(expected, rel=2e-11, abs=1e-12)


def test_gamma_zero_gain_matches_dense_svd_oracle_under_matlab_parity():
    L0 = _near_diagonal_operator()

    gain = _leading_gain(L0, _OMEGA, gamma=0.0, solver_cfg=_CFG_PARITY)
    expected = _dense_gain_oracle(L0, _OMEGA, gamma=0.0, matlab_parity=True)

    assert gain == pytest.approx(expected, rel=2e-11, abs=1e-12)


@pytest.mark.parametrize(
    ("solver_cfg", "omega_pole"),
    [(_CFG, -_OMEGA), (_CFG_PARITY, _OMEGA)],
    ids=["physical", "parity"],
)
def test_discount_rescues_blowup_near_pole(solver_cfg, omega_pole):
    # Capability contrast: near an (unstable) eigenvalue the UNshifted resolvent
    # gain blows up, and a real discount gamma>0 rescues well-posedness.
    # The eigenvalue 1j*_OMEGA is a pole at omega = i*lambda = -_OMEGA (physical)
    # and at omega = +_OMEGA (matlab_parity).
    L0 = _near_pole_operator()

    gain_unshifted = _leading_gain(L0, omega_pole, gamma=0.0, solver_cfg=solver_cfg)
    gain_discounted = _leading_gain(L0, omega_pole, gamma=0.25, solver_cfg=solver_cfg)

    # gamma=0: eigenvalue offset 1e-5 from the pole -> gain ~ 1/1e-5 = 1e5 (huge).
    assert np.isfinite(gain_unshifted)
    assert gain_unshifted > 1.0e4, (
        f"unshifted gain {gain_unshifted:.3e} should blow up near the pole"
    )
    # gamma=0.25: shifted eigenvalue ~ -0.25 -> gain ~ 1/0.25 = 4.0 (finite, small).
    assert np.isfinite(gain_discounted)
    assert gain_discounted == pytest.approx(4.0, rel=1e-3, abs=1e-3)
    # The discount must dramatically reduce the blow-up.
    assert gain_discounted < gain_unshifted / 1.0e3


def test_return_info_round_trip_and_default_tuple_shape():
    L0 = _stable_diagonal_operator()

    default_result = resolvent_analysis(
        L0,
        omega=_OMEGA,
        n_modes=1,
        W_diag=_weighted_identity(L0.shape[0]),
        solver_cfg=_CFG,
        gamma=0.125,
    )
    info_result = resolvent_analysis(
        L0,
        omega=_OMEGA,
        n_modes=1,
        W_diag=_weighted_identity(L0.shape[0]),
        solver_cfg=_CFG,
        gamma=0.125,
        return_info=True,
    )

    assert len(default_result) == 3
    assert len(info_result) == 4
    info = info_result[-1]
    assert info["resolvent_discount_gamma"] == 0.125
    assert info["factor_s"] >= 0.0
    assert info["svd_s"] >= 0.0
    np.testing.assert_allclose(info_result[0], default_result[0])


def test_gain_is_monotonic_non_increasing_with_discount():
    L0 = _stable_diagonal_operator()
    gammas = np.array([0.0, 0.1, 0.25, 0.5, 1.0])

    gains = np.array([_leading_gain(L0, _OMEGA, gamma=float(gamma)) for gamma in gammas])

    assert np.all(np.diff(gains) <= 1e-12), gains
