# SPDX-License-Identifier: GPL-3.0-only
"""Check the (beta, f) line walk against a direct resolvent solve."""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.config import Config
from linstabpy.linalg.backends import factor_backend_available
from linstabpy.solvers.router.map_walk import (
    LineWalker,
    WalkAborted,
    WalkDeferred,
    _GatedPcSolve,
    combine_probes,
    probe_right_hand_side,
    walk_is_cheaper,
)
from linstabpy.solvers.router.stable_sweep import _right_preconditioned_gmres

_CFG = {"resolvent": {"eigen": {"eps_tol": 1e-10}}}
_SIGMA_TOL = 1.0e-6


def test_walk_is_cheaper_rejects_a_slow_or_stale_probe() -> None:
    cheap = dict(
        probe_s=0.05,
        probe_iters=4,
        probe_residual=1.0e-12,
        last_factor_s=20.0,
        rtol=1.0e-10,
        restart=50,
        n_modes=1,
        n_applies=20,
    )
    assert walk_is_cheaper(**cheap) is True
    assert walk_is_cheaper(**{**cheap, "n_applies": 0}) is False
    assert walk_is_cheaper(**{**cheap, "probe_residual": 1.0e-6}) is False
    assert walk_is_cheaper(**{**cheap, "probe_iters": 9}) is False
    assert walk_is_cheaper(**{**cheap, "probe_iters": 5}) is False
    assert walk_is_cheaper(**{**cheap, "probe_iters": 0}) is False
    assert walk_is_cheaper(**{**cheap, "probe_s": 0.2}) is False
    assert walk_is_cheaper(**{**cheap, "last_factor_s": 0.0}) is False


def test_walk_is_cheaper_uses_quarter_factor_slack() -> None:
    assert (
        walk_is_cheaper(
            probe_s=0.2,
            probe_iters=4,
            probe_residual=1.0e-12,
            last_factor_s=20.0,
            rtol=1.0e-10,
            restart=50,
            n_modes=1,
            n_applies=20,
        )
        is False
    )


def test_combine_probes_uses_the_worse_side() -> None:
    probe_s, probe_iters, probe_residual = combine_probes(0.2, 4, 1.0e-12, 5.0, 80, 1.0e-8)
    assert probe_s == 5.2
    assert probe_iters == 80
    assert probe_residual == 1.0e-8
    assert (
        walk_is_cheaper(
            probe_s=probe_s,
            probe_iters=probe_iters,
            probe_residual=probe_residual,
            last_factor_s=20.0,
            rtol=1.0e-10,
            restart=50,
            n_modes=1,
            n_applies=0,
        )
        is False
    )


def test_walker_defers_when_the_job_cannot_finish_a_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walker = _walker()
    walker.last_factor_s = 60.0
    walker.last_direct_wall_s = 80.0
    walker.factor = object()
    monkeypatch.setenv("LINSTABPY_JOB_DEADLINE", str(time.time() + 5.0))
    with pytest.raises(WalkDeferred):
        walker.compute(sp.eye(4, format="csc", dtype=complex), 1.0, np.ones(4), 1)


def test_probe_right_hand_side_uses_last_solve_not_ones() -> None:
    last = np.array([0.3 - 0.2j, 1.1, -0.4 + 0.7j, 2.0], dtype=complex)
    rhs = probe_right_hand_side(4, last)
    assert rhs == pytest.approx(last)
    fallback = probe_right_hand_side(4, None)
    assert fallback.shape == (4,)
    assert not np.allclose(fallback, np.ones(4))
    assert np.linalg.norm(fallback) > 0.0


class _EyePc:
    backend = "eye"

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(rhs, dtype=complex)

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(rhs, dtype=complex)


def test_gated_raises_walk_aborted_after_deadline() -> None:
    operator = sp.eye(4, format="csc", dtype=complex)
    gated = _GatedPcSolve(
        operator,
        _EyePc(),
        rtol=1.0e-10,
        maxiter=50,
        restart=50,
        x0=None,
        deadline=time.perf_counter() - 1.0,
        apply_budget_s=0.0,
        raise_on_fail=False,
    )
    quiet = gated.solve(np.ones(4))
    assert gated.failed is True
    assert np.allclose(quiet, 0.0)
    gated.raise_on_fail = True
    with pytest.raises(WalkAborted):
        gated.solve(np.ones(4))


def test_gmres_stops_when_deadline_has_passed() -> None:
    calls = {"n": 0}

    def matvec(x: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return np.asarray(x, dtype=complex)

    _x, iters, residual = _right_preconditioned_gmres(
        matvec,
        lambda z: np.asarray(z, dtype=complex),
        np.ones(4),
        rtol=1.0e-16,
        maxiter=40,
        restart=10,
        deadline=time.perf_counter() - 1.0,
    )
    assert calls["n"] == 0
    assert iters == 0
    assert residual > 0.5


def _operator() -> sp.csc_matrix:
    diagonal = 2.7 + 0.22 * np.arange(8) + 1j * (0.17 - 0.025 * np.arange(8))
    lower = 0.045 - 0.025j + 0.003 * np.arange(7)
    upper = -0.035 + 0.02j - 0.002 * np.arange(7)
    return sp.diags([lower, diagonal, upper], [-1, 0, 1], format="csc", dtype=complex)


def _direct_gain(L0: sp.spmatrix, omega: float) -> float:
    from linstabpy.resolvent.analysis import resolvent_analysis

    gains, _, _ = resolvent_analysis(
        L0,
        omega=omega,
        n_modes=1,
        W_diag=np.ones(L0.shape[0]),
        solver_cfg=_CFG,
    )
    return float(gains[0])


def _walker() -> LineWalker:
    return LineWalker(solver_cfg=_CFG, cfg_obj=Config.load(_CFG), rtol=1.0e-8)


@pytest.fixture
def mumps_or_skip() -> None:
    pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
    pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
    if not factor_backend_available("mumps"):
        pytest.skip("SKIPPED: mumps backend unavailable")


def test_first_two_frequencies_are_mumps_anchors(mumps_or_skip: None) -> None:
    L0 = _operator()
    walker = _walker()
    try:
        first = walker.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        second = walker.compute(L0, 0.85, np.ones(L0.shape[0]), 1)
        assert first.source == "factor"
        assert second.source == "factor"
        assert walker.n_factor == 2
        assert walker.n_direct == 2
        assert walker.last_n_applies > 0
        assert first.gain == pytest.approx(_direct_gain(L0, 0.8), rel=_SIGMA_TOL, abs=_SIGMA_TOL)
        assert second.gain == pytest.approx(_direct_gain(L0, 0.85), rel=_SIGMA_TOL, abs=_SIGMA_TOL)
    finally:
        walker.close()


def test_walker_follows_the_sign_convention_of_its_config(mumps_or_skip: None) -> None:
    from linstabpy.resolvent.analysis import resolvent_analysis

    parity_cfg = {
        "resolvent": {"eigen": {"eps_tol": 1e-10}},
        "resolvent_sign_convention": "matlab_parity",
    }
    L0 = _operator()
    omega = 0.8
    walker = LineWalker(solver_cfg=parity_cfg, cfg_obj=Config.load(parity_cfg), rtol=1.0e-8)
    try:
        point = walker.compute(L0, omega, np.ones(L0.shape[0]), 1)
    finally:
        walker.close()
    gains, _, _ = resolvent_analysis(
        L0, omega=omega, n_modes=1, W_diag=np.ones(L0.shape[0]), solver_cfg=parity_cfg
    )
    assert point.gain == pytest.approx(float(gains[0]), rel=_SIGMA_TOL, abs=_SIGMA_TOL)
    # The operator is asymmetric in omega, so the other convention gives another gain.
    assert abs(point.gain - _direct_gain(L0, omega)) > 1e-3 * point.gain


def test_every_point_carries_the_farm_svd_residuals(mumps_or_skip: None) -> None:
    """A direct point carries rho_fwd / rho_adj like a farm point.

    Compute those residuals with one forward and one adjoint apply.
    The farm's own level is 1e-9 .. 6e-9 on the 200-point farm rows.
    """
    from linstabpy.resolvent.analysis import resolvent_analysis

    L0 = _operator()
    walker = _walker()
    try:
        point = walker.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        assert np.isfinite(point.rho_fwd) and np.isfinite(point.rho_adj)
        assert point.rho_fwd < 1e-8 and point.rho_adj < 1e-8
        *_, rho_fwd, rho_adj = resolvent_analysis(
            L0, omega=0.8, n_modes=1, W_diag=np.ones(L0.shape[0]), solver_cfg=_CFG, residuals=True
        )
        # Same operator, same triplet, same formula: the two paths agree to the
        # residuals' own noise level.
        assert point.rho_fwd == pytest.approx(float(rho_fwd[0]), abs=1e-9)
        assert point.rho_adj == pytest.approx(float(rho_adj[0]), abs=1e-9)
    finally:
        walker.close()


def test_first_frequency_factors_and_matches_direct(mumps_or_skip: None) -> None:
    L0 = _operator()
    walker = _walker()
    try:
        point = walker.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        assert point.source == "factor"
        assert walker.n_factor == 1
        assert point.gain == pytest.approx(_direct_gain(L0, 0.8), rel=_SIGMA_TOL, abs=_SIGMA_TOL)
    finally:
        walker.close()


def test_second_frequency_matches_direct_and_may_walk(mumps_or_skip: None) -> None:
    L0 = _operator()
    walker = _walker()
    try:
        walker.n_anchor = 1
        walker.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        walker.last_direct_wall_s = 1.0e9
        walker.last_factor_s = 1.0e9
        point = walker.compute(L0, 0.85, np.ones(L0.shape[0]), 1)
        assert point.source in {"walk", "fallback", "budget"}
        assert point.gain == pytest.approx(_direct_gain(L0, 0.85), rel=_SIGMA_TOL, abs=_SIGMA_TOL)
    finally:
        walker.close()


def test_failed_walk_falls_back_and_still_matches(mumps_or_skip: None) -> None:
    L0 = _operator()
    walker = _walker()
    walker.rtol = 1.0e-16
    walker.maxiter = 1
    walker.restart = 1
    try:
        walker.n_anchor = 1
        walker.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        walker.last_direct_wall_s = 1.0e9
        walker.last_factor_s = 1.0e9
        n0 = walker.n_factor
        point = walker.compute(L0, 1.6, np.ones(L0.shape[0]), 1)
        assert point.source == "fallback"
        assert walker.n_factor == n0 + 1
        assert point.gain == pytest.approx(_direct_gain(L0, 1.6), rel=_SIGMA_TOL, abs=_SIGMA_TOL)
    finally:
        walker.close()


def test_second_frequency_does_not_need_recycling_block(mumps_or_skip: None) -> None:
    L0 = _operator()
    walker = LineWalker(solver_cfg=_CFG, cfg_obj=Config.load(_CFG), rtol=1.0e-8)
    try:
        walker.n_anchor = 1
        first = walker.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        walker.last_direct_wall_s = 1.0e9
        walker.last_factor_s = 1.0e9
        second = walker.compute(L0, 0.85, np.ones(L0.shape[0]), 1)
        assert first.source == "factor"
        assert second.source in {"walk", "fallback", "budget"}
        assert walker.n_factor >= 1
    finally:
        walker.close()


def test_over_budget_walk_factors_this_omega(mumps_or_skip: None) -> None:
    L0 = _operator()
    walker = _walker()
    try:
        walker.n_anchor = 1
        first = walker.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        assert first.source == "factor"
        assert walker.last_factor_s > 0.0
        walker.last_factor_s = 1.0e-18
        walker.last_direct_wall_s = 1.0e-18
        n0 = walker.n_factor
        point = walker.compute(L0, 1.6, np.ones(L0.shape[0]), 1)
        assert point.source == "budget"
        assert walker.n_factor == n0 + 1
        assert point.gain == pytest.approx(_direct_gain(L0, 1.6), rel=_SIGMA_TOL, abs=_SIGMA_TOL)
    finally:
        walker.close()


def test_new_walker_factors_again(mumps_or_skip: None) -> None:
    L0 = _operator()
    first = _walker()
    second = _walker()
    try:
        first.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        second.compute(L0, 0.8, np.ones(L0.shape[0]), 1)
        assert first.n_factor == 1
        assert second.n_factor == 1
    finally:
        first.close()
        second.close()
