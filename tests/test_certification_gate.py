# SPDX-License-Identifier: GPL-3.0-only
"""Certification-gate tests for inexact RSVD actions."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.linalg.factorize import factorize
from linstabpy.solvers.router.certification import (
    certified_rsvd,
    check_adjoint_of_approx_factor,
    check_deterministic_linearity,
    check_forward_adjoint_consistency,
    check_multi_seed_rsvd,
    check_tolerance_tied_to_sigma_accuracy,
    derive_solve_tolerance,
)


def _problem():
    k = np.array(
        [
            [3.0 + 0.2j, 0.4 - 0.1j, 0.0, 0.1j],
            [0.2 + 0.3j, 2.5 - 0.1j, 0.3, 0.0],
            [0.0, -0.2j, 1.8 + 0.4j, 0.2 - 0.1j],
            [0.1, 0.0, 0.3j, 1.3 - 0.2j],
        ],
        dtype=complex,
    )
    h = np.linalg.inv(k)
    probes_a = [
        np.array([1.0, -0.5j, 0.25, 0.3 + 0.1j], dtype=complex),
        np.array([-0.1j, 0.7, -0.2 + 0.3j, 0.4], dtype=complex),
    ]
    probes_b = [
        np.array([0.2 - 0.1j, 0.3, -0.4j, 0.5], dtype=complex),
        np.array([0.6, -0.1 + 0.2j, 0.8j, -0.3], dtype=complex),
    ]
    return k, h, probes_a, probes_b


def _positive_cfg(k, probes_a, probes_b):
    sigma_tol = 1.0e-8
    inv_norm_est = np.linalg.norm(np.linalg.inv(k), 2)
    actual_tol = derive_solve_tolerance(sigma_tol, inv_norm_est, safety=0.1) * 0.5
    residuals = [actual_tol * 0.1, actual_tol * 0.2]
    return {
        "sigma_tol": sigma_tol,
        "inv_norm_est": inv_norm_est,
        "actual_solve_tol": actual_tol,
        "forward_tol": actual_tol,
        "adjoint_tol": actual_tol,
        "forward_residuals": residuals,
        "adjoint_residuals": residuals,
        "approx_solve": lambda x: np.linalg.solve(k, x),
        "approx_solve_adjoint": lambda x: np.linalg.solve(k.conj().T, x),
        "probes_a": probes_a,
        "probes_b": probes_b,
        "seeds": (2, 3, 4),
        "n_modes": 1,
        "oversample": 3,
        "linearity_tol": 1.0e-12,
        "adjoint_factor_tol": 1.0e-12,
        "consistency_tol": 1.0e-12,
        "seed_gain_tol": 1.0e-10,
        "seed_mode_tol": 1.0e-8,
    }


def test_tolerance_tied_positive_and_under_toleranced_negative():
    derived = derive_solve_tolerance(1.0e-6, 50.0)

    ok = check_tolerance_tied_to_sigma_accuracy(0.5 * derived, 1.0e-6, 50.0)
    assert ok.passed
    assert ok.measured <= ok.tol

    # Planted defect: the solve uses a fixed residual tolerance looser than the
    # value derived from the requested singular-value accuracy and ||K^-1||.
    bad = check_tolerance_tied_to_sigma_accuracy(10.0 * derived, 1.0e-6, 50.0)
    assert not bad.passed
    assert bad.measured > bad.tol


def test_forward_adjoint_consistency_positive_and_asymmetric_negative():
    ok = check_forward_adjoint_consistency([1.0e-9, 2.0e-9], [1.0e-9, 2.0e-9], 1.0e-8, 1.0e-8)
    assert ok.passed

    # Planted defect: adjoint KSP uses a different tolerance/preconditioner
    # generation and realizes a residual one order of magnitude away.
    bad = check_forward_adjoint_consistency(
        [1.0e-9, 2.0e-9],
        [1.0e-7, 2.0e-7],
        1.0e-8,
        1.0e-6,
    )
    assert not bad.passed


def test_forward_adjoint_consistency_rejects_matched_underconverged_residuals():
    # Planted defect: relative forward/adjoint mismatch is zero, but both
    # realized solves stagnated far above their configured tolerances.
    bad = check_forward_adjoint_consistency([1.0e-2, 1.0e-2], [1.0e-2, 1.0e-2], 1.0e-8, 1.0e-8)
    assert not bad.passed
    assert bad.measured == pytest.approx(1.0e6)


def test_adjoint_of_approx_factor_positive_and_ideal_adjoint_negative():
    k, _h, probes_a, probes_b = _problem()
    approx = k + np.diag([0.2, -0.1j, 0.15, -0.05])

    ok = check_adjoint_of_approx_factor(
        lambda x: np.linalg.solve(approx, x),
        lambda x: np.linalg.solve(approx.conj().T, x),
        probes_a,
        probes_b,
        tol=1.0e-12,
    )
    assert ok.passed

    # Planted defect: the forward solve uses the compressed/approximate factor,
    # but the adjoint solve incorrectly uses the ideal K^H factor.
    bad = check_adjoint_of_approx_factor(
        lambda x: np.linalg.solve(approx, x),
        lambda x: np.linalg.solve(k.conj().T, x),
        probes_a,
        probes_b,
        tol=1.0e-12,
    )
    assert not bad.passed


def test_deterministic_linearity_positive_and_adaptive_negative():
    _k, h, probes_a, probes_b = _problem()
    ok = check_deterministic_linearity(lambda x: h @ x, probes_a, probes_b, tol=1.0e-12)
    assert ok.passed

    # Planted defect: adaptive stopping creates a vector-norm dependent error,
    # so the map is no longer linear even though individual solves may converge.
    drift = np.array([1.0, -0.5j, 0.25, 0.1j], dtype=complex)
    bad = check_deterministic_linearity(
        lambda x: h @ x + 1.0e-4 * np.linalg.norm(x) * drift,
        probes_a,
        probes_b,
        tol=1.0e-12,
    )
    assert not bad.passed


def test_multi_seed_rsvd_positive_and_single_seed_disagreement_negative():
    _k, h, _probes_a, _probes_b = _problem()
    w = np.ones(h.shape[0])
    ok = check_multi_seed_rsvd(
        lambda x: h @ x, lambda x: h.conj().T @ x, w, w, (5, 6, 7), oversample=3
    )
    assert ok.passed

    # Planted defect: stateless input-dependent drift makes the randomized
    # normal action seed-sensitive.
    # A multi-seed check that reuses one seed would see identical probe vectors
    # and miss this disagreement.
    drift = np.array([1.0, -0.5j, 0.25, 0.1j], dtype=complex)

    def seed_sensitive_apply(x):
        return h @ x + 5.0e-2 * np.linalg.norm(x) * drift

    bad = check_multi_seed_rsvd(
        seed_sensitive_apply, lambda x: h.conj().T @ x, w, w, (5, 6, 7), oversample=3
    )
    assert not bad.passed


def test_certified_rsvd_matches_independent_direct_lu_oracle():
    k, h, probes_a, probes_b = _problem()
    w = np.ones(h.shape[0])
    cfg = _positive_cfg(k, probes_a, probes_b)
    fac = factorize(sp.csc_matrix(k))
    try:
        gains, responses, forcings, report = certified_rsvd(
            fac.solve,
            fac.solve_adjoint,
            w,
            w,
            cfg,
        )
    finally:
        fac.destroy()

    # Independent direct LU: form H from dense direct solves and compute
    # its singular values with NumPy, not by snapshotting certified_rsvd output.
    oracle_gain = np.linalg.svd(h, compute_uv=False)[0]
    assert report.passed
    assert report.checks["tolerance_tied_to_sigma_accuracy"].passed
    assert report.checks["forward_adjoint_consistency"].passed
    assert report.checks["adjoint_of_approx_factor"].passed
    assert report.checks["deterministic_tol_linearity"].passed
    assert report.checks["multi_seed_rsvd"].passed
    assert gains[0] == pytest.approx(oracle_gain, rel=cfg["sigma_tol"])
    assert responses.shape == forcings.shape == (h.shape[0], 1)


def test_certified_rsvd_raises_with_named_failed_check():
    k, _h, probes_a, probes_b = _problem()
    w = np.ones(k.shape[0])
    cfg = _positive_cfg(k, probes_a, probes_b)
    cfg["actual_solve_tol"] = derive_solve_tolerance(cfg["sigma_tol"], cfg["inv_norm_est"]) * 10.0

    with pytest.raises(RuntimeError, match="tolerance_tied_to_sigma_accuracy"):
        certified_rsvd(
            lambda x: np.linalg.solve(k, x),
            lambda x: np.linalg.solve(k.conj().T, x),
            w,
            w,
            cfg,
        )


def test_certified_rsvd_linearity_gate_covers_adjoint_action():
    k, h, probes_a, probes_b = _problem()
    w = np.ones(k.shape[0])
    cfg = _positive_cfg(k, probes_a, probes_b)
    drift = np.array([1.0, -0.5j, 0.25, 0.1j], dtype=complex)

    def nonlinear_adjoint(x):
        return h.conj().T @ x + 1.0e-4 * np.linalg.norm(x) * drift

    with pytest.raises(RuntimeError, match="deterministic_tol_linearity"):
        certified_rsvd(lambda x: h @ x, nonlinear_adjoint, w, w, cfg)


def test_single_seed_skips_multi_seed_and_still_returns_a_gain() -> None:
    k, h, probes_a, probes_b = _problem()
    w = np.ones(h.shape[0])
    cfg = _positive_cfg(k, probes_a, probes_b)
    cfg["seeds"] = (2,)
    gains, _responses, _forcings, report = certified_rsvd(
        lambda x: h @ x, lambda x: h.conj().T @ x, w, w, cfg
    )
    assert "multi_seed_rsvd" in report.checks
    assert report.checks["multi_seed_rsvd"].passed
    assert "one seed" in report.checks["multi_seed_rsvd"].message
    assert np.isfinite(gains[0])
    assert gains[0] == pytest.approx(np.linalg.svd(h, compute_uv=False)[0], rel=cfg["sigma_tol"])


def test_failed_check_still_returns_gain_when_certification_is_not_required() -> None:
    k, h, probes_a, probes_b = _problem()
    w = np.ones(k.shape[0])
    cfg = _positive_cfg(k, probes_a, probes_b)
    cfg["actual_solve_tol"] = derive_solve_tolerance(cfg["sigma_tol"], cfg["inv_norm_est"]) * 10.0
    cfg["certification_required"] = False
    gains, _responses, _forcings, report = certified_rsvd(
        lambda x: h @ x, lambda x: h.conj().T @ x, w, w, cfg
    )
    assert report.passed is False
    assert not report.checks["tolerance_tied_to_sigma_accuracy"].passed
    assert np.isfinite(gains[0])
