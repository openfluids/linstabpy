# SPDX-License-Identifier: GPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linstabpy.solvers.router import stable_sweep
from linstabpy.solvers.router.certification import certified_rsvd


@dataclass
class _ScipyFactorization:
    matrix: sp.csc_matrix
    lu: spla.SuperLU

    def solve(self, rhs):
        return self.lu.solve(np.asarray(rhs, dtype=complex))

    def solve_adjoint(self, rhs):
        return self.lu.solve(np.asarray(rhs, dtype=complex), trans="H")

    def residual(self, rhs, x):
        return float(np.linalg.norm(self.matrix @ x - rhs))


class _FactorizeRecorder:
    def __init__(self):
        self.factors: list[_ScipyFactorization] = []

    def __call__(self, operator, **_kwargs):
        matrix = sp.csc_matrix(operator, dtype=complex)
        factor = _ScipyFactorization(matrix=matrix, lu=spla.splu(matrix))
        self.factors.append(factor)
        return factor


def _operator():
    diagonal = np.array([2.0 + 0.1j, 2.7 - 0.2j, 3.4 + 0.3j, 4.2 - 0.1j, 5.0 + 0.2j])
    coupling = np.array([0.03 - 0.02j, -0.04 + 0.01j, 0.02 + 0.03j, -0.01j])
    return sp.diags([coupling, diagonal, 0.5 * coupling], [-1, 0, 1], format="csc", dtype=complex)


def _cfg(n):
    probes_a = (np.eye(n, dtype=complex)[0], np.eye(n, dtype=complex)[-1])
    probes_b = (np.eye(n, dtype=complex)[1], np.eye(n, dtype=complex)[-2])
    return {
        "sigma_tol": 1.0e-7,
        "inv_norm_est": 1.0,
        "actual_solve_tol": 1.0e-10,
        "fgmres_rtol": 1.0e-11,
        "fgmres_maxiter": n,
        "n_modes": 1,
        "oversample": n,
        "seeds": (4, 5, 6),
        "probes_a": probes_a,
        "probes_b": probes_b,
        "W": np.ones(n, dtype=complex),
        "invW": np.ones(n, dtype=complex),
        "linearity_tol": 1.0e-9,
    }


def _direct_gain_map(L0, omegas, cfg):
    gains = []
    for omega in omegas:
        matrix = stable_sweep.shifted_operator(L0, omega)
        factor = _ScipyFactorization(matrix=matrix, lu=spla.splu(matrix))

        def apply_h(rhs):
            return factor.solve(rhs)

        def apply_h_adj(rhs):
            return factor.solve_adjoint(rhs)

        direct_cfg = dict(cfg)
        direct_cfg.update(
            {
                "forward_residuals": (0.0, 0.0),
                "adjoint_residuals": (0.0, 0.0),
                "approx_solve": factor.solve,
                "approx_solve_adjoint": factor.solve_adjoint,
            }
        )
        omega_gains, *_ = certified_rsvd(apply_h, apply_h_adj, cfg["W"], cfg["invW"], direct_cfg)
        gains.append(float(omega_gains[0]))
    return np.asarray(gains)


def test_single_anchor_sweep_matches_direct_oracle(monkeypatch):
    L0 = _operator()
    omegas = np.array([0.95, 1.0, 1.05], dtype=complex)
    anchors = np.array([1.0], dtype=complex)
    cfg = _cfg(L0.shape[0])
    recorder = _FactorizeRecorder()
    monkeypatch.setattr(stable_sweep, "factorize", recorder)

    result = stable_sweep.sweep_stable_regime(L0, omegas, anchors, cfg)
    direct = _direct_gain_map(L0, omegas, cfg)

    np.testing.assert_allclose(result.gain, direct, rtol=cfg["sigma_tol"], atol=cfg["sigma_tol"])
    np.testing.assert_array_equal(result.anchor_indices, np.zeros(omegas.size, dtype=int))
    assert len(recorder.factors) == anchors.size
    assert result.iterations.shape == omegas.shape
    assert result.residuals.shape == omegas.shape
    assert result.responses.shape == (L0.shape[0], omegas.size)
    assert result.forcings.shape == (L0.shape[0], omegas.size)
    on_anchor = np.asarray(omegas) == anchors[0]
    assert np.all(result.iterations[on_anchor] == 0)
    assert np.all(result.iterations[~on_anchor] > 0)
    assert np.all(np.isfinite(result.residuals))
    assert all(report.passed for report in result.certification_reports)


def test_multi_anchor_sweep_matches_direct_oracle_and_nearest_anchor(monkeypatch):
    L0 = _operator()
    omegas = np.array([0.75, 0.9, 1.25, 1.6], dtype=complex)
    anchors = np.array([0.8, 1.5], dtype=complex)
    cfg = _cfg(L0.shape[0])
    recorder = _FactorizeRecorder()
    monkeypatch.setattr(stable_sweep, "factorize", recorder)

    result = stable_sweep.sweep_stable_regime(L0, omegas, anchors, cfg)
    direct = _direct_gain_map(L0, omegas, cfg)

    np.testing.assert_allclose(result.gain, direct, rtol=cfg["sigma_tol"], atol=cfg["sigma_tol"])
    np.testing.assert_array_equal(result.anchor_indices, np.array([0, 0, 1, 1]))
    assert len(recorder.factors) == anchors.size


def test_forward_and_adjoint_reuse_same_anchor_factorization(monkeypatch):
    L0 = _operator()
    omegas = np.array([0.95, 1.05], dtype=complex)
    anchors = np.array([1.0], dtype=complex)
    cfg = _cfg(L0.shape[0])
    recorder = _FactorizeRecorder()
    seen_pairs = []
    real_certified_rsvd = stable_sweep.certified_rsvd

    def recording_certified_rsvd(apply_h, apply_h_adj, W, invW, cert_cfg):
        seen_pairs.append((cert_cfg["approx_solve"], cert_cfg["approx_solve_adjoint"]))
        return real_certified_rsvd(apply_h, apply_h_adj, W, invW, cert_cfg)

    monkeypatch.setattr(stable_sweep, "factorize", recorder)
    monkeypatch.setattr(stable_sweep, "certified_rsvd", recording_certified_rsvd)

    stable_sweep.sweep_stable_regime(L0, omegas, anchors, cfg)

    assert len(recorder.factors) == 1
    factor = recorder.factors[0]
    assert all(
        forward.__self__ is factor and adjoint.__self__ is factor for forward, adjoint in seen_pairs
    )


def test_sweep_through_production_factorize_seam():
    # No monkeypatch: exercises the real linalg.factorize call signature and
    # PETSc-backed anchor factor end to end against the direct closed form.
    L0 = _operator()
    omegas = np.array([0.95, 1.0, 1.05], dtype=complex)
    anchors = np.array([1.0], dtype=complex)
    cfg = _cfg(L0.shape[0])

    result = stable_sweep.sweep_stable_regime(L0, omegas, anchors, cfg)
    direct = _direct_gain_map(L0, omegas, cfg)

    np.testing.assert_allclose(result.gain, direct, rtol=cfg["sigma_tol"], atol=cfg["sigma_tol"])
    assert all(report.passed for report in result.certification_reports)
    on_anchor = np.asarray(omegas) == anchors[0]
    assert np.all(result.iterations[on_anchor] == 0)
    assert np.all(result.iterations[~on_anchor] > 0)
