# SPDX-License-Identifier: GPL-3.0-only
"""Regime-b (NEAR_POLE) discounted-resolvent integration tests for resolvent_map."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.solvers.router.band_marking import NEAR_POLE, SMOOTH, STABLE
from linstabpy.solvers.router.resolvent_map import (
    _check_missing_engine_gate,
    _regime_b_discount_gamma,
)


def _tiny_problem():
    """Small stable LNSE problem reused for end-to-end resolvent_map routing checks."""
    from linstabpy.baseflow import BaseFlow, sutherland_air
    from linstabpy.mesh import Mesh

    mesh = Mesh.create((0.0, 1.0), (0.0, 1.0), nx=4, ny=4, fd_order=2)
    ones = np.ones_like(mesh.X)
    zeros = np.zeros_like(mesh.X)
    base_flow = BaseFlow(
        Re=1.0e3,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=ones.copy(),
        U=0.2 * ones.copy(),
        V=zeros.copy(),
        W=zeros.copy(),
        T=ones.copy(),
    )
    sutherland_air(base_flow)
    return mesh, base_flow


def _cfg(mesh, *, danger_radius, regime_b_gamma=None):
    cfg = {
        "mesh": mesh,
        "model": "2D",
        "bc_recipe": [{"type": "dirichlet", "borders": "lrtb", "variables": "ruvwT"}],
        "bc_penalty": 10.0,
        "assembly_backend": "scipy",
        "mask_bc_forcing": False,
        "n_modes_resolvent": 1,
        "stage_heartbeat_seconds": 0.0,
        "band_marking": {"n_modes": 1, "ncv": 3, "danger_radius": danger_radius},
        "solver": {
            "eigen": {"eps_tol": 1.0e-8, "eps_max_it": 1000},
            "resolvent": {"eigen": {"eps_tol": 1.0e-8}},
        },
    }
    if regime_b_gamma is not None:
        cfg["regime_b_discount_gamma"] = regime_b_gamma
    return cfg


def test_resolvent_map_near_pole_routes_regime_b_finite_gain():
    """End-to-end: a NEAR_POLE point is computed via regime-b (gamma>0) with a FINITE gain.

    A large danger_radius forces the band-marker to label the forcing frequency
    NEAR_POLE; the discounted-resolvent wiring must then (a) route it to
    engine='regime-b', (b) record the discount gamma, and (c) yield a finite gain
    (the CP1 well-posedness gate). The pole-taming numerics themselves are covered
    by tests/test_resolvent_discount.py; here we prove the resolvent_map wiring.
    """
    pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
    pytest.importorskip("slepc4py", reason="SKIPPED: missing SLEPc stack")

    from linstabpy import resolvent_map

    mesh, base_flow = _tiny_problem()
    cfg = _cfg(mesh, danger_radius=1.0e6, regime_b_gamma=0.25)
    beta = np.array([0.1])
    omega = np.array([0.2])

    result = resolvent_map(base_flow, beta, omega, cfg)

    assert result.regime_label[0, 0] == NEAR_POLE
    assert np.isfinite(result.gain[0, 0])
    pt = result.point_records[0]
    assert pt["engine"] == "regime-b"
    assert pt["resolvent_discount_gamma"] == pytest.approx(0.25)


def test_resolvent_map_stable_point_regime_a_bit_identical_to_gamma0():
    """A STABLE point still uses regime-a (gamma=0) and matches the direct gamma=0 oracle."""
    pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
    pytest.importorskip("slepc4py", reason="SKIPPED: missing SLEPc stack")

    from linstabpy import resolvent_map
    from linstabpy.operators.assembly import build_lnse_system
    from linstabpy.operators.energy_norm import compute_chu_norm
    from linstabpy.resolvent.analysis import resolvent_analysis

    mesh, base_flow = _tiny_problem()
    cfg = _cfg(mesh, danger_radius=0.0)  # no NEAR_POLE -> all STABLE -> regime-a
    beta = np.array([0.1])
    omega = np.array([0.2])

    result = resolvent_map(base_flow, beta, omega, cfg)
    assert result.regime_label[0, 0] == STABLE

    L0, _idx, _bc = build_lnse_system(
        mesh,
        base_flow,
        kx=float(beta[0]),
        model="2D",
        bc_recipe=cfg["bc_recipe"],
        bc_penalty=float(cfg["bc_penalty"]),
        backend="scipy",
        solver_cfg=cfg,
    )
    p = mesh.used_ind
    W_diag, _ = compute_chu_norm(
        mesh.W.ravel(order="F")[p],
        base_flow.RHO.ravel(order="F")[p],
        base_flow.T.ravel(order="F")[p],
        base_flow.kappa,
        base_flow.Ma,
        mesh.ngp,
    )
    direct_gains, *_ = resolvent_analysis(L0, float(omega[0]), 1, W_diag, solver_cfg=cfg)
    # gamma=0 regime-a path must be bit-identical to the direct gamma=0 oracle
    np.testing.assert_allclose(result.gain[0, 0], direct_gains[0], rtol=1.0e-7, atol=1.0e-7)


def test_regime_b_discount_gamma_explicit_override():
    """Test explicit regime_b_discount_gamma from config."""
    cfg = {
        "regime_b_discount_gamma": 0.5,
    }
    gamma = _regime_b_discount_gamma(cfg)
    assert gamma == pytest.approx(0.5)


def test_regime_b_discount_gamma_default():
    """Test default regime_b_discount_gamma when not in config."""
    cfg = {}
    gamma = _regime_b_discount_gamma(cfg)
    assert gamma == pytest.approx(0.25)


def test_regime_b_discount_gamma_validation():
    """Test that regime_b_discount_gamma must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        _regime_b_discount_gamma({"regime_b_discount_gamma": 0.0})

    with pytest.raises(ValueError, match="must be positive"):
        _regime_b_discount_gamma({"regime_b_discount_gamma": -0.1})


def test_check_missing_engine_gate_regime_b_returns_gamma():
    """Test that NEAR_POLE label routes to regime-b with gamma>0."""
    cfg = {
        "regime_b_discount_gamma": 0.3,
    }
    emitted = set()

    regime, gamma = _check_missing_engine_gate(NEAR_POLE, cfg, emitted)

    assert regime == "regime-b"
    assert gamma == pytest.approx(0.3)
    assert gamma > 0.0


def test_check_missing_engine_gate_regime_b_default_gamma():
    """Test regime-b with default gamma when not specified."""
    cfg = {}
    emitted = set()

    regime, gamma = _check_missing_engine_gate(NEAR_POLE, cfg, emitted)

    assert regime == "regime-b"
    assert gamma == pytest.approx(0.25)


def test_check_missing_engine_gate_stable_returns_regime_a():
    """Test that STABLE label routes to regime-a with gamma=0."""
    cfg = {}
    emitted = set()

    regime, gamma = _check_missing_engine_gate(STABLE, cfg, emitted)

    assert regime == "regime-a"
    assert gamma == pytest.approx(0.0)


def test_check_missing_engine_gate_smooth_returns_regime_a_fallback():
    """Test that SMOOTH label falls back to regime-a (regime-c not implemented)."""
    cfg = {}
    emitted = set()

    regime, gamma = _check_missing_engine_gate(SMOOTH, cfg, emitted)

    # SMOOTH should fall back to regime-a since regime-c is not implemented
    assert regime == "regime-a"
    assert gamma == pytest.approx(0.0)


def test_check_missing_engine_gate_smooth_raises_when_enabled():
    """Test that SMOOTH raises NotImplementedError when enable_regime_c=true."""
    cfg = {"enable_regime_c": True}
    emitted = set()

    with pytest.raises(NotImplementedError, match="regime-c"):
        _check_missing_engine_gate(SMOOTH, cfg, emitted)


def test_regime_b_discount_gamma_config_hierarchy():
    """Test cfg key lookup hierarchy for regime_b_discount_gamma."""
    # Direct key
    cfg1 = {"regime_b_discount_gamma": 0.15}
    assert _regime_b_discount_gamma(cfg1) == pytest.approx(0.15)

    # Nested under router (via _cfg_value lookup)
    cfg2 = {"solver": {"router": {}}}
    assert _regime_b_discount_gamma(cfg2) == pytest.approx(0.25)
