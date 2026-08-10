# SPDX-License-Identifier: GPL-3.0-only
"""Tests for the single-point resolvent driver stage's B/C forcing masks.

`_stage_run_resolvent` used to hand-roll a BC-only diagonal mask and set
``C_mat = B_mat.copy()``, so `input_region`/`output_region` config keys were
silently ignored on this path and B/C were always forced identical.  These
tests pin the fix: the stage now delegates to
`linstabpy.solvers.router.resolvent_map._forcing_response_mats`, the same
helper the multi-point `resolvent_map` sweep uses.
"""

from __future__ import annotations

import importlib
import types

import numpy as np

run_module = importlib.import_module("linstabpy.driver.run")


def _region_mesh_idx() -> tuple[types.SimpleNamespace, dict[str, np.ndarray], int]:
    # 2x3 grid (Ny=2, Nx=3), Fortran-order point layout, matching the fixture
    # in tests/test_router_entry.py.  Five state variables per point.
    X = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    Y = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    ngp = X.size
    idx = {
        f"{var}_j": np.arange(i * ngp, (i + 1) * ngp, dtype=np.int64)
        for i, var in enumerate(("rho", "u", "v", "w", "T"))
    }
    mesh = types.SimpleNamespace(
        ngp=ngp,
        used_ind=np.arange(ngp, dtype=np.int64),
        X=X,
        Y=Y,
        filters=None,
    )
    return mesh, idx, 5 * ngp


class _FakeMat:
    """Stand-in for the assembled operator: plain scipy path (no getInfo)."""


def _capture_resolvent_analysis(monkeypatch, calls: list[dict]):
    def _fake_resolvent_analysis(
        L0,
        omega,
        n_modes,
        W_diag,
        *,
        B_mat=None,
        C_mat=None,
        filter_fn=None,
        filter_ct_fn=None,
        solver_cfg=None,
        cfg=None,
        residuals=False,
        gamma=0.0,
    ):
        calls.append({"B_mat": B_mat, "C_mat": C_mat})
        ndof = W_diag.size
        gains = np.array([1.0])
        responses = np.zeros((ndof, 1), dtype=complex)
        forcings = np.zeros((ndof, 1), dtype=complex)
        rho_fwd = np.array([1e-12])
        rho_adj = np.array([1e-12])
        return gains, responses, forcings, rho_fwd, rho_adj

    monkeypatch.setattr(run_module, "resolvent_analysis", _fake_resolvent_analysis)
    return calls


def test_stage_run_resolvent_input_region_restricts_b_not_c(monkeypatch) -> None:
    mesh, idx, ndof = _region_mesh_idx()
    calls: list[dict] = []
    _capture_resolvent_analysis(monkeypatch, calls)
    monkeypatch.setattr(run_module, "_log", lambda message: None)
    monkeypatch.setattr(run_module, "_log_stage", lambda *a, **k: None)

    # Region excludes point index 0 (x=0,y=0) and point index 3 (x=0,y=1)
    # (the whole x=0 column) for variable 'v' only -- a genuinely partial mask.
    config = {
        "input_region": {"x": [1.0, 2.0], "vars": ["v"]},
    }
    total_bc_indices = np.array([], dtype=np.int64)

    run_module._stage_run_resolvent(
        run_resolvent=True,
        L0_resolv=_FakeMat(),
        mesh=mesh,
        idx=idx,
        config=config,
        W_diag=np.ones(ndof),
        omega_target=0.1 + 0j,
        n_modes_resolvent=1,
        total_bc_indices=total_bc_indices,
        solver_cfg={},
        cfg=None,
        heartbeat_seconds=0.0,
    )

    assert len(calls) == 1
    B_mat = calls[0]["B_mat"]
    C_mat = calls[0]["C_mat"]
    B_diag = np.asarray(B_mat.diagonal())
    C_diag = np.asarray(C_mat.diagonal())

    v_j = idx["v_j"]
    excluded = v_j[(mesh.X.ravel(order="F") < 1.0)]
    included = v_j[(mesh.X.ravel(order="F") >= 1.0)]

    assert excluded.size > 0 and included.size > 0
    # (a) B actually restricts: excluded v-DOFs are zero, included are not.
    assert np.all(B_diag[excluded] == 0.0)
    assert np.all(B_diag[included] != 0.0)
    # The same DOFs must NOT be zero in C: C is not restricted by input_region.
    assert np.all(C_diag[excluded] != 0.0)

    # (b) With only input_region given, C is the BC mask (all ones here, no
    # BC dofs), not a copy of B.
    assert not np.array_equal(B_diag, C_diag)
    assert np.all(C_diag == 1.0)


def test_stage_run_resolvent_default_path_matches_bc_only_mask(monkeypatch) -> None:
    """Regression: no region keys -> B == C == BC mask, same as before the fix."""
    mesh, idx, ndof = _region_mesh_idx()
    calls: list[dict] = []
    _capture_resolvent_analysis(monkeypatch, calls)
    monkeypatch.setattr(run_module, "_log", lambda message: None)
    monkeypatch.setattr(run_module, "_log_stage", lambda *a, **k: None)

    total_bc_indices = np.array([0, 4, 9], dtype=np.int64)
    config: dict = {}

    run_module._stage_run_resolvent(
        run_resolvent=True,
        L0_resolv=_FakeMat(),
        mesh=mesh,
        idx=idx,
        config=config,
        W_diag=np.ones(ndof),
        omega_target=0.1 + 0j,
        n_modes_resolvent=1,
        total_bc_indices=total_bc_indices,
        solver_cfg={},
        cfg=None,
        heartbeat_seconds=0.0,
    )

    B_mat = calls[0]["B_mat"]
    C_mat = calls[0]["C_mat"]
    B_diag = np.asarray(B_mat.diagonal())
    C_diag = np.asarray(C_mat.diagonal())

    expected = np.ones(ndof, dtype=float)
    expected[total_bc_indices] = 0.0
    np.testing.assert_array_equal(B_diag, expected)
    np.testing.assert_array_equal(C_diag, expected)
    assert np.array_equal(B_diag, C_diag)

