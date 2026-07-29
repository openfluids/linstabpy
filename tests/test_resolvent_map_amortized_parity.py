# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.solvers.router.band_marking import NEAR_POLE, STABLE, BandMarks
from linstabpy.solvers.router.stable_sweep import SweepResult

router_map = importlib.import_module("linstabpy.solvers.router.resolvent_map")


def _install_small_router_seams(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    direct_calls: list[float] = []
    L0 = sp.diags([2.0 + 0.1j, 3.0 + 0.2j, 5.0 + 0.3j], format="csc", dtype=complex)

    monkeypatch.setattr(router_map, "_heartbeat", lambda _stage, _cfg: nullcontext())
    monkeypatch.setattr(router_map, "_resolve_mesh", lambda _base_flow, _cfg: object())
    monkeypatch.setattr(router_map, "_compute_w_diag", lambda _mesh, _base_flow, _cfg: np.ones(3))
    monkeypatch.setattr(
        router_map,
        "_assemble_beta_operator",
        lambda _mesh, _base_flow, _beta, _cfg: router_map._AssemblyResult(
            L0, {}, np.array([], dtype=np.int64), 0.0
        ),
    )
    monkeypatch.setattr(
        router_map,
        "_mark_unstable_band",
        lambda _L0, f_range, _cfg: BandMarks(
            omegas=np.asarray(f_range, dtype=float),
            labels=np.array([STABLE, NEAR_POLE, STABLE, STABLE], dtype=object),
            distances=np.ones(4, dtype=float),
            danger_radius=0.0,
        ),
    )
    monkeypatch.setattr(
        router_map, "_forcing_response_mats", lambda *_args, **_kwargs: (None, None)
    )

    def direct_gain(omega: float) -> float:
        return float(10.0 + omega)

    def fake_direct(_L0, omega, *_args, **_kwargs):
        direct_calls.append(float(omega))
        return router_map._DirectPointResult(
            gain=direct_gain(float(omega)), iterations=7, residual=2.0e-10
        )

    def fake_choose(omega_range, band_marks, cfg):
        assert band_marks is not None
        return np.asarray([omega_range[0], omega_range[-1]], dtype=complex), ()

    def fake_sweep(_L0, omega_block, anchors, cfg):
        omegas = np.asarray(omega_block, dtype=complex).reshape(-1)
        assert np.asarray(anchors).size == 2
        return SweepResult(
            omega=omegas,
            gain=np.asarray([direct_gain(float(omega.real)) * (1.0 + 1.0e-8) for omega in omegas]),
            iterations=np.full(omegas.size, 3, dtype=int),
            residuals=np.full(omegas.size, 5.0e-9, dtype=float),
            anchor_indices=np.zeros(omegas.size, dtype=int),
            responses=np.zeros((3, omegas.size), dtype=complex),
            forcings=np.zeros((3, omegas.size), dtype=complex),
            certification_reports=tuple(
                SimpleNamespace(certified=True) for _ in range(omegas.size)
            ),
            metrics={"factorize_calls": 2, "gmres_iterations_total_actual": 9},
        )

    monkeypatch.setattr(router_map, "_direct_engine_result", fake_direct)
    monkeypatch.setattr(router_map, "choose_anchors_adaptive", fake_choose, raising=False)
    monkeypatch.setattr(router_map, "sweep_stable_regime", fake_sweep, raising=False)
    monkeypatch.setattr(
        router_map, "_collect_cost_metrics", lambda _stage_wall: {"cost_wall_total_s": 0.0}
    )
    return direct_calls


def test_amortized_stable_sweep_matches_direct_oracle_and_preserves_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    direct_calls = _install_small_router_seams(monkeypatch)
    beta = np.array([0.1, 0.2])
    omega = np.array([0.2, 0.4, 0.6, 0.8])
    direct_cfg = {"mesh": object(), "output_npz": tmp_path / "direct.npz"}
    amortized_cfg = {
        "mesh": object(),
        "output_npz": tmp_path / "amortized.npz",
        "analysis": {"amortized_stable_sweep": True},
    }

    direct = router_map.resolvent_map(object(), beta, omega, direct_cfg)
    direct_call_count = len(direct_calls)
    direct_calls.clear()

    amortized = router_map.resolvent_map(object(), beta, omega, amortized_cfg)

    np.testing.assert_allclose(amortized.gain, direct.gain, rtol=1.0e-4, atol=0.0)
    np.testing.assert_array_equal(amortized.regime_label, direct.regime_label)
    np.testing.assert_array_equal(amortized.reason_code, direct.reason_code)
    assert direct_call_count == beta.size * omega.size
    assert direct_calls == [0.4, 0.4]
    assert [record["engine"] for record in amortized.point_records].count("stable-sweep") == 6
    assert [record["engine"] for record in amortized.point_records].count("regime-b") == 2
    assert (tmp_path / "direct.npz").exists()
    assert (tmp_path / "amortized.npz").exists()
    for record in amortized.point_records:
        if record["engine"] == "stable-sweep":
            assert record["status"] == "ok"
            assert record["iterations"] == 3
            assert record["residual"] == pytest.approx(5.0e-9)
            assert record["sweep_metrics"]["factorize_calls"] == 2
