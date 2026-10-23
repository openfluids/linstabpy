# SPDX-License-Identifier: GPL-3.0-only
"""Check that resolvent_map walks beta lines and does not call the RSVD sweep."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.solvers.router.band_marking import NEAR_POLE, STABLE, BandMarks

router_map = importlib.import_module("linstabpy.solvers.router.resolvent_map")


def _install_small_router_seams(monkeypatch: pytest.MonkeyPatch) -> tuple[list[float], list[float]]:
    direct_calls: list[float] = []
    walk_calls: list[float] = []
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
        lambda _L0, f_range, _cfg, _cfg_obj=None: BandMarks(
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

    class FakeWalker:
        def __init__(self, **_kwargs) -> None:
            self.n_factor = 0

        def compute(self, _L0, omega, *_args, **_kwargs):
            walk_calls.append(float(omega))
            source = "factor" if self.n_factor == 0 else "walk"
            if source == "factor":
                self.n_factor = 1
            return SimpleNamespace(
                gain=direct_gain(float(omega)),
                residual=1.0e-12,
                source=source,
                iterations=0 if source == "factor" else 4,
            )

        def close(self) -> None:
            return None

    def _must_not_sweep(*_args, **_kwargs):
        raise AssertionError("sweep_stable_regime must not run on the map path")

    monkeypatch.setattr(router_map, "_direct_engine_result", fake_direct)
    monkeypatch.setattr(router_map, "LineWalker", FakeWalker)
    monkeypatch.setattr(router_map, "sweep_stable_regime", _must_not_sweep, raising=False)
    monkeypatch.setattr(
        router_map, "_collect_cost_metrics", lambda _stage_wall: {"cost_wall_total_s": 0.0}
    )
    return direct_calls, walk_calls


def test_map_walks_regime_a_without_recycling_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    direct_calls, walk_calls = _install_small_router_seams(monkeypatch)
    beta = np.array([0.1, 0.2])
    omega = np.array([0.2, 0.4, 0.6, 0.8])
    cfg = {"mesh": object(), "output_npz": tmp_path / "walk.npz"}

    result = router_map.resolvent_map(object(), beta, omega, cfg)

    assert "recycling" not in cfg
    assert all(record["engine"] != "stable-sweep" for record in result.point_records)
    assert walk_calls == [0.2, 0.6, 0.8, 0.2, 0.6, 0.8]
    assert direct_calls == [0.4, 0.4]
    assert [record["engine"] for record in result.point_records].count("regime-a") == 6
    assert [record["engine"] for record in result.point_records].count("regime-b") == 2
    sources = [record.get("walk_source") for record in result.point_records]
    assert sources.count("factor") == 2
    assert sources.count("walk") == 4
    assert sources.count("direct") == 2
    np.testing.assert_allclose(result.gain[0], [10.2, 10.4, 10.6, 10.8])
    assert (tmp_path / "walk.npz").exists()


def test_amortized_flag_does_not_enable_rsvd_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_small_router_seams(monkeypatch)
    result = router_map.resolvent_map(
        object(),
        np.array([0.1]),
        np.array([0.2, 0.4, 0.6, 0.8]),
        {
            "mesh": object(),
            "output_npz": tmp_path / "amortized.npz",
            "analysis": {"amortized_stable_sweep": True},
        },
    )
    assert all(record["engine"] != "stable-sweep" for record in result.point_records)


def test_single_frequency_stays_on_direct_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    created: list[int] = []
    direct_calls, _walk_calls = _install_small_router_seams(monkeypatch)

    class _MustNotBuild:
        def __init__(self, **_kwargs) -> None:
            created.append(1)

    monkeypatch.setattr(router_map, "LineWalker", _MustNotBuild)
    monkeypatch.setattr(
        router_map,
        "_mark_unstable_band",
        lambda _L0, f_range, _cfg, _cfg_obj=None: BandMarks(
            omegas=np.asarray(f_range, dtype=float),
            labels=np.array([STABLE], dtype=object),
            distances=np.ones(1, dtype=float),
            danger_radius=0.0,
        ),
    )
    result = router_map.resolvent_map(
        object(),
        np.array([0.1, 0.2]),
        np.array([0.5]),
        {"mesh": object(), "output_npz": tmp_path / "one_f.npz"},
    )
    assert created == []
    assert direct_calls == [0.5, 0.5]
    assert all(record["walk_source"] == "direct" for record in result.point_records)
    np.testing.assert_allclose(result.gain, [[10.5], [10.5]])
