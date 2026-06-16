# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib
from contextlib import nullcontext

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.solvers.router.band_marking import SMOOTH, STABLE, BandMarks

router_map = importlib.import_module("linstabpy.solvers.router.resolvent_map")

CP3_KEYS = {
    "cost_wall_total_s",
    "cost_wall_assemble_s",
    "cost_wall_factor_s",
    "cost_wall_solve_s",
    "cost_peak_rss_bytes",
    "cost_peak_gpu_mem_bytes",
    "cost_mpi_ranks",
    "cost_gpus",
    "cost_petsc_options",
    "cost_hardware_tag",
    "cost_git_sha",
}


def _cost_metrics() -> dict[str, object]:
    return {
        "cost_wall_total_s": 0.0,
        "cost_wall_assemble_s": 0.0,
        "cost_wall_factor_s": np.nan,
        "cost_wall_solve_s": 0.0,
        "cost_peak_rss_bytes": 0,
        "cost_peak_gpu_mem_bytes": np.nan,
        "cost_mpi_ranks": 1,
        "cost_gpus": 0,
        "cost_petsc_options": "",
        "cost_hardware_tag": "test",
        "cost_git_sha": "test",
    }


def _install_fake_seams(monkeypatch: pytest.MonkeyPatch, *, labels: tuple[str, ...] = (STABLE,)) -> None:
    monkeypatch.setattr(router_map, "_heartbeat", lambda _stage, _cfg: nullcontext())
    monkeypatch.setattr(router_map, "_resolve_mesh", lambda _base_flow, _cfg: object())
    monkeypatch.setattr(router_map, "_compute_w_diag", lambda _mesh, _base_flow, _cfg: np.ones(3))
    monkeypatch.setattr(
        router_map,
        "_assemble_beta_operator",
        lambda _mesh, _base_flow, _beta, _cfg: router_map._AssemblyResult(
            sp.eye(3, format="csc", dtype=complex), {}, np.array([], dtype=np.int64), 0.0
        ),
    )

    def fake_marks(_L0, f_range, _cfg):
        f = np.asarray(f_range, dtype=float).reshape(-1)
        repeated = np.resize(np.asarray(labels, dtype=object), f.size)
        return BandMarks(omegas=f, labels=repeated, distances=np.ones(f.size), danger_radius=0.0)

    monkeypatch.setattr(router_map, "_mark_unstable_band", fake_marks)
    monkeypatch.setattr(
        router_map,
        "_direct_engine_result",
        lambda _L0, omega, *_args, **_kwargs: router_map._DirectPointResult(
            gain=float(omega), iterations=None, residual=0.0
        ),
    )
    monkeypatch.setattr(router_map, "_collect_cost_metrics", lambda _stage_wall: _cost_metrics())


def test_public_import_and_router_result_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_seams(monkeypatch, labels=(STABLE,))

    from linstabpy import resolvent_map
    from linstabpy.solvers.router import RouterResult, resolvent_map as router_export

    assert resolvent_map is router_map.resolvent_map
    assert router_export is router_map.resolvent_map
    assert RouterResult is router_map.RouterResult

    result = resolvent_map(object(), [0.1, 0.2], [1.0, 2.0], {"mesh": object()})

    assert isinstance(result, RouterResult)
    assert result.gain.shape == (2, 2)
    assert result.regime_label.shape == result.gain.shape
    assert result.reason_code.shape == result.gain.shape
    assert result.regime_label.tolist() == [[STABLE, STABLE], [STABLE, STABLE]]
    assert set(result.cost_metrics) >= CP3_KEYS
    np.testing.assert_allclose(result.gain, [[1.0, 2.0], [1.0, 2.0]])
    assert len(result.point_records) == 4


@pytest.mark.parametrize(
    ("beta_range", "f_range", "match"),
    [
        ([], [1.0], "beta_range"),
        ([np.nan], [1.0], "beta_range"),
        ([0.1], [], "f_range"),
        ([0.1], [np.nan], "f_range"),
    ],
)
def test_resolvent_map_validates_empty_and_nan_ranges(beta_range, f_range, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        router_map.resolvent_map(object(), beta_range, f_range, {"mesh": object(), "W_diag": np.ones(3)})


def test_regime_c_toggle_defaults_off_and_enabled_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_seams(monkeypatch, labels=(SMOOTH,))

    with caplog.at_level("INFO"):
        result = router_map.resolvent_map(object(), [0.1], [1.0], {"mesh": object()})

    assert result.regime_label.tolist() == [[SMOOTH]]
    assert result.point_records[0]["engine"] == "regime-a"
    assert "missing_engine=rational_rom" in caplog.text

    with pytest.raises(NotImplementedError, match="rational_rom.*S9"):
        router_map.resolvent_map(
            object(), [0.1], [1.0], {"mesh": object(), "enable_regime_c": True}
        )


def test_failure_tolerance_and_npz_resume_retries_only_failed_points(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install_fake_seams(monkeypatch, labels=(STABLE, STABLE))
    output_npz = tmp_path / "router_resume.npz"
    calls: list[float] = []
    fail_second = {"enabled": True}

    def flaky_direct(_L0, omega, *_args, **_kwargs):
        calls.append(float(omega))
        if fail_second["enabled"] and np.isclose(omega, 2.0):
            raise RuntimeError("forced point failure")
        return router_map._DirectPointResult(gain=float(omega), iterations=3, residual=1.0e-9)

    monkeypatch.setattr(router_map, "_direct_engine_result", flaky_direct)
    cfg = {"mesh": object(), "output_npz": output_npz}

    first = router_map.resolvent_map(object(), [0.1], [1.0, 2.0], cfg)

    assert calls == [1.0, 2.0]
    assert output_npz.exists()
    assert first.reason_code.tolist() == [[0, 1]]
    assert np.isfinite(first.gain[0, 0])
    assert np.isnan(first.gain[0, 1])
    assert first.failure_records[0]["exception_class"] == "RuntimeError"

    calls.clear()
    fail_second["enabled"] = False
    resumed = router_map.resolvent_map(object(), [0.1], [1.0, 2.0], cfg)

    assert calls == [2.0]
    np.testing.assert_allclose(resumed.gain, [[1.0, 2.0]])
    assert resumed.reason_code.tolist() == [[0, 0]]
    assert [record["status"] for record in resumed.point_records] == ["skipped", "ok"]
