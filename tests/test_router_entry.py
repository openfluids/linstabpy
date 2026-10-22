# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.solvers.router.band_marking import NEAR_POLE, SMOOTH, STABLE, BandMarks

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


def _region_mesh_idx() -> tuple[SimpleNamespace, dict[str, np.ndarray], int]:
    # Shape is (Ny, Nx).  Fortran order maps (y=0, x=1) to point index 2;
    # C order would map it to point index 1, so the single-cell test guards
    # the project layout convention directly.
    X = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    Y = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    ngp = X.size
    idx = {
        f"{var}_j": np.arange(i * ngp, (i + 1) * ngp, dtype=np.int64)
        for i, var in enumerate(("rho", "u", "v", "w", "T"))
    }
    mesh = SimpleNamespace(X=X, Y=Y, used_ind=np.arange(ngp, dtype=np.int64))
    return mesh, idx, 5 * ngp


def _install_fake_seams(
    monkeypatch: pytest.MonkeyPatch, *, labels: tuple[str, ...] = (STABLE,)
) -> None:
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

    class _FakeWalker:
        def __init__(self, **_kwargs) -> None:
            pass

        def compute(self, _L0, omega, *_args, **_kwargs):
            point = router_map._direct_engine_result(_L0, float(omega), *_args, **_kwargs)
            return SimpleNamespace(
                gain=point.gain,
                residual=0.0 if point.residual is None else point.residual,
                source=point.source,
                iterations=0 if point.iterations is None else point.iterations,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(router_map, "LineWalker", _FakeWalker)
    monkeypatch.setattr(router_map, "_collect_cost_metrics", lambda _stage_wall: _cost_metrics())


def test_public_import_and_router_result_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_seams(monkeypatch, labels=(STABLE,))

    from linstabpy import resolvent_map
    from linstabpy.solvers.router import RouterResult
    from linstabpy.solvers.router import resolvent_map as router_export

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
        router_map.resolvent_map(
            object(), beta_range, f_range, {"mesh": object(), "W_diag": np.ones(3)}
        )


def test_regime_c_toggle_defaults_off_and_enabled_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_seams(monkeypatch, labels=(SMOOTH,))

    with caplog.at_level("INFO"):
        result = router_map.resolvent_map(object(), [0.1], [1.0], {"mesh": object()})

    assert result.regime_label.tolist() == [[SMOOTH]]
    assert result.point_records[0]["engine"] == "regime-a"
    assert "missing_engine=rational_rom" in caplog.text

    with pytest.raises(NotImplementedError, match="rational_rom is not implemented yet"):
        router_map.resolvent_map(
            object(), [0.1], [1.0], {"mesh": object(), "enable_regime_c": True}
        )


def test_skip_band_marking_bypasses_evp_and_forces_regime_a(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Fake seams would label SMOOTH if band-marking ran; the override below turns
    # any call into a hard failure, so a green result proves the EVP did not run.
    _install_fake_seams(monkeypatch, labels=(SMOOTH,))

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("_mark_unstable_band called despite skip_band_marking=true")

    monkeypatch.setattr(router_map, "_mark_unstable_band", _must_not_run)

    with caplog.at_level("INFO"):
        result = router_map.resolvent_map(
            object(), [0.1], [1.0, 2.0], {"mesh": object(), "skip_band_marking": True}
        )

    # EVP skipped -> every frequency is regime-a and the certified engine still runs.
    assert result.regime_label.tolist() == [[STABLE, STABLE]]
    assert all(rec["engine"] == "regime-a" for rec in result.point_records)
    assert "band-marking SKIPPED" in caplog.text
    np.testing.assert_allclose(result.gain, [[1.0, 2.0]])


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


def _ok_direct(_L0, omega, *_args, **_kwargs):
    return router_map._DirectPointResult(gain=float(omega), iterations=3, residual=1.0e-9)


def test_store_records_the_sign_convention_and_resumes_under_the_same_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install_fake_seams(monkeypatch, labels=(STABLE, STABLE))
    monkeypatch.setattr(router_map, "_direct_engine_result", _ok_direct)
    for name in ("physical", "matlab_parity"):
        cfg = {
            "mesh": object(),
            "output_npz": tmp_path / f"{name}.npz",
            "solver": {"resolvent_sign_convention": name},
        }
        router_map.resolvent_map(object(), [0.1], [1.0, 2.0], cfg)
        with np.load(cfg["output_npz"], allow_pickle=False) as data:
            assert str(data["resolvent_sign_convention"]) == name
        # Same convention again: the store is accepted and every point is skipped.
        resumed = router_map.resolvent_map(object(), [0.1], [1.0, 2.0], cfg)
        assert [record["status"] for record in resumed.point_records] == ["skipped", "skipped"]


@pytest.mark.parametrize("stored", ["matlab_parity", None])
def test_resume_refuses_a_store_from_the_other_convention(
    monkeypatch: pytest.MonkeyPatch, tmp_path, stored: str | None
) -> None:
    _install_fake_seams(monkeypatch, labels=(STABLE, STABLE))
    monkeypatch.setattr(router_map, "_direct_engine_result", _ok_direct)
    output_npz = tmp_path / "old_map.npz"
    router_map.resolvent_map(
        object(),
        [0.1],
        [1.0, 2.0],
        {
            "mesh": object(),
            "output_npz": output_npz,
            "solver": {"resolvent_sign_convention": "matlab_parity"},
        },
    )
    if stored is None:
        # A store written before the key existed counts as matlab_parity.
        with np.load(output_npz, allow_pickle=False) as data:
            kept = {k: data[k] for k in data.files if k != "resolvent_sign_convention"}
        np.savez(output_npz, **kept)

    with pytest.raises(router_map.StoreSignConventionError) as excinfo:
        router_map.resolvent_map(
            object(), [0.1], [1.0, 2.0], {"mesh": object(), "output_npz": output_npz}
        )
    message = str(excinfo.value)
    assert "'matlab_parity'" in message
    assert "'physical'" in message
    assert "solver.resolvent_sign_convention: matlab_parity" in message
    assert "new output_npz path" in message


def _flow(scale: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(RHO=scale * np.ones((2, 2)), Ma=0.3)


def test_resume_refuses_a_store_from_another_problem_and_keeps_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install_fake_seams(monkeypatch, labels=(STABLE, STABLE))
    monkeypatch.setattr(router_map, "_direct_engine_result", _ok_direct)
    output_npz = tmp_path / "map.npz"
    cfg = {"mesh": object(), "output_npz": output_npz, "bc_penalty": 10.0, "n_modes": 1}
    router_map.resolvent_map(_flow(), [0.1], [1.0, 2.0], cfg)
    before = output_npz.read_bytes()

    # A config value, the base flow and the forcing norm each change a gain.
    with pytest.raises(router_map.StoreFingerprintError, match="cfg.bc_penalty"):
        router_map.resolvent_map(_flow(), [0.1], [1.0, 2.0], {**cfg, "bc_penalty": 20.0})
    with pytest.raises(router_map.StoreFingerprintError, match="base_flow.RHO"):
        router_map.resolvent_map(_flow(1.5), [0.1], [1.0, 2.0], cfg)
    monkeypatch.setattr(router_map, "_compute_w_diag", lambda *_a: 2.0 * np.ones(3))
    with pytest.raises(router_map.StoreFingerprintError, match="W_diag"):
        router_map.resolvent_map(_flow(), [0.1], [1.0, 2.0], cfg)
    assert output_npz.read_bytes() == before

    # A heartbeat change does not alter a gain: the store resumes.
    monkeypatch.setattr(router_map, "_compute_w_diag", lambda *_a: np.ones(3))
    same = router_map.resolvent_map(
        _flow(), [0.1], [1.0, 2.0], {**cfg, "stage_heartbeat_seconds": 5.0}
    )
    assert [r["status"] for r in same.point_records] == ["skipped", "skipped"]


def test_resume_of_a_store_without_fingerprint_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_seams(monkeypatch, labels=(STABLE, STABLE))
    monkeypatch.setattr(router_map, "_direct_engine_result", _ok_direct)
    output_npz = tmp_path / "legacy.npz"
    cfg = {"mesh": object(), "output_npz": output_npz}
    router_map.resolvent_map(_flow(), [0.1], [1.0, 2.0], cfg)
    with np.load(output_npz, allow_pickle=False) as data:
        kept = {k: data[k] for k in data.files if k != "fingerprint_json"}
    np.savez(output_npz, **kept)

    with caplog.at_level("WARNING", logger=router_map.LOGGER.name):
        resumed = router_map.resolvent_map(_flow(), [0.1], [1.0, 2.0], cfg)

    assert "cannot be verified" in caplog.text
    assert [r["status"] for r in resumed.point_records] == ["skipped", "skipped"]


def test_discount_used_per_point_is_returned_stored_and_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_seams(monkeypatch, labels=(STABLE, NEAR_POLE))
    monkeypatch.setattr(router_map, "_direct_engine_result", _ok_direct)
    output_npz = tmp_path / "discount.npz"
    cfg = {"mesh": object(), "output_npz": output_npz, "regime_b_discount_gamma": 0.5}

    with caplog.at_level("INFO", logger=router_map.LOGGER.name):
        result = router_map.resolvent_map(_flow(), [0.1], [1.0, 2.0], cfg)

    np.testing.assert_array_equal(result.discount_gamma, [[0.0, 0.5]])
    assert "discount_gamma=0.5 " in caplog.text
    with np.load(output_npz, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["discount_gamma"], [[0.0, 0.5]])
    resumed = router_map.resolvent_map(_flow(), [0.1], [1.0, 2.0], cfg)
    np.testing.assert_array_equal(resumed.discount_gamma, [[0.0, 0.5]])

    # A store from before the array existed: the discount of a NEAR_POLE point is unknown.
    with np.load(output_npz, allow_pickle=False) as data:
        kept = {k: data[k] for k in data.files if k != "discount_gamma"}
    np.savez(output_npz, **kept)
    legacy = router_map.resolvent_map(_flow(), [0.1], [1.0, 2.0], cfg)
    assert legacy.discount_gamma[0, 0] == 0.0
    assert np.isnan(legacy.discount_gamma[0, 1])


def test_region_mask_uses_fortran_order_single_variable_cell() -> None:
    mesh, idx, ndof = _region_mesh_idx()
    mask = router_map._build_region_mask(
        mesh,
        idx,
        ndof,
        {"x": [1.0, 1.0], "y": [0.0, 0.0], "vars": ["v"]},
        sp.eye(ndof, format="csc"),
        {},
        name="input_region",
    )

    active = np.flatnonzero(mask.diagonal())

    assert active.tolist() == [int(idx["v_j"][2])]
    assert active.tolist() != [int(idx["v_j"][1])]


@pytest.mark.parametrize(
    ("region", "match"),
    [
        ({"x": [10.0, 11.0], "vars": ["u"]}, "outside domain"),
        ({"x": [0.4, 0.6], "y": [0.4, 0.6], "vars": ["u"]}, "zero active points"),
        ({"vars": ["pressure"]}, "unknown variable"),
    ],
)
def test_region_mask_loud_fails_for_bad_specs(region: dict[str, object], match: str) -> None:
    mesh, idx, ndof = _region_mesh_idx()

    with pytest.raises(ValueError, match=match):
        router_map._build_region_mask(
            mesh,
            idx,
            ndof,
            region,
            sp.eye(ndof, format="csc"),
            {},
            name="output_region",
        )


def test_region_mask_logs_variable_coverage(caplog: pytest.LogCaptureFixture) -> None:
    mesh, idx, ndof = _region_mesh_idx()

    with caplog.at_level("INFO", logger=router_map.LOGGER.name):
        router_map._build_region_mask(
            mesh,
            idx,
            ndof,
            {"x": [1.0, 1.0], "y": [0.0, 1.0], "vars": ["u", "T"]},
            sp.eye(ndof, format="csc"),
            {},
            name="input_region",
        )

    assert "resolvent_map region_mask name=input_region var=u n_active=2 n_total=6" in caplog.text
    assert "resolvent_map region_mask name=input_region var=T n_active=2 n_total=6" in caplog.text
    assert "bbox_x=[1,1]" in caplog.text
    assert "bbox_y=[0,1]" in caplog.text
