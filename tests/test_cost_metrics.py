# SPDX-License-Identifier: GPL-3.0-only

import inspect
import json
import math

import numpy as np

import linstabpy.utils.cost_metrics as cost_metrics_module
from linstabpy.results_runtime import _STAGE_WALL, _log_stage, _stage_start, _stage_write_npz
from linstabpy.utils.cost_metrics import collect_cost_metrics


COST_KEYS = {
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


def test_collect_cost_metrics_has_full_schema():
    result = collect_cost_metrics(
        {"assemble_operator_bc": 1.5, "stability": 2.0, "resolvent": 0.5},
        git_sha="abc",
        peak_rss_bytes=12345,
    )

    assert set(result) == COST_KEYS
    assert result["cost_wall_assemble_s"] == 1.5
    assert result["cost_wall_solve_s"] == 2.5
    assert result["cost_wall_total_s"] == 4.0
    assert result["cost_peak_rss_bytes"] == 12345
    assert result["cost_mpi_ranks"] >= 1
    assert isinstance(result["cost_petsc_options"], str)
    assert result["cost_git_sha"] == "abc"


def test_cpu_only_gpu_mem_is_nan(monkeypatch):
    monkeypatch.setenv("LINSTAB2D_GPUS", "0")

    result = collect_cost_metrics({}, git_sha="test", peak_rss_bytes=0)

    assert math.isnan(result["cost_peak_gpu_mem_bytes"])
    assert result["cost_gpus"] == 0


def test_factor_is_nan():
    result = collect_cost_metrics({}, git_sha="test", peak_rss_bytes=0)

    assert math.isnan(result["cost_wall_factor_s"])


def test_npz_and_sidecar_agreement(tmp_path):
    _STAGE_WALL.clear()
    _STAGE_WALL.update(
        {
            "assemble_operator_bc": 1.5,
            "stability": 2.0,
            "resolvent": 0.5,
        }
    )
    output_npz = tmp_path / "t.npz"

    _stage_write_npz(
        output_npz=output_npz,
        heartbeat_seconds=1e9,
        payload={"omega": np.zeros(2)},
        peak_rss_bytes=99999,
    )

    with np.load(output_npz) as npz:
        sidecar = json.load(open(output_npz.with_suffix(".metrics.json")))
        assert "omega" in npz.files
        assert npz["cost_peak_rss_bytes"].item() == 99999
        for key in COST_KEYS:
            assert key in npz.files
            assert key in sidecar
            npz_value = npz[key].item()
            sidecar_value = sidecar[key]
            if isinstance(npz_value, float) and math.isnan(npz_value):
                assert sidecar_value is None
            elif isinstance(npz_value, float):
                assert sidecar_value == npz_value
            else:
                assert sidecar_value == npz_value


def test_log_stage_populates_stage_wall():
    # Regression guard: _log_stage MUST accumulate elapsed wall time into
    # _STAGE_WALL, otherwise every cost_wall_*_s in the saved artifacts is 0/nan.
    # (This wiring was silently missing on a prior commit; the unit tests passed
    # because they injected a hand-built stage_wall dict. This test exercises the
    # REAL wiring.)
    _STAGE_WALL.clear()
    start = _stage_start()  # (perf_counter, peak_rss) tuple branch
    _log_stage("assemble_operator_bc", start)
    _log_stage("assemble_operator_bc", start)  # second call must accumulate
    assert "assemble_operator_bc" in _STAGE_WALL
    assert _STAGE_WALL["assemble_operator_bc"] >= 0.0
    # float (non-tuple) branch must also record.
    _log_stage("stability", float(start[0]))
    assert "stability" in _STAGE_WALL
    _STAGE_WALL.clear()


def test_collect_cost_metrics_has_no_mpi_collective():
    # Regression guard: collect_cost_metrics runs inside the rank-0-only save
    # block, so it must contain NO MPI collective (allreduce/bcast/barrier) -- one
    # there deadlocks every multi-rank run. The cross-rank reduction lives in the
    # collective region of compute(); peak_rss_bytes is threaded in. See
    # feedback_mpi_collective_rank0.
    src = inspect.getsource(cost_metrics_module.collect_cost_metrics)
    # Match CALL syntax (".allreduce(") not the bare word, so the docstring/comment
    # explaining why there is no collective does not trip the guard.
    for collective in (
        ".allreduce(",
        ".bcast(",
        ".barrier(",
        ".gather(",
        ".scatter(",
        ".reduce(",
        ".Allreduce(",
        ".Bcast(",
    ):
        assert collective not in src, (
            f"collect_cost_metrics must not call MPI {collective!r} "
            "(rank-0-only path -> deadlock)"
        )


def test_hardware_tag_from_env(monkeypatch):
    monkeypatch.setenv("LINSTAB2D_HARDWARE_TAG", "testbox")

    result = collect_cost_metrics({}, git_sha="test", peak_rss_bytes=0)

    assert result["cost_hardware_tag"] == "testbox"
