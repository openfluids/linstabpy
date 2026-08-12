# SPDX-License-Identifier: GPL-3.0-only

import inspect
import json
import math
import socket

import numpy as np
import pytest

import linstabpy.utils.cost_metrics as cost_metrics_module
from linstabpy.driver.io import save_results
from linstabpy.driver.run import _STAGE_WALL, _log_stage, _stage_start
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


def test_gpu_mem_falls_back_to_nvidia_smi_when_cupy_missing(monkeypatch):
    # Regression guard for a SILENT hole. _gpu_memory_bytes used to have a
    # cupy-only body wrapped in a bare `except Exception`, so on any build without
    # cupy -- including both reference virtual environments -- every declared-GPU
    # run reported nan, indistinguishable from "no GPU declared". Assert that with
    # cupy unavailable the nvidia-smi fallback is what produces the number.
    monkeypatch.setattr(cost_metrics_module, "_gpu_memory_bytes_cupy", lambda: math.nan)
    monkeypatch.setattr(
        cost_metrics_module, "_gpu_memory_bytes_nvidia_smi", lambda: 1234.0 * 1024 * 1024
    )

    assert cost_metrics_module._gpu_memory_bytes(1) == 1234.0 * 1024 * 1024
    # Still nan when no GPU is declared -- the fallback must not invent a value.
    assert math.isnan(cost_metrics_module._gpu_memory_bytes(0))


def test_gpu_mem_prefers_cupy_when_available(monkeypatch):
    monkeypatch.setattr(cost_metrics_module, "_gpu_memory_bytes_cupy", lambda: 42.0)
    monkeypatch.setattr(
        cost_metrics_module,
        "_gpu_memory_bytes_nvidia_smi",
        lambda: pytest.fail("nvidia-smi must not run when cupy succeeded"),
    )

    assert cost_metrics_module._gpu_memory_bytes(1) == 42.0


def test_gpu_mem_is_nan_when_both_probes_fail(monkeypatch):
    # Fail-closed: if neither probe works the field must be nan, never a guess.
    monkeypatch.setattr(cost_metrics_module, "_gpu_memory_bytes_cupy", lambda: math.nan)
    monkeypatch.setattr(cost_metrics_module, "_gpu_memory_bytes_nvidia_smi", lambda: math.nan)

    assert math.isnan(cost_metrics_module._gpu_memory_bytes(1))


def test_nvidia_smi_probe_handles_missing_binary(monkeypatch):
    # nvidia-smi absent (CPU-only host) must yield nan, not an exception that
    # would take down the whole save path.
    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(cost_metrics_module.subprocess, "run", _boom)
    assert math.isnan(cost_metrics_module._gpu_memory_bytes_nvidia_smi())


def test_nvidia_smi_probe_parses_mib_to_bytes(monkeypatch):
    class _Result:
        stdout = "1887\n107\n"

    monkeypatch.setattr(cost_metrics_module.subprocess, "run", lambda *a, **k: _Result())
    # Device 0 only, matching cupy's cp.cuda.Device() default, converted MiB -> bytes.
    assert cost_metrics_module._gpu_memory_bytes_nvidia_smi() == 1887.0 * 1024 * 1024


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

    save_results(
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
            f"collect_cost_metrics must not call MPI {collective!r} (rank-0-only path -> deadlock)"
        )


def test_hardware_tag_from_env(monkeypatch):
    monkeypatch.setenv("LINSTAB2D_HARDWARE_TAG", "testbox")

    result = collect_cost_metrics({}, git_sha="test", peak_rss_bytes=0)

    assert result["cost_hardware_tag"] == "testbox"


def test_hardware_tag_default_is_unknown_not_hostname(monkeypatch):
    monkeypatch.delenv("LINSTAB2D_HARDWARE_TAG", raising=False)

    result = collect_cost_metrics({}, git_sha="test", peak_rss_bytes=0)

    assert result["cost_hardware_tag"] == "unknown"
    assert result["cost_hardware_tag"] != socket.gethostname()
