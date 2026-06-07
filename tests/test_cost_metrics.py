# SPDX-License-Identifier: GPL-3.0-only

import json
import math

import numpy as np

from linstabpy.results_runtime import _STAGE_WALL, _stage_write_npz
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
    )

    assert set(result) == COST_KEYS
    assert result["cost_wall_assemble_s"] == 1.5
    assert result["cost_wall_solve_s"] == 2.5
    assert result["cost_wall_total_s"] == 4.0
    assert result["cost_mpi_ranks"] >= 1
    assert isinstance(result["cost_petsc_options"], str)
    assert result["cost_git_sha"] == "abc"


def test_cpu_only_gpu_mem_is_nan(monkeypatch):
    monkeypatch.setenv("LINSTAB2D_GPUS", "0")

    result = collect_cost_metrics({}, git_sha="test")

    assert math.isnan(result["cost_peak_gpu_mem_bytes"])
    assert result["cost_gpus"] == 0


def test_factor_is_nan():
    result = collect_cost_metrics({}, git_sha="test")

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
    )

    with np.load(output_npz) as npz:
        sidecar = json.load(open(output_npz.with_suffix(".metrics.json")))
        assert "omega" in npz.files
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


def test_hardware_tag_from_env(monkeypatch):
    monkeypatch.setenv("LINSTAB2D_HARDWARE_TAG", "testbox")

    result = collect_cost_metrics({}, git_sha="test")

    assert result["cost_hardware_tag"] == "testbox"
