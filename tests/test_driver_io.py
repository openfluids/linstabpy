# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np

import linstabpy.utils.cost_metrics as cost_metrics_module
from linstabpy.driver.io import _is_root_rank, _jsonify_metrics_value, load_results, save_results
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

STAMPED_PROVENANCE_KEYS = {
    "git_sha",
    "config_md5",
    "hostname",
    "compute_mode",
    "mpi_stack",
    "python_version",
    "numpy_version",
    "scipy_version",
    "petsc4py_version",
    "slepc4py_version",
    "petsc_arch",
    "mumps_version",
    "mpi_impl",
    "blas",
}

PROVENANCE_KEYS = {
    "stability_all_omega",
    "stability_all_postcheck_rel",
    "stability_all_target_distance",
    "stability_all_reject_reason_code",
    "stability_acceptance_conjugate_aware",
    "stability_accepted_indices",
    "stability_rejected_indices",
}


def test_jsonify_metrics_value_handles_numpy_and_complex() -> None:
    assert _jsonify_metrics_value(float("nan")) is None
    assert _jsonify_metrics_value(np.float64(float("nan"))) is None
    assert _jsonify_metrics_value(np.float32(np.nan)) is None
    assert _jsonify_metrics_value(np.float64(1.5)) == 1.5
    assert _jsonify_metrics_value(np.complex128(1.0 + 2.0j)) == {"real": 1.0, "imag": 2.0}
    assert _jsonify_metrics_value(3.0 + 4.0j) == {"real": 3.0, "imag": 4.0}
    assert _jsonify_metrics_value(np.array(5, dtype=np.int64)) == 5
    assert _jsonify_metrics_value(np.array([1.0, float("nan")])) == [1.0, None]
    assert _jsonify_metrics_value((1.0 + 0.0j, 2)) == [{"real": 1.0, "imag": 0.0}, 2]


def _assert_npz_value_unchanged(actual, expected):
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype
    if actual_array.dtype.kind in {"S", "U", "O"}:
        np.testing.assert_array_equal(actual_array, expected_array)
    elif actual_array.dtype.kind == "c":
        np.testing.assert_allclose(actual_array, expected_array, rtol=0.0, atol=0.0)
    else:
        np.testing.assert_array_equal(actual_array, expected_array)


def test_save_and_load_results_preserve_provenance_and_cost_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("LINSTAB2D_GPUS", "0")
    monkeypatch.delenv("LINSTAB2D_HARDWARE_TAG", raising=False)
    output_npz = tmp_path / "roundtrip.npz"
    payload = {
        "stability_all_omega": np.array([1.0 + 2.0j, -1.0 - 2.0j]),
        "stability_all_postcheck_rel": np.array([1.0e-8, 2.0e-8]),
        "stability_all_target_distance": np.array([0.1, 0.2]),
        "stability_all_reject_reason_code": np.array([0, 3], dtype=np.int32),
        "stability_acceptance_conjugate_aware": np.bool_(True),
        "stability_accepted_indices": np.array([0], dtype=np.int32),
        "stability_rejected_indices": np.array([1], dtype=np.int32),
    }
    stage_wall = {
        "assemble_operator_bc": 1.25,
        "stability": 2.5,
        "resolvent": 3.75,
    }

    save_results(
        output_npz=output_npz,
        heartbeat_seconds=1e9,
        payload=payload,
        peak_rss_bytes=123456,
        stage_wall=stage_wall,
    )

    with load_results(output_npz) as data:
        assert set(data.files) == PROVENANCE_KEYS | COST_KEYS | STAMPED_PROVENANCE_KEYS
        for key, expected in payload.items():
            _assert_npz_value_unchanged(data[key], expected)

        assert data["cost_wall_total_s"].item() == sum(stage_wall.values())
        assert data["cost_wall_assemble_s"].item() == stage_wall["assemble_operator_bc"]
        assert data["cost_wall_solve_s"].item() == stage_wall["stability"] + stage_wall["resolvent"]
        assert math.isnan(data["cost_wall_factor_s"].item())
        assert data["cost_peak_rss_bytes"].item() == 123456
        assert math.isnan(data["cost_peak_gpu_mem_bytes"].item())
        assert data["cost_mpi_ranks"].item() >= 1
        assert data["cost_gpus"].item() == 0
        assert isinstance(data["cost_petsc_options"].item(), str)
        assert data["cost_hardware_tag"].item() == "unknown"
        assert isinstance(data["cost_git_sha"].item(), str)


def test_is_root_rank_when_petsc_is_absent() -> None:
    assert _is_root_rank() is True


def test_cost_metrics_without_petsc_or_gpu(monkeypatch) -> None:
    monkeypatch.setenv("LINSTAB2D_GPUS", "0")
    monkeypatch.delenv("LINSTAB2D_HARDWARE_TAG", raising=False)
    result = collect_cost_metrics({}, git_sha="deadbeef", peak_rss_bytes=1)
    assert result["cost_mpi_ranks"] == 1
    assert result["cost_petsc_options"] == ""
    assert math.isnan(result["cost_peak_gpu_mem_bytes"])
    assert result["cost_gpus"] == 0
    assert math.isnan(cost_metrics_module._gpu_memory_bytes_cupy())
    assert math.isnan(cost_metrics_module._gpu_memory_bytes_nvidia_smi())
    assert math.isnan(cost_metrics_module._gpu_memory_bytes(1))


def test_nvidia_smi_probe_empty_and_unparseable(monkeypatch) -> None:
    class _Empty:
        stdout = "\n"

    monkeypatch.setattr(cost_metrics_module.subprocess, "run", lambda *a, **k: _Empty())
    assert math.isnan(cost_metrics_module._gpu_memory_bytes_nvidia_smi())

    class _Bad:
        stdout = "not-a-number\n"

    monkeypatch.setattr(cost_metrics_module.subprocess, "run", lambda *a, **k: _Bad())
    assert math.isnan(cost_metrics_module._gpu_memory_bytes_nvidia_smi())
