# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np

from linstabpy.driver.io import load_results, save_results

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
