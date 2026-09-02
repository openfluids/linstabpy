# SPDX-License-Identifier: GPL-3.0-only

from __future__ import annotations

import copy
import json

import numpy as np

from linstabpy.driver.io import save_results
from linstabpy.metrics_schema import SCHEMA, _type_name, validate_metrics


def _complete_metrics_payload() -> dict[str, object]:
    return {
        "git_sha": "abc123",
        "config_md5": "0" * 32,
        "hostname": "test-host",
        "compute_mode": "cpu",
        "mpi_stack": "unknown",
        "python_version": "3.13.0",
        "numpy_version": "2.0.0",
        "scipy_version": "1.14.0",
        "petsc4py_version": "3.24.0",
        "slepc4py_version": "3.24.0",
        "petsc_arch": "arch-python",
        "mumps_version": "5.8.1",
        "mpi_impl": "MPI test implementation",
        "blas": "mkl",
        "case": "tiny-check",
        "fd_order": 6,
        "stride": 4,
        "nx": 5,
        "ny": 4,
        "ndof": 100,
        "nnz": 1234,
        "beta": 0.2,
        "lambda_over_L": 31.4,
        "omega_target": np.array([0.1 + 0.0j]),
        "stability_eps_tol": 1.0e-8,
        "stability_eps_ncv": 20,
        "provenance_factor_backend": "mumps",
        "cost_wall_total_s": 3.0,
        "cost_wall_assemble_s": 1.0,
        "cost_wall_factor_s": np.nan,
        "stability_eps_setup_seconds": 0.25,
        "stability_eps_solve_seconds": 0.75,
        "cost_wall_solve_s": 1.0,
        "cost_mpi_ranks": 1,
        "cost_gpus": 0,
        "n_threads": 1,
        "cost_peak_rss_bytes": 4096,
        "cost_peak_gpu_mem_bytes": np.nan,
        "cost_petsc_options": "",
        "stability_eps_nconv": 2,
        "stability_accepted_count": 1,
        "omega": np.array([0.1 + 0.2j]),
        "gains": np.array([2.0]),
        "stability_postcheck_residual_rel": np.array([1.0e-10]),
        "stability_postcheck_target_distance": np.array([0.01]),
        "stability_all_reject_reason_code": np.array([0], dtype=np.int32),
    }


def test_complete_dict_has_no_errors():
    assert validate_metrics(_complete_metrics_payload()) == []


def test_one_missing_key_per_required_section_surfaces_errors():
    for section, fields in SCHEMA.items():
        required = [key for key, spec in fields.items() if spec.required]
        if not required:
            continue
        key = required[0]
        payload = _complete_metrics_payload()
        payload.pop(key)

        errors = validate_metrics(payload)

        assert f"{section}.{key}: missing required key" in errors


def test_multiple_missing_keys_are_all_reported_at_once():
    payload = _complete_metrics_payload()
    for key in ("git_sha", "hostname", "cost_wall_total_s"):
        payload.pop(key)

    errors = validate_metrics(payload)

    assert "provenance.git_sha: missing required key" in errors
    assert "provenance.hostname: missing required key" in errors
    assert "cost.cost_wall_total_s: missing required key" in errors
    assert len(errors) == 3


def test_type_mismatch_is_reported():
    payload = _complete_metrics_payload()
    payload["cost_mpi_ranks"] = "one"

    errors = validate_metrics(payload)
    expected = _type_name(SCHEMA["cost"]["cost_mpi_ranks"].types)
    assert errors == [f"cost.cost_mpi_ranks: expected {expected}, got str"]


def test_quality_section_is_run_type_optional():
    payload = _complete_metrics_payload()
    for key in SCHEMA["quality"]:
        payload.pop(key, None)

    assert validate_metrics(payload) == []


def test_production_sidecar_payload_passes_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("LINSTAB2D_GPUS", "0")
    output_npz = tmp_path / "tiny_check.npz"
    payload = _complete_metrics_payload()
    provenance = {key: str(payload.pop(key)) for key in SCHEMA["provenance"]}
    # Let save_results provide the cost/provenance block through the production
    # sidecar path while the tiny payload supplies problem and quality metrics.
    for key in list(payload):
        if key.startswith("cost_"):
            payload.pop(key)
    payload.pop("cost_mpi_ranks", None)
    payload.pop("cost_gpus", None)
    payload.pop("cost_petsc_options", None)

    save_results(
        output_npz=output_npz,
        heartbeat_seconds=1.0e9,
        payload=copy.deepcopy(payload),
        peak_rss_bytes=4096,
        stage_wall={"assemble_operator_bc": 1.0, "stability": 1.0},
        provenance=provenance,
    )

    sidecar = json.loads(output_npz.with_suffix(".metrics.json").read_text())

    assert "schema_errors" not in sidecar
    assert validate_metrics(sidecar) == []
    assert sidecar["case"] == "tiny-check"
    assert sidecar["cost_peak_rss_bytes"] == 4096


def test_type_name_formats_a_single_type_and_a_union() -> None:
    assert _type_name(str) == "str"
    assert _type_name((int, float)) == "int or float"
