# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
from pathlib import Path

import numpy as np

from examples.song.reference_paths import BASEFLOW_NPZ, RUN_FD4_N30_YAML
from examples.song.song import (
    _align_mode_to_reference,
    _match_selected_modes,
    _resolve_baseflow_input_options,
    _resolve_stability_request_count,
    _select_reference_mode_indices,
    _stability_archive_fields,
    _summarize_rank_metrics,
)
from linstabpy.baseflow import load_case_fields
from linstabpy.utils.yaml_io import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SONG_CONFIG_PATH = RUN_FD4_N30_YAML


def test_song2023_fd4_config_matches_supp_anchor() -> None:
    cfg = load_config(SONG_CONFIG_PATH)

    assert cfg["baseflow_input"]["transpose_input"] is True
    assert cfg["baseflow_input"]["zero_spanwise"] is False
    assert cfg["baseflow_preprocess"]["enabled"] is False
    assert cfg["fd_order"] == 4
    assert cfg["n_modes_stability"] == 30
    assert cfg["stability_search_extra_modes"] == 0
    assert cfg["beta"] == 19.2
    assert cfg["numerics"]["operator_dissipation"]["enabled"] is False
    assert cfg["numerics"]["operator_dissipation"]["epsilon"] == 0.0

    options = _resolve_baseflow_input_options(cfg)
    assert options == {"transpose_input": True, "zero_spanwise": False}
    assert _resolve_stability_request_count(cfg) == 30


def test_song2023_loader_transpose_matches_mesh_convention() -> None:
    X, Y, *_ = load_case_fields(
        BASEFLOW_NPZ,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        transpose_input=True,
        preprocess={"enabled": False},
    )

    assert X[0, -1] > X[0, 0]
    assert abs(X[-1, 0] - X[0, 0]) < 1e-12
    assert Y[-1, 0] > Y[0, 0]
    assert abs(Y[0, -1] - Y[0, 0]) < 1e-12


def test_song2023_reference_mode_selection_prefers_steady_and_oscillatory_pair() -> None:
    matlab_omega = np.asarray(
        [
            0.0 + 0.0589j,
            0.0 - 0.1158j,
            -0.3282 - 0.2984j,
            0.3282 - 0.2984j,
            0.0 - 0.5426j,
        ],
        dtype=complex,
    )
    python_omega = np.asarray(
        [
            0.0 + 0.0573j,
            0.0 - 0.1085j,
            -0.3265 - 0.2924j,
            0.3265 - 0.2924j,
        ],
        dtype=complex,
    )

    selected = _select_reference_mode_indices(matlab_omega)
    assert selected == [0, 3]

    matched = _match_selected_modes(matlab_omega, python_omega)
    assert matched == [(0, 0), (3, 3)]


def test_song2023_phase_alignment_removes_simple_sign_flip() -> None:
    reference = np.asarray([1.0 + 0.0j, -2.0 + 0.0j, 3.0 + 0.0j])
    candidate = -reference
    aligned = _align_mode_to_reference(candidate, reference)
    assert np.allclose(aligned, reference)


def test_song2023_rank_metric_summary_uses_max_rank_wall_time() -> None:
    summary = _summarize_rank_metrics(
        [
            [("assemble", 10.0, 100.0), ("solve", 20.0, 200.0), ("save", 1.0, 10.0)],
            [("assemble", 12.0, 110.0), ("solve", 18.0, 150.0)],
        ],
        [31.0, 30.0],
        [800.0, 700.0],
    )
    assert summary["total_time_max"] == 31.0
    assert summary["peak_memory_max"] == 800.0
    assert summary["peak_memory_sum"] == 1500.0
    assert summary["step_summary"]["assemble"]["time_max"] == 12.0
    assert summary["step_summary"]["solve"]["memory_max"] == 200.0
    assert summary["step_summary"]["save"]["count"] == 1


def test_song2023_stability_archive_fields_trim_to_saved_modes() -> None:
    archive = _stability_archive_fields(
        {
            "method": "slepc",
            "nev_requested": 24,
            "ncv_used": 48,
            "nconv": 22,
            "nextract_total": 22,
            "extract_all_converged": False,
            "eps_iterations": 37,
            "eps_reason": "CONVERGED_TOL",
            "ksp_reason": "CONVERGED_RTOL",
            "conv_test": "relative",
            "eps_tol": 1e-10,
            "eps_max_it": 500,
            "eps_setup_seconds": 3.5,
            "eps_solve_seconds": 12.25,
            "st_solver": "mumps",
            "true_residual": True,
            "residual_error_type": "relative",
            "residual_estimates": np.array([1.0, 2.0, 3.0, 4.0]),
            "residual_absolute": np.array([5.0, 6.0, 7.0, 8.0]),
            "residual_relative": np.array([9.0, 10.0, 11.0, 12.0]),
            "residual_backward": np.array([13.0, 14.0, 15.0, 16.0]),
            "leading_weighted_norm_error": 1.0e-12,
            "weighted_norm_tol": 1.0e-10,
        },
        n_keep=2,
    )

    assert archive["stability_eps_nev"] == 24
    assert archive["stability_eps_nconv"] == 22
    assert archive["stability_eps_iterations"] == 37
    assert archive["stability_true_residual"] is True
    assert str(archive["stability_method"]) == "slepc"
    assert str(archive["stability_eps_reason"]) == "CONVERGED_TOL"
    assert str(archive["stability_st_solver"]) == "mumps"
    assert np.allclose(archive["stability_residual_estimates"], [1.0, 2.0])
    assert np.allclose(archive["stability_residual_absolute"], [5.0, 6.0])
    assert np.allclose(archive["stability_residual_relative"], [9.0, 10.0])
    assert np.allclose(archive["stability_residual_backward"], [13.0, 14.0])
