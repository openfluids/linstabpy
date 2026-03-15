from __future__ import annotations

from pathlib import Path

import numpy as np

from linstab2d.utils.jsonc import load_jsonc

from validation.song_2023.baseflow_loader import load_my_case
from validation.song_2023 import run_song_2023_matrix
from validation.song_2023.run_song_2023_order_preflight import _memory_limit_hit
from validation.song_2023.run_song_2023_scaling import _should_stop
from validation.song_2023.song_2023 import (
    _align_mode_to_reference,
    _match_selected_modes,
    _resolve_baseflow_input_options,
    _resolve_stability_request_count,
    _select_reference_mode_indices,
    _summarize_rank_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SONG_CASE_DIR = REPO_ROOT / "validation" / "song_2023"
SONG_CONFIG_PATH = SONG_CASE_DIR / "song_2023.jsonc"


def test_song2023_default_config_uses_transposed_filtered_input() -> None:
    cfg = load_jsonc(SONG_CONFIG_PATH)

    assert cfg["baseflow_input"]["transpose_input"] is True
    assert cfg["baseflow_input"]["zero_spanwise"] is False
    assert cfg["baseflow_preprocess"]["enabled"] is True
    assert cfg["stability_search_extra_modes"] == 4

    options = _resolve_baseflow_input_options(cfg)
    assert options == {"transpose_input": True, "zero_spanwise": False}
    assert _resolve_stability_request_count(cfg) == 24


def test_song2023_loader_transpose_matches_mesh_convention() -> None:
    X, Y, *_ = load_my_case(
        SONG_CASE_DIR,
        Re=1.0e5,
        Ma=2.15,
        Pr=0.72,
        kappa=1.4,
        T_0=153.33,
        transpose_input=True,
        preprocess_cfg={"enabled": False},
    )

    # Repo mesh convention is (ny, nx): x varies along columns, y along rows.
    assert X[0, -1] > X[0, 0]
    assert abs(X[-1, 0] - X[0, 0]) < 1e-12
    assert Y[-1, 0] > Y[0, 0]
    assert abs(Y[0, -1] - Y[0, 0]) < 1e-12


def test_song2023_matrix_runner_clears_and_collects_outputs(tmp_path, monkeypatch) -> None:
    song_dir = tmp_path / "song"
    run_dir = tmp_path / "run"
    song_dir.mkdir()
    run_dir.mkdir()

    outputs = tuple(
        song_dir / name
        for name in (
            "song_2023.npz",
            "song_2023.log",
            "song_2023.png",
            "song_2023_baseflow.png",
            "song_2023_support_direct_modes.png",
            "song_2023_resolvent.png",
        )
    )
    monkeypatch.setattr(run_song_2023_matrix, "SONG_DIR", song_dir)
    monkeypatch.setattr(run_song_2023_matrix, "CANONICAL_OUTPUTS", outputs)

    for path in outputs:
        path.write_bytes(b"test")
    run_song_2023_matrix._clear_canonical_outputs()
    assert not any(path.exists() for path in outputs)

    for path in outputs:
        path.write_bytes(b"test")
    target_npz, target_log = run_song_2023_matrix._collect_outputs(run_dir)
    assert target_npz == run_dir / "song_2023.npz"
    assert target_log == run_dir / "song_2023.log"
    assert target_npz.exists()
    assert target_log.exists()
    assert (run_dir / "song_2023.png").exists()
    assert (run_dir / "song_2023_baseflow.png").exists()
    assert (run_dir / "song_2023_support_direct_modes.png").exists()
    assert (run_dir / "song_2023_resolvent.png").exists()
    assert not any(path.exists() for path in outputs)


def test_song2023_reference_mode_selection_prefers_steady_and_oscillatory_pair() -> None:
    matlab_omega = np.asarray([
        0.0 + 0.0589j,
        0.0 - 0.1158j,
        -0.3282 - 0.2984j,
        0.3282 - 0.2984j,
        0.0 - 0.5426j,
    ], dtype=complex)
    python_omega = np.asarray([
        0.0 + 0.0573j,
        0.0 - 0.1085j,
        -0.3265 - 0.2924j,
        0.3265 - 0.2924j,
    ], dtype=complex)

    selected = _select_reference_mode_indices(matlab_omega)
    assert selected == [0, 3]

    matched = _match_selected_modes(matlab_omega, python_omega)
    assert matched == [(0, 0), (3, 3)]


def test_song2023_phase_alignment_removes_simple_sign_flip() -> None:
    reference = np.asarray([1.0 + 0.0j, -2.0 + 0.0j, 3.0 + 0.0j])
    candidate = -reference
    aligned = _align_mode_to_reference(candidate, reference)
    assert np.allclose(aligned, reference)


def test_song2023_order_preflight_memory_guard() -> None:
    assert _memory_limit_hit({"peak_memory_mib": 12001.0}, 12000.0) is True
    assert _memory_limit_hit({"peak_memory_mib": 11999.0}, 12000.0) is False
    assert _memory_limit_hit({}, 12000.0) is False
    assert _memory_limit_hit({"peak_memory_mib": 12001.0}, None) is False


def test_song2023_scaling_stop_logic() -> None:
    guard = {
        "stop_on_peak_memory_mib_max_rank": 12000.0,
        "stop_on_peak_memory_mib_sum_ranks": 14000.0,
        "stop_on_swap_delta_mib": 256.0,
    }
    assert (
        _should_stop(
            {
                "ranks": 4,
                "swap_delta_mib": 10.0,
                "summary": {
                    "peak_memory_mib": 1000.0,
                    "peak_memory_mib_sum_ranks": 2000.0,
                    "distributed_matrix": "False",
                },
            },
            guard,
        )
        == "distributed proof missing for ranks > 1"
    )
    assert "peak_memory_mib=" in _should_stop(
        {
            "ranks": 1,
            "swap_delta_mib": 10.0,
            "summary": {
                "peak_memory_mib": 13000.0,
                "peak_memory_mib_sum_ranks": 13000.0,
                "distributed_matrix": "False",
            },
        },
        guard,
    )


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
