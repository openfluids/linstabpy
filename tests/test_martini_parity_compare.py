# SPDX-License-Identifier: GPL-3.0-only
"""Focused tests for Martini parity comparison helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from linstabpy.utils.jsonc import load_jsonc
from scripts.validation.martini2024.tools import (
    compare_martini_examples,
    extract_martini_to_npz,
    generate_martini_matlab_exports,
)
from scripts.validation.martini2024.tools.compare_martini_examples import (
    _compare_cavity_case,
    _compare_streaky_bl_case,
    _exit_code_from_counts,
    compare_curvilinear_protocol_case,
    compare_resolvent_case,
)
from tests._paths import MARTINI_FIXTURE_ROOT, martini_fixture
from tests._paths import REPO_ROOT as repo


def test_martini_manifest_paths_are_repo_relative() -> None:
    manifest = json.loads(
        (repo / "scripts/validation/martini2024/tools/martini_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["input_dir"] == "martini_examples"
    assert manifest["output_dir"] == "scripts/validation/martini2024/fixtures"
    assert not manifest["input_dir"].startswith("/")
    assert not manifest["output_dir"].startswith("/")
    assert all(
        entry["npz_file"].startswith("scripts/validation/martini2024/fixtures/")
        for entry in manifest["files"]
    )


def test_martini_npz_default_uses_validation_fixture_root() -> None:
    assert compare_martini_examples.ROOT == repo
    assert (
        compare_martini_examples.DEFAULT_CONFIG
        == repo / "scripts/validation/martini2024/tools/martini_cases.jsonc"
    )
    assert compare_martini_examples.DEFAULT_NPZ_DIR == MARTINI_FIXTURE_ROOT
    assert extract_martini_to_npz.ROOT == repo
    assert extract_martini_to_npz.DEFAULT_OUTPUT_DIR == MARTINI_FIXTURE_ROOT
    assert (
        extract_martini_to_npz.DEFAULT_MANIFEST
        == repo / "scripts/validation/martini2024/tools/martini_manifest.json"
    )


def test_martini_fixture_preflight_reports_required_exports(tmp_path) -> None:
    config_path = tmp_path / "cases.jsonc"
    config_path.write_text(
        """
        {
          "cases": [
            {
              "case_id": "one",
              "matlab_results_stem": "example_1_Blasius_results",
              "matlab_logs_stem": "example_1_Blasius_logs"
            },
            {
              "case_id": "two",
              "matlab_results_stem": "example_1_Blasius_results",
              "matlab_direct_stem": "example_4_Cavity_results_direct"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "fixtures"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "example_1_Blasius_results.mat").touch()
    (output_dir / "example_4_Cavity_results_direct.npz").touch()

    status = extract_martini_to_npz.fixture_status(input_dir, output_dir, config_path)

    assert status["required_stems"] == [
        "example_1_Blasius_results",
        "example_1_Blasius_logs",
        "example_4_Cavity_results_direct",
    ]
    assert status["raw_mat_missing"] == [
        "example_1_Blasius_logs",
        "example_4_Cavity_results_direct",
    ]
    assert status["npz_missing"] == [
        "example_1_Blasius_results",
        "example_1_Blasius_logs",
    ]


def test_martini_fixture_status_does_not_create_directories(tmp_path) -> None:
    config_path = tmp_path / "cases.jsonc"
    config_path.write_text(
        '{"cases": [{"matlab_results_stem": "example_1_Blasius_results"}]}',
        encoding="utf-8",
    )
    input_dir = tmp_path / "missing_raw"
    output_dir = tmp_path / "missing_fixtures"

    status = extract_martini_to_npz.fixture_status(input_dir, output_dir, config_path)

    assert status["raw_mat_missing"] == ["example_1_Blasius_results"]
    assert status["npz_missing"] == ["example_1_Blasius_results"]
    assert not input_dir.exists()
    assert not output_dir.exists()


def test_martini_matlab_export_runner_writes_expected_exports(tmp_path) -> None:
    source_dir = tmp_path / "baseflows"
    output_dir = tmp_path / "raw_exports"

    script = generate_martini_matlab_exports.build_matlab_export_script(source_dir, output_dir)

    assert "example_0_TestCurvilinearDerivative_results.mat" in script
    assert "example_1_Blasius_results.mat" in script
    assert "example_2_Jet_results.mat" in script
    assert "example_3_Streaky_BL_results_base.mat" in script
    assert "example_3_Streaky_BL_results_temporal.mat" in script
    assert "example_3_Streaky_BL_results_spatial.mat" in script
    assert "example_4_Cavity_results_direct.mat" in script
    assert "example_4_Cavity_results_adjoint.mat" in script
    assert "example_4_Cavity_results_base.mat" in script
    assert "example_5_Paraboloid_results.mat" in script
    assert "getenv('LINSTABPY_MARTINI_EXPORT_DIR')" in script
    assert "spatialAna  = true;" in script
    assert "if exist(''omega'',''var'')" in script


def test_martini_matlab_export_runner_can_select_case_blocks(tmp_path) -> None:
    source_dir = tmp_path / "baseflows"
    output_dir = tmp_path / "raw_exports"

    script = generate_martini_matlab_exports.build_matlab_export_script(
        source_dir, output_dir, only=("example4", "example5")
    )

    assert "example_4_Cavity_results_direct.mat" in script
    assert "example_5_Paraboloid_results.mat" in script
    assert "example_1_Blasius_results.mat" not in script
    assert "example_3_Streaky_BL_results_temporal.mat" not in script


def test_martini_comparator_can_fail_on_hard_fail_counts() -> None:
    hard_fail = "HARD" + "_FAIL"
    passed = "PASS"

    cases = [
        ({"status": hard_fail, "count": 1}, True, 1),
        ({"status": hard_fail, "count": 1}, False, 0),
        ({"status": passed, "count": 6}, True, 0),
    ]
    for counts, fail_on_hard_fail, expected in cases:
        actual = _exit_code_from_counts(
            {counts["status"]: counts["count"]},
            fail_on_hard_fail=fail_on_hard_fail,
        )
        if actual != expected:
            raise AssertionError((counts, fail_on_hard_fail, actual, expected))


def test_compare_resolvent_case_handles_matlab_mode_rows_and_phase() -> None:
    gains = np.array([3.0, 2.0, 1.0], dtype=float)

    responses_py = np.array(
        [
            [1.0 + 0.0j, 0.0 + 1.0j, 1.0 - 1.0j],
            [2.0 + 1.0j, 3.0 + 0.0j, 0.5 + 0.5j],
            [0.5 - 0.5j, 4.0 + 1.0j, 2.0 + 0.0j],
            [1.5 + 0.5j, 2.0 - 1.0j, 3.0 + 0.0j],
        ],
        dtype=complex,
    )
    forcings_py = np.array(
        [
            [0.5 + 0.5j, 1.0 + 0.0j, 1.5 - 0.5j],
            [1.0 + 0.0j, 0.5 + 0.5j, 2.0 + 1.0j],
            [1.5 - 0.5j, 2.0 + 0.0j, 0.5 + 0.5j],
            [2.0 + 1.0j, 1.5 - 0.5j, 1.0 + 0.0j],
        ],
        dtype=complex,
    )

    responses_mat = (responses_py * np.array([1j, -1j, np.exp(0.3j)])).T
    forcings_mat = (forcings_py * np.array([-1j, np.exp(-0.2j), 1j])).T

    report = compare_resolvent_case(
        gains_py=gains,
        responses_py=responses_py,
        forcings_py=forcings_py,
        gains_mat=gains,
        responses_mat=responses_mat,
        forcings_mat=forcings_mat,
        n_compare=3,
    )

    assert report["gains"]["max_abs_error"] == 0.0
    assert report["gains"]["max_rel_error"] == 0.0
    assert report["modes"]["max_forcing_rel_l2"] < 1e-12
    assert report["modes"]["max_response_rel_l2"] < 1e-12


def test_compare_curvilinear_protocol_case_matches_direct_fields() -> None:
    x = np.arange(9, dtype=float).reshape(3, 3)
    y = x + 10.0
    f = np.cos(x)
    df_an = -np.sin(x)
    df = df_an + 1e-14

    report = compare_curvilinear_protocol_case(
        python_npz={
            "x": x,
            "y": y,
            "f": f,
            "df": df,
            "df_an": df_an,
        },
        matlab_results={
            "results_TestCurv__x": x,
            "results_TestCurv__y": y,
            "results_TestCurv__f": f,
            "results_TestCurv__df": df,
            "results_TestCurv__df_an": df_an,
        },
        matlab_logs={"timing_total": np.array([0.5], dtype=float)},
        tolerances={"field_abs": 1e-12, "field_rel": 1e-12},
    )

    assert report["python_protocol_pass"] is True
    assert report["fields"]["df"]["max_abs_error"] == 0.0
    assert report["fields"]["df"]["rel_l2"] == 0.0


# Legacy MATLAB-export parity check. Disabled from pytest collection because
# the compact export bundle is not part of the normal test fixture set.
def legacy_test_example_0_reference_fields_match_exported_matlab_bundle() -> None:
    bundle_path = martini_fixture("example_0_TestCurvilinearDerivative_results.npz")
    if not bundle_path.exists():
        pytest.skip(f"Missing MATLAB parity bundle: {bundle_path}")
    from examples.example_0_curvilinear_validation import _compute_reference_fields

    matlab = np.load(bundle_path, allow_pickle=False)
    cfg = load_jsonc(repo / "examples/example_0_curvilinear_validation.jsonc")
    fields = _compute_reference_fields(cfg)

    np.testing.assert_allclose(fields["x"], matlab["results_TestCurv__x"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fields["y"], matlab["results_TestCurv__y"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fields["f"], matlab["results_TestCurv__f"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        fields["df_an"], matlab["results_TestCurv__df_an"], atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(fields["df"], matlab["results_TestCurv__df"], atol=1e-12, rtol=1e-12)


def test_compare_cavity_case_uses_global_best_spectrum_pairs() -> None:
    matlab_direct_omega = np.array(
        [
            1.50 + 0.10j,
            0.86 - 0.32j,
            0.45 - 0.04j,
            0.46 - 0.07j,
        ],
        dtype=complex,
    )
    python_direct_omega = np.array(
        [
            1.45 + 0.95j,
            0.855 + 0.0j,
            0.8557 - 0.3196j,
            0.4526 - 0.0365j,
            0.4605 - 0.0650j,
        ],
        dtype=complex,
    )

    ndof = 300
    direct_modes = np.zeros((ndof, 4), dtype=complex)
    adjoint_modes = np.zeros((ndof, 4), dtype=complex)
    for i in range(4):
        direct_modes[i, i] = 1.0
        adjoint_modes[i, i] = np.exp(0.1j)

    python_npz = {
        "omega": python_direct_omega,
        # Python stores adjoint eigenvalues as i * conj(omega_direct).
        "omega_adj": 1j * np.conj(matlab_direct_omega),
        "direct_modes": np.column_stack(
            [
                np.full(ndof, 5.0 + 1.0j),
                direct_modes[:, 0],
                direct_modes[:, 1],
                direct_modes[:, 2],
                direct_modes[:, 3],
            ]
        ),
        "adjoint_modes": adjoint_modes,
    }
    matlab_direct = {
        "results_Cavity_direct__omega": matlab_direct_omega,
        "results_Cavity_direct__V": direct_modes,
    }
    matlab_adjoint = {
        "results_Cavity_adjoint__omegaAdj": np.conj(matlab_direct_omega),
        "results_Cavity_adjoint__VAdj": adjoint_modes,
    }

    report = _compare_cavity_case(
        python_npz=python_npz,
        matlab_direct=matlab_direct,
        matlab_adjoint=matlab_adjoint,
        n_compare=3,
        omega_target=1.5 + 1.0j,
    )

    assert report["direct"]["selection_strategy"] == "global_best_match"
    assert report["adjoint"]["selection_strategy"] == "global_best_match"
    assert report["direct"]["spectrum_max_abs_error"] < 1e-2
    assert report["adjoint"]["spectrum_max_abs_error"] < 1e-2
    assert report["direct"]["mode_max_rel_l2"] < 1e-12
    assert report["adjoint"]["mode_max_rel_l2"] < 1e-12


def test_compare_streaky_bl_case_matches_temporal_modes_and_spatial_spectrum() -> None:
    temporal_omega = np.array(
        [0.1 + 0.2j, 0.3 + 0.4j, 0.5 + 0.6j],
        dtype=complex,
    )
    spatial_alpha = np.array(
        [0.7 + 0.1j, 0.8 + 0.2j, 0.9 + 0.3j],
        dtype=complex,
    )

    temporal_modes_py = np.array(
        [
            [1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j],
            [0.5 + 0.5j, 1.5 - 0.5j, 2.5 + 0.0j],
            [2.0 - 1.0j, 1.0 + 0.0j, 0.5 + 0.5j],
            [1.5 + 0.5j, 0.5 - 0.5j, 1.0 + 1.0j],
        ],
        dtype=complex,
    )
    temporal_modes_mat = (temporal_modes_py * np.array([1j, -1j, np.exp(0.4j)])).T

    report = _compare_streaky_bl_case(
        python_npz={
            "temporal_omega": temporal_omega,
            "temporal_modes": temporal_modes_py,
            "spatial_alpha": spatial_alpha,
        },
        matlab_temporal={
            "results_StreakyBL_temporal__omega": temporal_omega,
            "results_StreakyBL_temporal__V": temporal_modes_mat,
        },
        matlab_spatial={
            "results_StreakyBL_spatial__alphas": spatial_alpha,
        },
        n_compare=3,
        temporal_omega_key="results_StreakyBL_temporal/omega",
        temporal_modes_key="results_StreakyBL_temporal/V",
        spatial_alpha_key="results_StreakyBL_spatial/alphas",
    )

    assert report["temporal"]["selection_strategy"] == "global_best_match"
    assert report["spatial"]["selection_strategy"] == "global_best_match"
    assert report["temporal"]["spectrum_max_abs_error"] == 0.0
    assert report["temporal"]["spectrum_max_rel_error"] == 0.0
    assert report["temporal"]["mode_max_rel_l2"] < 1e-12
    assert report["spatial"]["spectrum_max_abs_error"] == 0.0
    assert report["spatial"]["spectrum_max_rel_error"] == 0.0
