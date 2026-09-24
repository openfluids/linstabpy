# SPDX-License-Identifier: GPL-3.0-only
"""Focused tests for Martini parity comparison helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.linstab2d import (
    compare_martini_examples,
    extract_martini_to_npz,
    generate_martini_matlab_exports,
)
from examples.linstab2d.compare_martini_examples import (
    _compare_cavity_case,
    _compare_streaky_bl_case,
    _exit_code_from_counts,
    compare_curvilinear_protocol_case,
    compare_resolvent_case,
    load_matlab_bundle,
)
from linstabpy.utils.yaml_io import load_config
from tests._paths import MARTINI_FIXTURE_ROOT, martini_fixture
from tests._paths import REPO_ROOT as repo


def test_martini_manifest_paths_are_repo_relative() -> None:
    manifest = json.loads(
        (repo / "examples/linstab2d/martini_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["input_dir"] == "martini_examples"
    assert manifest["output_dir"] == "examples/linstab2d/LinStab2DResults"
    assert not manifest["input_dir"].startswith("/")
    assert not manifest["output_dir"].startswith("/")
    assert all(entry["npz_file"].startswith("examples/linstab2d/") for entry in manifest["files"])


def test_martini_npz_default_uses_validation_fixture_root() -> None:
    assert compare_martini_examples.ROOT == repo
    assert compare_martini_examples.DEFAULT_CONFIG == repo / "examples/linstab2d/martini_cases.yaml"
    assert compare_martini_examples.DEFAULT_NPZ_DIR == MARTINI_FIXTURE_ROOT
    assert extract_martini_to_npz.ROOT == repo
    assert extract_martini_to_npz.DEFAULT_OUTPUT_DIR == MARTINI_FIXTURE_ROOT
    assert (
        extract_martini_to_npz.DEFAULT_MANIFEST == repo / "examples/linstab2d/martini_manifest.json"
    )


def test_martini_fixture_preflight_reports_required_exports(tmp_path) -> None:
    config_path = tmp_path / "cases.yaml"
    config_path.write_text(
        """
        cases:
          - case_id: one
            matlab_results_stem: example_1_Blasius_results
          - case_id: two
            matlab_direct_stem: example_4_Cavity_results_direct
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
        "example_4_Cavity_results_direct",
    ]
    assert status["raw_mat_missing"] == [
        "example_4_Cavity_results_direct",
    ]
    assert status["npz_missing"] == [
        "example_1_Blasius_results",
    ]


def test_martini_fixture_status_does_not_create_directories(tmp_path) -> None:
    config_path = tmp_path / "cases.yaml"
    config_path.write_text(
        "cases:\n  - matlab_results_stem: example_1_Blasius_results\n",
        encoding="utf-8",
    )
    input_dir = tmp_path / "missing_raw"
    output_dir = tmp_path / "missing_fixtures"

    status = extract_martini_to_npz.fixture_status(input_dir, output_dir, config_path)

    assert status["raw_mat_missing"] == ["example_1_Blasius_results"]
    assert status["npz_missing"] == ["example_1_Blasius_results"]
    assert not input_dir.exists()
    assert not output_dir.exists()


def test_martini_linstab2d_scripts_contain_parity_drivers() -> None:
    scripts_dir = repo / "examples/linstab2d/linstab2D_scripts"
    expected = [
        "example_0_TestCurvilinearDerivative.m",
        "example_1_Blasius.m",
        "example_2_Jet.m",
        "example_3_Streaky_BL.m",
        "example_4_Cavity.m",
        "example_5_Paraboloid.m",
        "run_ex3_paper_temporal_Lx2pi.m",
        "run_ex3_paper_spatial.m",
        "run_ex3_bundled_parity.m",
        "martini2024_setup_paths.m",
        "run_martini_exports.sh",
        "aux_matlab/AuxiliaryFunctions/martini2024_pack_parity_export.m",
    ]
    for rel in expected:
        path = scripts_dir / rel
        assert path.is_file(), f"missing {rel}"

    ex3 = (scripts_dir / "example_3_Streaky_BL.m").read_text(encoding="utf-8")
    assert "martini2024_pack_parity_export" in ex3
    assert "floquet_extX" not in ex3


def test_martini_matlab_export_runner_invokes_shell_script(tmp_path, monkeypatch) -> None:
    scripts_dir = repo / "examples/linstab2d/linstab2D_scripts"
    linstab2d = tmp_path / "LinStab2D"
    linstab2d.mkdir()
    (linstab2d / "aux_matlab").mkdir()
    (linstab2d / "BaseFlows").mkdir()

    captured: dict[str, object] = {}

    def fake_run(command, check=False, env=None, cwd=None):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = env
        captured["cwd"] = cwd
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(generate_martini_matlab_exports.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_martini_matlab_exports.py",
            "--linstab2d-root",
            str(linstab2d),
            "--output-dir",
            str(tmp_path / "raw"),
            "--only",
            "example4",
            "--only",
            "example5",
        ],
    )

    rc = generate_martini_matlab_exports.main()
    assert rc == 0
    assert captured["command"] == [
        "bash",
        str(scripts_dir / "run_martini_exports.sh"),
        "--only",
        "example_4",
        "--only",
        "example_5",
    ]
    assert captured["env"]["LINSTAB2D_ROOT"] == str(linstab2d.resolve())
    assert captured["cwd"] == str(scripts_dir)


def test_split_martini_fixtures_partitions_modes(tmp_path) -> None:
    from examples.linstab2d.split_martini_fixtures import process_file, split_payload

    payload = {
        "__meta_schema_version": np.asarray(["x"], dtype=str),
        "results_Cavity_direct__omega": np.arange(50, dtype=float),
        "results_Cavity_direct__V": np.random.randn(100, 50),
        "results_Cavity_direct__plot_grid__X": np.zeros((10, 10), dtype=float),
        "results_Cavity_direct__plot_grid__Y": np.zeros((10, 10), dtype=float),
        "results_Cavity_direct__plot_grid__usedInd": np.arange(20, dtype=float),
        "results_Cavity_direct__idx__u_j": np.arange(20, 40, dtype=float),
        "results_Cavity_direct__idx__v_j": np.arange(40, 60, dtype=float),
        "results_Cavity_direct__idx__w_j": np.arange(60, 80, dtype=float),
    }
    values, modes = split_payload(payload)
    assert set(modes) == {"results_Cavity_direct__V"}
    assert "results_Cavity_direct__omega" in values

    src = tmp_path / "example_4_Cavity_results_direct.npz"
    np.savez_compressed(src, **payload)
    out = tmp_path / "out"
    records = process_file(src, out)
    assert records[0]["kind"] == "copy"
    dst = out / src.name
    with np.load(dst, allow_pickle=False) as z:
        keys = [k for k in z.files if not k.startswith("__meta")]
        assert any("saved_mode_indices" in k for k in keys)
        v_key = [k for k in keys if k.endswith("__V")][0]
        assert z[v_key].shape[0] == 3
    assert not (out / f"{src.stem}_values.npz").exists()

    ex3_payload = {
        "__meta_schema_version": np.asarray(["x"], dtype=str),
        "results_StreakyBL_temporal__omega": np.arange(50, dtype=float),
        "results_StreakyBL_temporal__V": np.random.randn(100, 50),
        "results_StreakyBL_temporal__plot_grid__X": np.zeros((10, 10), dtype=float),
        "results_StreakyBL_temporal__plot_grid__Y": np.zeros((10, 10), dtype=float),
        "results_StreakyBL_temporal__plot_grid__usedInd": np.arange(100, dtype=float),
        "results_StreakyBL_temporal__idx__w_j": np.arange(1, 21, dtype=float),
    }
    src = tmp_path / "example_3_paper_temporal_Lx2pi_nP2_results_temporal.npz"
    np.savez_compressed(src, **ex3_payload)
    out = tmp_path / "out"
    records = process_file(src, out)
    assert records[0]["kind"] == "copy"
    dst = out / src.name
    with np.load(dst, allow_pickle=False) as z:
        keys = [k for k in z.files if not k.startswith("__meta")]
        assert any("saved_mode_indices" in k for k in keys)
        v_key = [k for k in keys if k.endswith("__V")][0]
        assert z[v_key].shape[0] == 3
    assert not (out / f"{src.stem}_values.npz").exists()


def test_slice_resolvent_uv_handles_double_bblasius_prefix(tmp_path) -> None:
    from examples.linstab2d.compare_martini_examples import _mode_matrix
    from examples.linstab2d.split_martini_fixtures import process_file, slice_resolvent_uv

    # The raw MATLAB export bundles are not distributed with the repository;
    # point LINSTABPY_MATLAB_FIXTURES_NPZ at a directory of raw fixture .npz
    # files to run this check.
    fixtures_dir = os.environ.get("LINSTABPY_MATLAB_FIXTURES_NPZ", "")
    if not fixtures_dir:
        pytest.skip(
            "Missing raw MATLAB fixture bundles; set LINSTABPY_MATLAB_FIXTURES_NPZ to enable"
        )
    source = Path(fixtures_dir) / "example_1_Blasius_results.npz"
    if not source.is_file():
        pytest.skip(f"Missing reference fixture: {source}")

    with np.load(source, allow_pickle=False) as raw:
        payload = {key: raw[key] for key in raw.files}

    slim = slice_resolvent_uv(payload)
    forces_key = next(key for key in slim if key.endswith("__forces"))
    forces = _mode_matrix(slim[forces_key])
    assert forces.shape == (60_000, 3)
    assert forces.dtype == np.complex128
    uj_key = next(key for key in slim if key.endswith("idx__u_j"))
    vj_key = next(key for key in slim if key.endswith("idx__v_j"))
    uj = np.asarray(slim[uj_key]).reshape(-1)
    vj = np.asarray(slim[vj_key]).reshape(-1)
    assert int(uj.min()) == 30_001
    assert int(vj.min()) == 60_001
    gains_key = next(key for key in slim if key.endswith("__gains"))
    np.testing.assert_allclose(
        np.asarray(slim[gains_key]).reshape(-1),
        np.array([14.65714005, 1.89911838, 1.34277958]),
        rtol=0,
        atol=1e-6,
    )

    src = tmp_path / source.name
    np.savez_compressed(src, **payload)
    out = tmp_path / "LinStab2DResults"
    records = process_file(src, out)
    assert records[0]["kind"] == "copy"
    dst = out / src.name
    with np.load(dst, allow_pickle=False) as z:
        forces_out = _mode_matrix(z[forces_key])
        assert forces_out.shape == (60_000, 3)
        np.testing.assert_allclose(forces_out, forces, rtol=0, atol=0)


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
        tolerances={"field_abs": 1e-12, "field_rel": 1e-12},
    )

    assert report["python_protocol_pass"] is True
    assert report["fields"]["df"]["max_abs_error"] == 0.0
    assert report["fields"]["df"]["rel_l2"] == 0.0


# MATLAB-export parity check; the reference bundle is kept out of the repository
# and not distributed. Skips cleanly when the bundle is absent.
def test_example_0_reference_fields_match_exported_matlab_bundle() -> None:
    bundle_path = martini_fixture("example_0_TestCurvilinearDerivative_results.npz")
    if not bundle_path.exists():
        pytest.skip(f"Missing MATLAB parity bundle: {bundle_path}")
    from examples.linstab2d.example_0_curvilinear_validation import _compute_reference_fields

    # load_matlab_bundle strips the ``results__`` wrapper the slim
    # LinStab2DResults/ fixtures carry; the field keys below are the logical ones.
    matlab = load_matlab_bundle(MARTINI_FIXTURE_ROOT, "example_0_TestCurvilinearDerivative_results")
    cfg = load_config(repo / "examples/linstab2d/example_0_curvilinear_validation.yaml")
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


def test_compare_case_streaky_bl_walks_campaign_npzs(tmp_path, monkeypatch) -> None:
    """The dispatcher compares the per-campaign NPZs Example 3 actually writes.

    Guards the wiring that rotted once: the manifest pointed at a combined
    example_3_streaky_bl.npz no code writes, so every full parity run
    HARD_FAILed on a missing artifact while the campaign NPZs sat on disk.
    """
    monkeypatch.setattr(compare_martini_examples, "ROOT", tmp_path)
    npz_dir = tmp_path / "bundles"
    npz_dir.mkdir()

    temporal_omega = np.array([0.1 + 0.2j, 0.3 + 0.4j], dtype=complex)
    temporal_modes = np.array(
        [
            [1.0 + 0.0j, 2.0 + 1.0j],
            [0.5 + 0.5j, 1.5 - 0.5j],
            [2.0 - 1.0j, 1.0 + 0.0j],
        ],
        dtype=complex,
    )
    spatial_alpha = np.array([0.7 + 0.1j, 0.8 + 0.2j], dtype=complex)
    residuals = np.array([1.0e-12, 1.0e-12])

    np.savez_compressed(
        tmp_path / "camp_temporal.npz",
        temporal_omega=temporal_omega,
        temporal_modes=temporal_modes,
        temporal_residuals=residuals,
    )
    np.savez_compressed(
        tmp_path / "camp_spatial.npz",
        temporal_omega=temporal_omega,
        temporal_modes=temporal_modes,
        temporal_residuals=residuals,
        spatial_alpha=spatial_alpha,
        spatial_residuals=residuals,
    )
    matlab_temporal_payload = {
        "results__results_StreakyBL_temporal__omega": temporal_omega,
        "results__results_StreakyBL_temporal__V": temporal_modes,
    }
    np.savez_compressed(npz_dir / "camp_temporal_results_temporal.npz", **matlab_temporal_payload)
    np.savez_compressed(npz_dir / "camp_spatial_results_temporal.npz", **matlab_temporal_payload)
    np.savez_compressed(
        npz_dir / "camp_spatial_results_spatial.npz",
        **{"results__results_StreakyBL_spatial__alphas": spatial_alpha},
    )

    case = {
        "case_id": "example_3_streaky_bl",
        "label": "Example 3",
        "comparison_kind": "streaky_bl",
        "comparison_scope": "temporal_and_spatial",
        "campaigns": [
            {
                "name": "t1",
                "python_npz": "camp_temporal.npz",
                "matlab_temporal_stem": "camp_temporal_results_temporal",
            },
            {
                "name": "sp",
                "python_npz": "camp_spatial.npz",
                "matlab_temporal_stem": "camp_spatial_results_temporal",
                "matlab_spatial_stem": "camp_spatial_results_spatial",
            },
        ],
        "matlab_temporal_omega_key": "results_StreakyBL_temporal/omega",
        "matlab_temporal_modes_key": "results_StreakyBL_temporal/V",
        "matlab_spatial_alpha_key": "results_StreakyBL_spatial/alphas",
        "n_compare": 2,
        "tolerances": {"temporal_residual_max": 1.0e-8, "spatial_residual_max": 1.0e-8},
    }

    result = compare_martini_examples._compare_case(case, npz_dir)

    assert result["status"] == "PASS"
    campaigns = result["metrics"]["campaigns"]
    assert set(campaigns) == {"t1", "sp"}
    assert campaigns["t1"]["campaign_status"] == "PASS"
    assert campaigns["t1"]["spatial"]["n_compare"] == 0
    assert campaigns["sp"]["spatial"]["residual_max"] <= 1.0e-8
    assert result["python_npz"] == ["camp_temporal.npz", "camp_spatial.npz"]
    assert len(result["matlab_npz"]) == 3


def test_compare_case_streaky_bl_missing_campaign_npz_is_hard_fail(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(compare_martini_examples, "ROOT", tmp_path)
    case = {
        "case_id": "example_3_streaky_bl",
        "comparison_kind": "streaky_bl",
        "campaigns": [
            {
                "name": "t1",
                "python_npz": "absent.npz",
                "matlab_temporal_stem": "camp_temporal_results_temporal",
            },
        ],
        "matlab_temporal_omega_key": "results_StreakyBL_temporal/omega",
        "matlab_temporal_modes_key": "results_StreakyBL_temporal/V",
        "matlab_spatial_alpha_key": "results_StreakyBL_spatial/alphas",
    }

    result = compare_martini_examples._compare_case(case, tmp_path)

    assert result["status"] == "HARD_FAIL"
    assert "Missing Python artifact" in result["note"]
