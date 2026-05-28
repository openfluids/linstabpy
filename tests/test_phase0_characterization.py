import json

import numpy as np
import pytest

from scripts.validation import phase0_characterization as phase0


def test_high_mode_output_writes_separate_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(phase0, "_git_sha", lambda: "test-sha")
    monkeypatch.setattr(phase0, "_bc_characterization", lambda: [{"label": "bc"}])
    monkeypatch.setattr(phase0, "_coefficient_characterization", lambda: [{"label": "coeff"}])
    monkeypatch.setattr(
        phase0,
        "_high_mode_baseline",
        lambda *, n_modes, ncv: [{"label": "high", "n_modes": n_modes, "ncv": ncv}],
    )

    main_output = tmp_path / "phase0.json"
    high_output = tmp_path / "high.json"
    rc = phase0.main(
        [
            "--output",
            str(main_output),
            "--high-mode-output",
            str(high_output),
            "--high-mode-count",
            "101",
            "--high-mode-ncv",
            "141",
        ]
    )

    assert rc == 0
    main_payload = json.loads(main_output.read_text(encoding="utf-8"))
    high_payload = json.loads(high_output.read_text(encoding="utf-8"))
    assert "high_mode_baseline" not in main_payload
    assert high_payload["git_sha"] == "test-sha"
    assert high_payload["high_mode_baseline"] == [{"label": "high", "n_modes": 101, "ncv": 141}]


def test_include_high_mode_embeds_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(phase0, "_git_sha", lambda: "test-sha")
    monkeypatch.setattr(phase0, "_bc_characterization", lambda: [])
    monkeypatch.setattr(phase0, "_coefficient_characterization", lambda: [])
    monkeypatch.setattr(
        phase0,
        "_high_mode_baseline",
        lambda *, n_modes, ncv: [{"label": "embedded", "n_modes": n_modes, "ncv": ncv}],
    )

    output = tmp_path / "phase0.json"
    rc = phase0.main(
        [
            "--output",
            str(output),
            "--include-high-mode",
            "--high-mode-count",
            "100",
            "--high-mode-ncv",
            "140",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["high_mode_baseline"] == [{"label": "embedded", "n_modes": 100, "ncv": 140}]


def test_coefficient_snapshot_output_writes_npz(monkeypatch, tmp_path):
    monkeypatch.setattr(phase0, "_git_sha", lambda: "test-sha")
    monkeypatch.setattr(phase0, "_bc_characterization", lambda: [])
    monkeypatch.setattr(phase0, "_coefficient_characterization", lambda: [])
    monkeypatch.setattr(
        phase0,
        "_coefficient_snapshot_arrays",
        lambda: (
            {"cases": [{"label": "case", "arrays": {"A0": {"npz_key": "case/A0"}}}]},
            {"case/A0": np.array([[1.0, 2.0]])},
        ),
    )

    output = tmp_path / "phase0.json"
    snapshot = tmp_path / "coefficients.npz"
    rc = phase0.main(
        [
            "--output",
            str(output),
            "--coefficient-snapshot-output",
            str(snapshot),
        ]
    )

    assert rc == 0
    with np.load(snapshot) as data:
        metadata = json.loads(str(data["__metadata_json__"].item()))
        np.testing.assert_allclose(data["case/A0"], [[1.0, 2.0]])
    assert metadata["git_sha"] == "test-sha"
    assert metadata["artifact_kind"] == "phase0_coefficient_tensor_snapshot"
    assert metadata["cases"][0]["arrays"]["A0"]["npz_key"] == "case/A0"


@pytest.mark.parametrize(
    "label",
    [
        "case1_overlap",
        "case1_exclude_corners",
        "sheared_curvilinear_neumann",
        "same_border_later_neumann_owns_overlap",
        "pure_neumann_lr_exclude_corners",
        "axisymmetric_dirichlet_ltr",
    ],
)
def test_bc_characterization_includes_representative_recipes(label):
    reports = phase0._bc_characterization()

    assert label in {report["label"] for report in reports}


def test_bc_characterization_records_later_rule_row_ownership():
    reports = {report["label"]: report for report in phase0._bc_characterization()}
    overlap_case = reports["same_border_later_neumann_owns_overlap"]

    sets = overlap_case["sets"]
    assert sets["dirichlet_indices"]["count"] == sets["neumann_indices"]["count"]
    assert sets["overlap_indices"]["count"] == sets["dirichlet_indices"]["count"]
    assert sets["effective_dirichlet_indices"]["count"] == 0
    assert sets["effective_neumann_indices"]["count"] == sets["overlap_indices"]["count"]


def test_bc_characterization_records_axisymmetric_model():
    reports = {report["label"]: report for report in phase0._bc_characterization()}
    axy_case = reports["axisymmetric_dirichlet_ltr"]

    assert axy_case["model"] == "axy"
    assert axy_case["sets"]["dirichlet_indices"]["count"] > 0
    assert axy_case["sets"]["neumann_indices"]["count"] == 0
