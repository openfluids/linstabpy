import json

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
