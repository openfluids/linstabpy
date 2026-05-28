from pathlib import Path

from scripts.validation import check_public_provenance


def _write(path: Path, text: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def test_public_provenance_checker_reports_internal_terms(tmp_path):
    _write(tmp_path / "README.md", "Reviewed by Codex from /home/rfrantz/work\n")

    matches = check_public_provenance.scan_paths(
        [tmp_path / "README.md"],
        repo_root=tmp_path,
    )

    assert [(item.category, item.line) for item in matches] == [
        ("internal-review-tool", 1),
        ("private-local-path", 1),
    ]


def test_public_provenance_checker_skips_binary_files(tmp_path):
    binary = tmp_path / "fixture.mat"
    _write(binary, b"\x00Codex /home/rfrantz")

    matches = check_public_provenance.scan_paths([binary], repo_root=tmp_path)

    assert matches == []


def test_public_provenance_checker_avoids_substring_false_positive(tmp_path):
    path = tmp_path / "docs" / "PLAN.md"
    _write(path, "shock-breathing vs BL-pulsation remains a mode label\n")

    matches = check_public_provenance.scan_paths([path], repo_root=tmp_path)

    assert matches == []


def test_public_provenance_checker_excludes_its_own_pattern_files(tmp_path):
    scanner = tmp_path / "scripts" / "validation" / "check_public_provenance.py"
    _write(scanner, 'PATTERN = "Codex /home/rfrantz"\n')

    matches = check_public_provenance.scan_paths([scanner], repo_root=tmp_path)

    assert matches == []


def test_public_provenance_checker_main_writes_json(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    _write(repo / "README.md", "Plain public docs.\n")

    monkeypatch.setattr(
        check_public_provenance, "_tracked_files", lambda root: [repo / "README.md"]
    )

    report = tmp_path / "report.json"
    rc = check_public_provenance.main(["--repo-root", str(repo), "--output", str(report)])

    assert rc == 0
    assert "pass (0 matches)" in capsys.readouterr().out
    assert '"match_count": 0' in report.read_text(encoding="utf-8")
