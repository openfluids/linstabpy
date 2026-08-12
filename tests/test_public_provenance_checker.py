from pathlib import Path

from tests import public_provenance


def _write(path: Path, text: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def test_public_provenance_checker_reports_internal_terms(tmp_path):
    _write(tmp_path / "README.md", "Reviewed by Codex from $HOME/work\n")

    matches = public_provenance.scan_paths(
        [tmp_path / "README.md"],
        repo_root=tmp_path,
    )

    assert [(item.category, item.line) for item in matches] == [
        ("internal-review-tool", 1),
        ("private-local-path", 1),
    ]


def test_public_provenance_checker_skips_binary_files(tmp_path):
    binary = tmp_path / "fixture.mat"
    _write(binary, b"\x00Codex $HOME")

    matches = public_provenance.scan_paths([binary], repo_root=tmp_path)

    assert matches == []


def test_public_provenance_checker_avoids_substring_false_positive(tmp_path):
    path = tmp_path / "docs" / "PLAN.md"
    _write(path, "shock-breathing vs BL-pulsation remains a mode label\n")

    matches = public_provenance.scan_paths([path], repo_root=tmp_path)

    assert matches == []


def test_public_provenance_checker_excludes_its_own_pattern_files(tmp_path):
    scanner = tmp_path / "tests" / "public_provenance.py"
    _write(scanner, 'PATTERN = "Codex $HOME"\n')

    matches = public_provenance.scan_paths([scanner], repo_root=tmp_path)

    assert matches == []


def test_public_provenance_checker_main_writes_json(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    _write(repo / "README.md", "Plain public docs.\n")

    monkeypatch.setattr(public_provenance, "_tracked_files", lambda root: [repo / "README.md"])

    report = tmp_path / "report.json"
    rc = public_provenance.main(["--repo-root", str(repo), "--output", str(report)])

    assert rc == 0
    assert "pass (0 matches)" in capsys.readouterr().out
    assert '"match_count": 0' in report.read_text(encoding="utf-8")


def test_tracked_tree_has_no_private_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    matches = public_provenance.private_matches(public_provenance.scan_tracked(root))
    assert matches == []
