import json
from pathlib import Path

from scripts.validation import check_docstrings


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_docstring_checker_reports_missing_public_symbols(tmp_path):
    root = tmp_path / "pkg"
    _write(
        root / "__init__.py",
        '"""Package docs."""\n\n'
        "def documented():\n"
        '    """Function docs."""\n'
        "    return 1\n\n"
        "def missing():\n"
        "    return 2\n\n"
        "def _private():\n"
        "    return 3\n",
    )

    symbols = check_docstrings.scan(root)
    missing = [
        item.qualified_name
        for item in symbols
        if item.required and not item.has_docstring and not item.allowlisted
    ]

    assert "linstabpy.missing" in missing
    assert "linstabpy._private" not in missing


def test_docstring_checker_can_require_private_helpers(tmp_path):
    root = tmp_path / "pkg"
    _write(root / "operators.py", '"""Module docs."""\n\ndef _kernel():\n    return 1\n')

    symbols = check_docstrings.scan(root, require_private={"linstabpy.operators._kernel"})
    missing = [
        item.qualified_name
        for item in symbols
        if item.required and not item.has_docstring and not item.allowlisted
    ]

    if missing != ["linstabpy.operators._kernel"]:
        raise AssertionError(missing)


def test_docstring_checker_skips_private_module_symbols_by_default(tmp_path):
    root = tmp_path / "pkg"
    _write(root / "_solver_cfg.py", "def get_option():\n    return None\n")

    symbols = check_docstrings.scan(root)
    missing = [
        item.qualified_name
        for item in symbols
        if item.required and not item.has_docstring and not item.allowlisted
    ]

    assert missing == []


def test_docstring_checker_allowlist_requires_rationale(tmp_path):
    root = tmp_path / "pkg"
    _write(root / "__init__.py", "def missing():\n    return 1\n")
    allowlist = {"linstabpy.missing": "Queued for public API pass."}

    symbols = check_docstrings.scan(root, allowlisted=allowlist)
    item = next(symbol for symbol in symbols if symbol.qualified_name == "linstabpy.missing")

    assert item.required
    assert item.allowlisted
    assert item.rationale == "Queued for public API pass."


def test_docstring_checker_main_writes_json_report(tmp_path, capsys):
    root = tmp_path / "pkg"
    repo = tmp_path / "repo"
    report = tmp_path / "report.json"
    _write(root / "__init__.py", '"""Package docs."""\n\ndef ok():\n    """Docs."""\n')
    _write(repo / "src" / "linstabpy" / "module.py", '"""Docs."""\n')
    _write(repo / "tests" / "test_module.py", "def test_ok():\n    pass\n")
    _write(repo / "examples" / "example.py", "print('ok')\n")
    _write(repo / "scripts" / "validation" / "check.py", "print('ok')\n")
    _write(repo / "docs" / "guide.md", "# Guide\n")

    rc = check_docstrings.main(
        ["--root", str(root), "--repo-root", str(repo), "--output", str(report)]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "File inventory:" in captured.out
    assert "missing=0" in captured.out
    with report.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["file_inventory"]["inventory_source"] == "filesystem-fallback"
    assert payload["file_inventory"]["source_python_files"] == 0
    assert payload["file_inventory"]["test_python_files"] == 1
    assert payload["file_inventory"]["example_python_files"] == 1
    assert payload["file_inventory"]["validation_python_files"] == 1
    assert payload["file_inventory"]["documentation_markdown_files"] == 1
    assert payload["summary"]["missing_required"] == 0


def test_docstring_checker_writes_markdown_coverage_table(tmp_path):
    root = tmp_path / "pkg"
    report = tmp_path / "coverage.md"
    _write(
        root / "__init__.py",
        '"""Package docs."""\n\ndef documented():\n    """Function docs."""\n    return 1\n',
    )

    rc = check_docstrings.main(["--root", str(root), "--markdown-output", str(report)])

    assert rc == 0
    text = report.read_text(encoding="utf-8")
    assert "# Public API Docstring Coverage" in text
    assert "| Required symbols | 2 |" in text
    assert "| `linstabpy.documented` | function |" in text
    assert "| `linstabpy.documented` | function | `__init__.py:3` | documented |  |" in text
