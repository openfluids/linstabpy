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
    report = tmp_path / "report.json"
    _write(root / "__init__.py", '"""Package docs."""\n\ndef ok():\n    """Docs."""\n')

    rc = check_docstrings.main(["--root", str(root), "--output", str(report)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "missing=0" in captured.out
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["missing_required"] == 0
