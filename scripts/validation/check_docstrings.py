#!/usr/bin/env python3
"""Inventory and gate docstring coverage for public Python symbols."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "src" / "linstabpy"


@dataclass(frozen=True)
class Symbol:
    kind: str
    name: str
    qualified_name: str
    path: str
    line: int
    has_docstring: bool
    required: bool
    allowlisted: bool = False
    rationale: str | None = None


@dataclass(frozen=True)
class FileSet:
    files: list[Path]
    source: str


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("linstabpy", *parts)) if parts else "linstabpy"


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_public_module(module: str) -> bool:
    return all(_is_public(part) for part in module.split("."))


def _is_public_qualified(module: str, qualified_name: str) -> bool:
    if not _is_public_module(module):
        return False
    rel = qualified_name.removeprefix(f"{module}.")
    return all(_is_public(part) for part in rel.split("."))


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()


def _docstring(node: ast.AST) -> str | None:
    return ast.get_docstring(node, clean=False)


def _allowlist(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid docstring allowlist JSON: {path}") from exc
    if not isinstance(data, dict):
        raise TypeError("docstring allowlist must be a JSON object")
    allowlisted: dict[str, str] = {}
    for qualified_name, rationale in data.items():
        if not isinstance(qualified_name, str) or not isinstance(rationale, str):
            raise TypeError("docstring allowlist keys and rationales must be strings")
        if not rationale.strip():
            raise ValueError(f"allowlist entry for {qualified_name!r} needs a rationale")
        allowlisted[qualified_name] = rationale
    return allowlisted


def _iter_child_symbols(
    *,
    rel_path: str,
    module: str,
    body: list[ast.stmt],
    parent: str,
    allowlisted: dict[str, str],
    require_private: set[str],
) -> list[Symbol]:
    symbols: list[Symbol] = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            qualified = f"{parent}.{node.name}"
            public = _is_public_qualified(module, qualified)
            required = public or qualified in require_private
            symbols.append(
                _symbol(
                    "class",
                    node.name,
                    qualified,
                    rel_path,
                    node.lineno,
                    _docstring(node) is not None,
                    required,
                    allowlisted,
                )
            )
            symbols.extend(
                _iter_child_symbols(
                    rel_path=rel_path,
                    module=module,
                    body=node.body,
                    parent=qualified,
                    allowlisted=allowlisted,
                    require_private=require_private,
                )
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            qualified = f"{parent}.{node.name}"
            public = _is_public_qualified(module, qualified)
            required = public or qualified in require_private
            symbols.append(
                _symbol(
                    "function",
                    node.name,
                    qualified,
                    rel_path,
                    node.lineno,
                    _docstring(node) is not None,
                    required,
                    allowlisted,
                )
            )
    return symbols


def _symbol(
    kind: str,
    name: str,
    qualified_name: str,
    rel_path: str,
    line: int,
    has_docstring: bool,
    required: bool,
    allowlisted: dict[str, str],
) -> Symbol:
    rationale = allowlisted.get(qualified_name)
    return Symbol(
        kind=kind,
        name=name,
        qualified_name=qualified_name,
        path=rel_path,
        line=line,
        has_docstring=has_docstring,
        required=required,
        allowlisted=rationale is not None,
        rationale=rationale,
    )


def scan(
    root: Path,
    *,
    allowlisted: dict[str, str] | None = None,
    require_private: set[str] | None = None,
) -> list[Symbol]:
    root = root.resolve()
    allowlisted = allowlisted or {}
    require_private = require_private or set()
    symbols: list[Symbol] = []
    for path in sorted(root.rglob("*.py")):
        rel_path = _relative_path(path, root)
        module = _module_name(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_required = module in require_private or _is_public_module(module)
        symbols.append(
            _symbol(
                "module",
                module.rsplit(".", 1)[-1],
                module,
                rel_path,
                1,
                _docstring(tree) is not None,
                module_required,
                allowlisted,
            )
        )
        symbols.extend(
            _iter_child_symbols(
                rel_path=rel_path,
                module=module,
                body=tree.body,
                parent=module,
                allowlisted=allowlisted,
                require_private=require_private,
            )
        )
    return symbols


def _summary(symbols: list[Symbol]) -> dict[str, Any]:
    required = [item for item in symbols if item.required]
    missing = [item for item in required if not item.has_docstring and not item.allowlisted]
    allowlisted_missing = [item for item in required if not item.has_docstring and item.allowlisted]
    by_kind: dict[str, dict[str, int]] = {}
    for item in symbols:
        bucket = by_kind.setdefault(
            item.kind,
            {"total": 0, "required": 0, "missing": 0, "allowlisted_missing": 0},
        )
        bucket["total"] += 1
        if item.required:
            bucket["required"] += 1
            if not item.has_docstring and item.allowlisted:
                bucket["allowlisted_missing"] += 1
            elif not item.has_docstring:
                bucket["missing"] += 1
    return {
        "total_symbols": len(symbols),
        "required_symbols": len(required),
        "missing_required": len(missing),
        "allowlisted_missing": len(allowlisted_missing),
        "by_kind": by_kind,
    }


def _tracked_files(repo_root: Path) -> FileSet:
    repo_root = repo_root.resolve()
    git = shutil.which("git")
    if git is None:
        return FileSet(_fallback_files(repo_root), "filesystem-fallback")
    try:
        result = subprocess.run(
            [git, "ls-files"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return FileSet(_fallback_files(repo_root), "filesystem-fallback")
    return FileSet(
        [repo_root / line for line in result.stdout.splitlines() if line], "git-ls-files"
    )


def _fallback_files(repo_root: Path) -> list[Path]:
    ignored = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "artifacts"}
    return sorted(
        path
        for path in repo_root.rglob("*")
        if path.is_file() and not any(part in ignored for part in path.relative_to(repo_root).parts)
    )


def _count_matching(files: list[Path], repo_root: Path, prefix: str, suffix: str) -> int:
    prefix_path = Path(prefix)
    count = 0
    for path in files:
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        if rel.parts[: len(prefix_path.parts)] == prefix_path.parts and rel.name.endswith(suffix):
            count += 1
    return count


def _file_inventory(repo_root: Path, source_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    file_set = _tracked_files(repo_root)
    files = file_set.files
    py_files = [path for path in files if path.suffix == ".py"]
    source_files = [path for path in py_files if path.is_relative_to(source_root)]
    return {
        "inventory_source": file_set.source,
        "repo_root": _relative_path(repo_root, repo_root),
        "source_root": _relative_path(source_root, repo_root),
        "tracked_files": len(files),
        "tracked_python_files": len(py_files),
        "source_python_files": len(source_files),
        "test_python_files": _count_matching(files, repo_root, "tests", ".py"),
        "example_python_files": _count_matching(files, repo_root, "examples", ".py"),
        "validation_python_files": _count_matching(files, repo_root, "scripts/validation", ".py"),
        "documentation_markdown_files": _count_matching(files, repo_root, "docs", ".md"),
    }


def _payload(symbols: list[Symbol], *, repo_root: Path, source_root: Path) -> dict[str, Any]:
    return {
        "file_inventory": _file_inventory(repo_root, source_root),
        "summary": _summary(symbols),
        "symbols": [asdict(item) for item in symbols],
    }


def _coverage_status(item: Symbol) -> str:
    if item.has_docstring:
        return "documented"
    if item.allowlisted:
        return "allowlisted"
    if item.required:
        return "missing"
    return "not required"


def _coverage_markdown(symbols: list[Symbol], *, repo_root: Path, source_root: Path) -> str:
    payload = _payload(symbols, repo_root=repo_root, source_root=source_root)
    files = payload["file_inventory"]
    summary = payload["summary"]
    lines = [
        "# Public API Docstring Coverage",
        "",
        "Generated by `scripts/validation/check_docstrings.py` from the tracked",
        "package source. The table accounts for every required public symbol and",
        "selected private helper passed through `--require-private`.",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Tracked files | {files['tracked_files']} |",
        f"| Tracked Python files | {files['tracked_python_files']} |",
        f"| Source Python files | {files['source_python_files']} |",
        f"| Required symbols | {summary['required_symbols']} |",
        f"| Missing required docstrings | {summary['missing_required']} |",
        f"| Allowlisted missing docstrings | {summary['allowlisted_missing']} |",
        "",
        "## Required Symbols",
        "",
        "| Symbol | Kind | Location | Status | Rationale |",
        "|---|---|---|---|---|",
    ]
    required = sorted(
        (item for item in symbols if item.required),
        key=lambda item: (item.path, item.line, item.qualified_name),
    )
    for item in required:
        rationale = item.rationale or ""
        lines.append(
            f"| `{item.qualified_name}` | {item.kind} | "
            f"`{item.path}:{item.line}` | {_coverage_status(item)} | {rationale} |"
        )
    lines.append("")
    return "\n".join(lines)


def _print_text(symbols: list[Symbol], *, repo_root: Path, source_root: Path) -> None:
    payload = _payload(symbols, repo_root=repo_root, source_root=source_root)
    files = payload["file_inventory"]
    summary = payload["summary"]
    print(
        "File inventory: "
        f"tracked={files['tracked_files']} "
        f"python={files['tracked_python_files']} "
        f"src_python={files['source_python_files']} "
        f"tests={files['test_python_files']} "
        f"examples={files['example_python_files']} "
        f"validation={files['validation_python_files']}"
    )
    print(
        "Docstring coverage: "
        f"required={summary['required_symbols']} "
        f"missing={summary['missing_required']} "
        f"allowlisted_missing={summary['allowlisted_missing']} "
        f"total={summary['total_symbols']}"
    )
    for kind, bucket in sorted(summary["by_kind"].items()):
        print(
            f"  {kind}: required={bucket['required']} missing={bucket['missing']} "
            f"allowlisted_missing={bucket['allowlisted_missing']} total={bucket['total']}"
        )
    missing = [
        item
        for item in symbols
        if item.required and not item.has_docstring and not item.allowlisted
    ]
    if missing:
        print("\nMissing required docstrings:")
        for item in missing:
            print(f"  {item.path}:{item.line}: {item.qualified_name} ({item.kind})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument(
        "--require-private",
        action="append",
        default=[],
        metavar="QUALIFIED_NAME",
        help="Require a private/internal helper by qualified name.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report to this path.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Write a human-readable Markdown coverage table.",
    )
    args = parser.parse_args(argv)

    symbols = scan(
        args.root,
        allowlisted=_allowlist(args.allowlist),
        require_private=set(args.require_private),
    )
    payload = _payload(symbols, repo_root=args.repo_root, source_root=args.root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            _coverage_markdown(symbols, repo_root=args.repo_root, source_root=args.root),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(symbols, repo_root=args.repo_root, source_root=args.root)
    return 1 if payload["summary"]["missing_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
