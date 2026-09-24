#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Scan tracked public files for private paths and internal review provenance."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PATTERNS: tuple[tuple[str, str], ...] = (
    ("internal-review-process", r"\b(?:AI-assisted|generated review|generated draft)\b"),
    ("internal-process-language", r"\b(?:prompts?|agents?)\b"),
    ("private-local-path", r"(?:/home/rfrantz|/Users/rfrantz|/media/hamada)"),
    ("private-user-or-host", r"\b(?:ghamada|pa-dyf-calc|nexus|supermicro|ensam)\b|postprochamada"),
)

EXCLUDED_PATHS = {
    "tests/public_provenance.py",
    "tests/test_public_provenance_checker.py",
    "CHANGELOG.md",
}

PRIVATE_CATEGORIES = frozenset({"private-local-path", "private-user-or-host"})


@dataclass(frozen=True)
class Match:
    path: str
    line: int
    column: int
    category: str
    pattern: str
    text: str


def _tracked_files(repo_root: Path) -> list[Path]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for the public provenance scan")
    result = subprocess.run(
        [git, "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return [repo_root / line for line in result.stdout.splitlines() if line]


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\0" in chunk


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def scan_paths(paths: list[Path], *, repo_root: Path) -> list[Match]:
    compiled = [(category, re.compile(pattern, re.IGNORECASE)) for category, pattern in PATTERNS]
    matches: list[Match] = []
    for path in sorted(paths):
        if _is_binary(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = _relative(path, repo_root)
        if rel in EXCLUDED_PATHS:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for category, pattern in compiled:
                for found in pattern.finditer(line):
                    matches.append(
                        Match(
                            path=rel,
                            line=line_no,
                            column=found.start() + 1,
                            category=category,
                            pattern=pattern.pattern,
                            text=line.strip(),
                        )
                    )
    return matches


def scan_tracked(repo_root: Path) -> list[Match]:
    repo_root = repo_root.resolve()
    return scan_paths(_tracked_files(repo_root), repo_root=repo_root)


def private_matches(matches: list[Match]) -> list[Match]:
    """Return only private-path and private-host hits."""
    return [match for match in matches if match.category in PRIVATE_CATEGORIES]


def _payload(matches: list[Match]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    for match in matches:
        by_category[match.category] = by_category.get(match.category, 0) + 1
    return {
        "status": "fail" if matches else "pass",
        "match_count": len(matches),
        "by_category": dict(sorted(by_category.items())),
        "matches": [asdict(match) for match in matches],
    }


def _print_text(matches: list[Match]) -> None:
    if not matches:
        print("Public provenance scan: pass (0 matches)")
        return
    print(f"Public provenance scan: fail ({len(matches)} matches)")
    for match in matches:
        print(f"{match.path}:{match.line}:{match.column}: {match.category}: {match.text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report.")
    args = parser.parse_args(argv)

    matches = scan_tracked(args.repo_root)
    payload = _payload(matches)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(matches)
    return 1 if matches else 0


if __name__ == "__main__":
    raise SystemExit(main())
