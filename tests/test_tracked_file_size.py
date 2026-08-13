# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from pathlib import Path

from tests.tracked_file_size import (
    DEFAULT_TRACKED_LIMIT_MIB,
    _bytes_from_mib,
    _format_bytes,
    _git_list,
    _oversized,
    _repo_root,
    main,
)


def test_bytes_from_mib_uses_binary_units() -> None:
    assert _bytes_from_mib(1.5) == 1_572_864


def test_format_bytes_reports_mib() -> None:
    assert _format_bytes(5 * 1024 * 1024) == "5.0 MiB"


def test_oversized_reports_paths_relative_to_root(tmp_path: Path) -> None:
    small = tmp_path / "small.bin"
    large = tmp_path / "large.bin"
    small.write_bytes(b"a" * 8)
    large.write_bytes(b"b" * 16)

    assert _oversized([small, large], tmp_path, limit_bytes=10) == [(16, Path("large.bin"))]


def test_no_tracked_file_exceeds_limit() -> None:
    root = _repo_root()
    tracked = _git_list(root, ["ls-files"])
    assert _oversized(tracked, root, _bytes_from_mib(DEFAULT_TRACKED_LIMIT_MIB)) == []


def test_tracked_tree_has_no_plan_or_goal_markdown() -> None:
    root = _repo_root()
    tracked = _git_list(root, ["ls-files"])
    banned = [
        path.relative_to(root).as_posix()
        for path in tracked
        if path.name.startswith(("plan", "goal")) and path.suffix == ".md"
    ]
    assert banned == []


def test_main_returns_zero_on_this_repo() -> None:
    assert main([]) == 0
