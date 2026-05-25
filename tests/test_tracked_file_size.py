# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from pathlib import Path

from scripts.check_tracked_file_size import _bytes_from_mib, _format_bytes, _oversized


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
