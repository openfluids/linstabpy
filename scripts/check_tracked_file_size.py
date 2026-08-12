#!/usr/bin/env python3
"""Check tracked Git files against the repository file-size policy."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 60 MiB is set by examples/martini/example_3_Streaky_BL_results_temporal_modes.npz
# after the complex64 pack. GitHub rejects files at 100 MB.
DEFAULT_TRACKED_LIMIT_MIB = 60.0
DEFAULT_IGNORED_REPORT_LIMIT_MIB = 60.0


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _git_list(root: Path, args: list[str]) -> list[Path]:
    result = subprocess.run(
        ["git", *args, "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    raw_paths = result.stdout.split(b"\0")
    return [root / raw.decode() for raw in raw_paths if raw]


def _bytes_from_mib(value: float) -> int:
    return int(value * 1024 * 1024)


def _format_bytes(size: int) -> str:
    mib = size / (1024 * 1024)
    return f"{mib:.1f} MiB"


def _oversized(paths: list[Path], root: Path, limit_bytes: int) -> list[tuple[int, Path]]:
    large: list[tuple[int, Path]] = []
    for path in paths:
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > limit_bytes:
            large.append((size, path.relative_to(root)))
    return sorted(large, reverse=True)


def _print_large(header: str, rows: list[tuple[int, Path]]) -> None:
    print(header)
    for size, path in rows:
        print(f"  {_format_bytes(size):>10}  {path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when a tracked file exceeds the GitHub-facing size limit. "
            "Ignored local outputs over the report threshold are listed separately."
        )
    )
    parser.add_argument(
        "--limit-mib",
        type=float,
        default=DEFAULT_TRACKED_LIMIT_MIB,
        help="Maximum allowed tracked file size in MiB.",
    )
    parser.add_argument(
        "--ignored-report-limit-mib",
        type=float,
        default=DEFAULT_IGNORED_REPORT_LIMIT_MIB,
        help="Ignored local files above this size are reported but do not fail the check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = _repo_root()

    tracked = _git_list(root, ["ls-files"])
    tracked_large = _oversized(tracked, root, _bytes_from_mib(args.limit_mib))

    ignored = _git_list(root, ["ls-files", "--others", "--ignored", "--exclude-standard"])
    ignored_large = _oversized(ignored, root, _bytes_from_mib(args.ignored_report_limit_mib))

    if tracked_large:
        _print_large(
            f"Tracked files over {args.limit_mib:g} MiB are not allowed:",
            tracked_large,
        )
    else:
        print(f"OK: no tracked files exceed {args.limit_mib:g} MiB.")

    if ignored_large:
        _print_large(
            f"Ignored local files over {args.ignored_report_limit_mib:g} MiB:",
            ignored_large,
        )

    return 1 if tracked_large else 0


if __name__ == "__main__":
    raise SystemExit(main())
