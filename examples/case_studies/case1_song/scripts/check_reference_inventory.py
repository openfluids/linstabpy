#!/usr/bin/env python3
"""Validate current Case 1 MATLAB exports against a locked inventory manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=CASE_DIR / "parity_reference_inventory.json",
    )
    args = parser.parse_args()

    with args.inventory.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    failures: list[str] = []
    seen: set[Path] = set()

    for entry in manifest.get("files", []):
        path = Path(entry["path"])
        seen.add(path)
        if not path.exists():
            failures.append(f"missing: {path}")
            continue

        expected_size = int(entry["size_bytes"])
        actual_size = int(path.stat().st_size)
        if actual_size != expected_size:
            failures.append(f"size mismatch: {path} expected={expected_size} actual={actual_size}")

        expected_sha = str(entry["sha256"])
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            failures.append(f"sha mismatch: {path} expected={expected_sha} actual={actual_sha}")

    if failures:
        print("Case 1 reference inventory check: FAIL")
        for msg in failures:
            print(f" - {msg}")
        return 1

    print(
        "Case 1 reference inventory check: PASS",
        f"files={len(seen)}",
        f"inventory={args.inventory}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
