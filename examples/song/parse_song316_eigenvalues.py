#!/usr/bin/env python3
"""Parse the 31.6 deg EVP reference spectrum (data courtesy of Z. Song).

The input file is the machine-readable eigenvalues.dat supplied with
``../data/song_316deg``. It is treated as read-only reference data: no digitizing,
refitting, or value editing. The parser is tolerant of UTF-8 BOMs and CRLF line
endings, expects exactly two numeric columns (omega_r, omega_i), sorts the rows
least-damped first (descending omega_i), and prints a clean table.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


EigenRow = tuple[float, float]


def load_eigenvalues(path: Path) -> list[EigenRow]:
    """Return [omega_r, omega_i] rows from *path*."""
    text = path.read_text(encoding="utf-8-sig")
    rows: list[EigenRow] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise SystemExit(
                f"{path}:{lineno}: expected exactly 2 columns (omega_r omega_i), "
                f"got {len(parts)}: {stripped!r}"
            )
        try:
            omega_r, omega_i = (float(parts[0]), float(parts[1]))
        except ValueError as exc:
            raise SystemExit(f"{path}:{lineno}: non-numeric row: {stripped!r}") from exc
        rows.append((omega_r, omega_i))

    if not rows:
        raise SystemExit(f"{path}: no eigenvalue rows found")

    if any(not (math.isfinite(omega_r) and math.isfinite(omega_i)) for omega_r, omega_i in rows):
        raise SystemExit(f"{path}: non-finite eigenvalue encountered")
    return rows


def sort_least_damped_first(eigs: list[EigenRow]) -> list[EigenRow]:
    """Sort by descending omega_i; tie-break by increasing |omega_r|."""
    return sorted(eigs, key=lambda row: (-row[1], abs(row[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to data/song_316deg/eigenvalues.dat")
    args = parser.parse_args()

    eigs = sort_least_damped_first(load_eigenvalues(args.path))
    lead = eigs[0]

    print(f"parsed {len(eigs)} eigenvalues from {args.path}")
    print(
        "least-damped mode: "
        f"omega_r={lead[0]: .15e} omega_i={lead[1]: .15e}"
    )
    print("# sorted least-damped first: index omega_r omega_i")
    for idx, (omega_r, omega_i) in enumerate(eigs, start=1):
        print(f"{idx:02d} {omega_r: .15e} {omega_i: .15e}")


if __name__ == "__main__":
    main()
