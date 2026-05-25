# SPDX-License-Identifier: GPL-3.0-only
"""Shared MPI subprocess helpers for test harnesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def run_mpi_python(n_ranks: int, script: str) -> subprocess.CompletedProcess[str]:
    """Launch a Python script under mpiexec and return the completed process."""
    # Pass env explicitly: when petsc4py is already imported in the parent,
    # inheriting os.environ implicitly via subprocess causes mpiexec to exit
    # with code 1 and no output. An explicit dict copy avoids this.
    return subprocess.run(
        ["mpiexec", "-n", str(n_ranks), sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )


def parse_mpi_reports(stdout: str) -> list[dict]:
    """Extract JSON report lines emitted by MPI rank scripts."""
    return [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]


def assert_split_ownership(reports: list[dict], n_ranks: int) -> None:
    """Verify that MPI ownership ranges partition the matrix correctly."""
    assert len(reports) == n_ranks, reports
    reports = sorted(reports, key=lambda item: item["rank"])
    assert {report["rank"] for report in reports} == set(range(n_ranks))

    ranges = [tuple(report["ownership_range"]) for report in reports]
    assert len(set(ranges)) == n_ranks
    assert ranges[0][0] == 0
    for prev, curr in zip(ranges, ranges[1:]):
        assert prev[1] == curr[0], ranges

    nrows = reports[0]["nrows"]
    assert ranges[-1][1] == nrows
    assert sum(r1 - r0 for r0, r1 in ranges) == nrows
    assert any(r0 > 0 for r0, _ in ranges)
