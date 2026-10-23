# SPDX-License-Identifier: GPL-3.0-only
"""MPI guard: only rank 0 writes the resolvent-map store.

Every rank holds the same map. Before the fix every rank wrote the store.
The ranks then raced on one temporary file, and all but one failed with FileNotFoundError.
"""

from __future__ import annotations

import json
import shutil
import textwrap

import pytest

from tests.mpi_helpers import run_mpi_python

_SCRIPT = textwrap.dedent(
    """\
    import json, sys, tempfile
    from pathlib import Path
    import numpy as np
    from petsc4py import PETSc
    import importlib
    rm = importlib.import_module("linstabpy.solvers.router.resolvent_map")

    calls = []
    rm.atomic_savez = lambda path, **payload: calls.append(str(path))
    beta, f = np.array([0.1]), np.array([1.0, 2.0])
    shape = (1, 2)
    rm._save_store(
        Path(tempfile.gettempdir()) / "unused_store.npz",
        beta=beta, f=f, gain=np.ones(shape), regime_label=np.full(shape, "stable"),
        reason_code=np.zeros(shape, dtype=np.int16), discount_gamma=np.zeros(shape),
        failure_records=[], point_records=[], sign_convention="physical", fingerprint={},
    )
    print(json.dumps({"rank": PETSc.COMM_WORLD.getRank(), "writes": len(calls)}))
    sys.stdout.flush()
    """
)


@pytest.mark.slow
def test_only_rank_zero_writes_the_store() -> None:
    if shutil.which("mpiexec") is None:
        pytest.skip("SKIPPED: no mpiexec on PATH (MPI guard needs >1 rank)")
    proc = run_mpi_python(2, _SCRIPT)
    assert proc.returncode == 0, f"MPI script crashed (rc={proc.returncode}):\n{proc.stderr}"
    reports = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    writes = {r["rank"]: r["writes"] for r in reports}
    assert writes == {0: 1, 1: 0}, writes
