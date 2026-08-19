# SPDX-License-Identifier: GPL-3.0-only
"""MPI regression guard for petsc_vec: no replication across ranks.

petsc_vec converts a full-global numpy array (every rank holds the whole
vector) into a PETSc Vec.
The original implementation called ``createWithArray`` with no
communicator/size split.
Under MPI each rank then treated its entire copy as a local slice.
The resulting vec had global size ``N * nranks``.
The data replicated across ranks.
Fed into a distributed operator, that oversized vec injected silent error.

The entire unit suite is otherwise serial, which is why that failure hid
behind green tests.
This guard spawns real mpiexec subprocesses so the replication failure
mode runs at >1 rank.
The size check is deterministic and launcher-independent.
A correct petsc_vec yields global size N at any rank count.
The old path yields N * nranks.
"""

from __future__ import annotations

import json
import shutil
import textwrap

import pytest

from tests.mpi_helpers import run_mpi_python

_N = 100_000

_SCRIPT = textwrap.dedent(
    f"""\
    import json, sys
    import numpy as np
    from petsc4py import PETSc
    from linstabpy.utils.petsc_core import petsc_vec, vec_to_numpy

    rank = PETSc.COMM_WORLD.getRank()
    N = {_N}
    arr = (np.arange(N) + 1j * np.arange(N)).astype(complex)
    v = petsc_vec(arr)
    report = {{
        "rank": rank,
        "global_size": int(v.getSize()),
        "expected": N,
    }}
    # Round-trip must reconstruct the original full-global array exactly.
    back = vec_to_numpy(v)
    report["roundtrip_exact"] = bool(
        back.shape[0] == N and np.array_equal(back, arr)
    )
    v.destroy()
    if rank == 0:
        print(json.dumps(report))
    sys.stdout.flush()
    """
)


@pytest.mark.slow
@pytest.mark.parametrize("n_ranks", [2, 4])
def test_petsc_vec_no_replication_under_mpi(n_ranks):
    if shutil.which("mpiexec") is None:
        pytest.skip("SKIPPED: no mpiexec on PATH (MPI regression guard needs >1 rank)")
    proc = run_mpi_python(n_ranks, _SCRIPT)
    assert proc.returncode == 0, f"MPI script crashed (rc={proc.returncode}):\n{proc.stderr}"
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    assert lines, f"No JSON output from MPI script:\n{proc.stdout}\n{proc.stderr}"
    report = json.loads(lines[0])
    assert report["global_size"] == report["expected"], (
        f"petsc_vec replicated the array under MPI: global size "
        f"{report['global_size']} != {report['expected']} (= N * {n_ranks} bug)"
    )
    assert report["roundtrip_exact"], "petsc_vec -> vec_to_numpy round-trip was not exact"
