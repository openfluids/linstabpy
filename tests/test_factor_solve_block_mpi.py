# SPDX-License-Identifier: GPL-3.0-only
"""MPI guard for the direct block solve (factorize ``solve_block``).

The block solve packs the right-hand sides into a PETSc dense matrix.
Under MPI each rank owns a row block of that matrix.
We run the solve at 2 and 3 ranks.
The checks are the use of the block path (no column-by-column fallback)
and the solution against a dense solve.
"""

from __future__ import annotations

import json
import shutil
import textwrap

import pytest

from tests.mpi_helpers import run_mpi_python

_SCRIPT = textwrap.dedent(
    """\
    import json, sys
    import numpy as np
    import scipy.sparse as sp
    from petsc4py import PETSc
    from linstabpy.linalg.factorize import factorize

    comm = PETSc.COMM_WORLD
    n, k = 40, 5
    rng = np.random.default_rng(0)
    A = sp.random(n, n, density=0.2, random_state=1) + 5 * sp.eye(n)
    A = (A + 1j * sp.random(n, n, density=0.1, random_state=2)).tocsr()
    B = rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
    lu = factorize(A, solver_cfg={})
    X = lu.solve_block(B)
    report = {
        "err": float(np.abs(X - np.linalg.solve(A.toarray(), B)).max()),
        "block_calls": lu.metrics["solve_block_calls"],
        "fallback_calls": lu.metrics["solve_block_fallback_calls"],
    }
    if comm.getRank() == 0:
        print(json.dumps(report))
    sys.stdout.flush()
    """
)


@pytest.mark.slow
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_solve_block_dense_layout_under_mpi(n_ranks: int) -> None:
    if shutil.which("mpiexec") is None:
        pytest.skip("SKIPPED: no mpiexec on PATH (MPI guard needs >1 rank)")
    proc = run_mpi_python(n_ranks, _SCRIPT)
    assert proc.returncode == 0, f"MPI script crashed (rc={proc.returncode}):\n{proc.stderr}"
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    assert lines, f"No JSON output from MPI script:\n{proc.stdout}\n{proc.stderr}"
    report = json.loads(lines[0])
    assert report["block_calls"] == 1, report
    assert report["fallback_calls"] == 0, report
    assert report["err"] < 1e-12, report
