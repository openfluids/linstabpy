"""Test that _apply_inv_B_petsc raises RuntimeError for axy + distributed PETSc.

G-001 guard: axisymmetric model must fail fast with a clean RuntimeError
when the PETSc matrix spans multiple MPI ranks, rather than silently
corrupting data via petsc_to_scipy on a distributed matrix.
"""
from __future__ import annotations

import textwrap
import pytest

from tests.mpi_helpers import run_mpi_python


_SCRIPT = textwrap.dedent("""\
    import sys, json
    from petsc4py import PETSc
    import numpy as np
    import scipy.sparse as sp

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    # Build a minimal axy-like distributed PETSc matrix.
    # 10 rows, distributed across ranks.
    n = 10
    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes([n, n])
    mat.setType("mpiaij")
    mat.setPreallocationNNZ(1)
    r0, r1 = mat.getOwnershipRange()
    for i in range(r0, r1):
        mat.setValue(i, i, 1.0 + 0j)
    mat.assemblyBegin()
    mat.assemblyEnd()

    from linstab2d.operators.assembly import _apply_inv_B_petsc

    # Dummy base-flow arrays (values don't matter for the guard)
    ngp = n // 5 if n >= 5 else 1
    dummy = np.ones(ngp)

    result = {}
    try:
        _apply_inv_B_petsc(
            mat, "axy", ngp,
            RHO=dummy, U=dummy, V=dummy, W=dummy, T=dummy,
            kappa=1.4, Ma=0.5,
        )
        result["raised"] = False
        result["error"] = None
    except RuntimeError as exc:
        result["raised"] = True
        result["error"] = str(exc)
    except Exception as exc:
        result["raised"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        mat.destroy()

    if rank == 0:
        print(json.dumps(result))
    sys.stdout.flush()
""")


@pytest.mark.slow
def test_axy_distributed_raises_runtime_error():
    """_apply_inv_B_petsc must raise RuntimeError for axy + 2 MPI ranks."""
    proc = run_mpi_python(2, _SCRIPT)
    assert proc.returncode == 0, (
        f"MPI script crashed (rc={proc.returncode}):\n{proc.stderr}"
    )
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    assert lines, f"No JSON output from MPI script:\n{proc.stdout}\n{proc.stderr}"
    import json
    report = json.loads(lines[0])
    assert report["raised"], (
        f"Expected RuntimeError but got: {report['error']}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "axy" in report["error"].lower() or "axisymmetric" in report["error"].lower(), (
        f"Error message should mention 'axy': {report['error']}"
    )
