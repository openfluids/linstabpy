# SPDX-License-Identifier: GPL-3.0-only
"""Check that every rank of one frequency walk takes the same branch."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap

import pytest

from tests.mpi_helpers import run_mpi_python

_TIMEOUT_S = 60

_SETUP = textwrap.dedent(
    """\
    import json, os, sys, time
    import numpy as np
    import scipy.sparse as sp
    from mpi4py import MPI
    from linstabpy.config import Config
    import linstabpy.solvers.router.map_walk as mw

    os.environ.pop("LINSTABPY_JOB_DEADLINE", None)
    os.environ.pop("SLURM_JOB_END_TIME", None)
    rank = MPI.COMM_WORLD.Get_rank()
    diagonal = 2.7 + 0.22 * np.arange(8) + 1j * (0.17 - 0.025 * np.arange(8))
    lower = 0.045 - 0.025j + 0.003 * np.arange(7)
    upper = -0.035 + 0.02j - 0.002 * np.arange(7)
    L0 = sp.diags([lower, diagonal, upper], [-1, 0, 1], format="csc", dtype=complex)
    cfg = {"resolvent": {"eigen": {"eps_tol": 1e-10}}}
    walker = mw.LineWalker(solver_cfg=cfg, cfg_obj=Config.load(cfg), rtol=1.0e-8)
    walker.n_anchor = 1
    weight = np.ones(8)
    """
)

# Rank 0 accepts the walk. Rank 1 rejects it. The probe itself stays in step.
_CHEAP_SPLIT = _SETUP + textwrap.dedent(
    """\
    def split_cheap(**kwargs):
        return rank == 0

    try:
        walker.compute(L0, 0.8, weight, 1)
        walker.last_factor_s = 1.0e6
        mw.walk_is_cheaper = split_cheap
        point = walker.compute(L0, 0.8, weight, 1)
        print(json.dumps({"rank": rank, "source": point.source, "gain": point.gain}), flush=True)
    finally:
        walker.close()
    """
)

# Rank 1's clock jumps only after GMRES has started, on the check that sits
# directly before the collective preconditioner apply.
_CLOCK_SPLIT = _SETUP + textwrap.dedent(
    """\
    real_gmres = mw._right_preconditioned_gmres
    state = {"on": False, "n": 0}

    def wrapped_gmres(*args, **kwargs):
        state["on"] = True
        return real_gmres(*args, **kwargs)

    real_clock = time.perf_counter

    def skewed():
        now = real_clock()
        if rank == 1 and state["on"]:
            state["n"] += 1
            if state["n"] >= 2:
                return now + 1.0e9
        return now

    try:
        walker.compute(L0, 0.8, weight, 1)
        walker.last_factor_s = 1.0e6
        mw._right_preconditioned_gmres = wrapped_gmres
        time.perf_counter = skewed
        point = walker.compute(L0, 0.8, weight, 1)
        print(json.dumps({"rank": rank, "source": point.source, "gain": point.gain}), flush=True)
    finally:
        walker.close()
    """
)


def _reports(script: str) -> list[dict]:
    if shutil.which("mpiexec") is None:
        pytest.skip("SKIPPED: no mpiexec on PATH (MPI guard needs >1 rank)")
    try:
        proc = run_mpi_python(2, script, timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        tail = exc.stderr or ""
        if not isinstance(tail, str):
            tail = tail.decode("utf-8", errors="replace")
        pytest.fail("ranks took different branches and the run timed out\n" + tail[-2000:])
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stderr}"
    reports = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    assert len(reports) == 2, proc.stdout
    return sorted(reports, key=lambda item: item["rank"])


def _assert_same_branch(reports: list[dict]) -> None:
    assert [item["rank"] for item in reports] == [0, 1]
    assert reports[0]["source"] == reports[1]["source"]
    assert reports[0]["source"] != "walk"
    assert reports[0]["gain"] == pytest.approx(reports[1]["gain"], rel=1e-6, abs=1e-8)


def test_walk_branch_agrees_when_one_rank_rejects_the_probe() -> None:
    _assert_same_branch(_reports(_CHEAP_SPLIT))


def test_walk_branch_agrees_when_one_rank_clock_hits_the_deadline() -> None:
    reports = _reports(_CLOCK_SPLIT)
    _assert_same_branch(reports)
    assert reports[0]["source"] == "budget"
