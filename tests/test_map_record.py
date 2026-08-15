# SPDX-License-Identifier: GPL-3.0-only
"""Check map point records flush each row and resume unfinished work."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from linstabpy.solvers.router.map_record import (
    append_row,
    atomic_savez,
    finished_pairs,
    job_remaining_s,
    last_factor_seconds,
    remaining,
    time_left_for_factor,
)


def test_append_row_then_resume_skips_finished_frequencies(tmp_path: Path) -> None:
    path = tmp_path / "line.csv"
    fields = ("beta_Xsh", "f_Xsh_Uinf", "gain")
    append_row(path, fields, (40.0, 1.0, 92.89))
    append_row(path, fields, (40.0, 1.4, 156.68))
    done = finished_pairs(path, beta_field="beta_Xsh", f_field="f_Xsh_Uinf")
    left = remaining([1.0, 1.4, 1.6, 1.8], done, beta=40.0)
    assert left == [1.6, 1.8]


def test_atomic_savez_replaces_the_store(tmp_path: Path) -> None:
    path = tmp_path / "map.npz"
    atomic_savez(path, gain=np.asarray([1.0, 2.0]))
    atomic_savez(path, gain=np.asarray([3.0, 4.0]))
    with np.load(path) as data:
        np.testing.assert_allclose(data["gain"], [3.0, 4.0])
    assert not path.with_name(path.stem + ".tmp.npz").exists()


def test_last_factor_seconds_reads_the_last_positive_factor(tmp_path: Path) -> None:
    path = tmp_path / "line.csv"
    fields = ("beta_Xsh", "f_Xsh_Uinf", "factor_s")
    append_row(path, fields, (40.0, 1.0, 556.6))
    append_row(path, fields, (40.0, 1.4, 0.0))
    append_row(path, fields, (40.0, 1.6, 844.4))
    assert last_factor_seconds(path) == pytest.approx(844.4)


def test_time_left_for_factor_blocks_a_short_job() -> None:
    assert time_left_for_factor(remaining_s=None, need_s=60.0) is True
    assert time_left_for_factor(remaining_s=120.0, need_s=60.0) is True
    assert time_left_for_factor(remaining_s=10.0, need_s=60.0) is False
    assert time_left_for_factor(remaining_s=10.0, need_s=0.0) is True


def test_job_remaining_s_reads_deadline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINSTABPY_JOB_DEADLINE", str(time.time() + 25.0))
    left = job_remaining_s()
    assert left is not None
    assert 10.0 < left < 30.0
