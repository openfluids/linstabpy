# SPDX-License-Identifier: GPL-3.0-only
"""Check the Savarino map-walk driver against the farm CSV."""

from __future__ import annotations

from pathlib import Path

from examples.savarino.gen_campaign import x_max
from examples.savarino.savarino_mapwalk import FARM_CSV, FIELDS, load_farm, relative_error

DRIVER = Path(__file__).resolve().parents[1] / "examples/savarino/savarino_mapwalk.py"


def test_farm_beta40_comes_from_committed_csv() -> None:
    rows = [row for row in load_farm(FARM_CSV) if abs(row["beta"] - 40.0) < 1.0e-12]
    assert [row["f"] for row in rows] == [1.0, 1.4, 1.6, 1.8, 2.0, 2.1, 2.15, 2.2, 2.5]
    assert abs(sum(row["wall_s"] for row in rows) - 8700.38) < 0.01
    assert FARM_CSV.name == "savarino_map2_200pt.csv"


def test_farm_forcing_window_switches_at_f2() -> None:
    assert x_max(1.0) == 1.7954167
    assert x_max(2.0) == 1.9470833


def test_driver_uses_linewalker_not_rsvd() -> None:
    src = DRIVER.read_text(encoding="utf-8")
    assert "LineWalker" in src
    assert "append_row" in src
    assert "certified_rsvd" not in src
    assert "sweep_stable_regime" not in src
    assert relative_error(10.0, 10.0) == 0.0


def test_every_row_states_its_build_and_box() -> None:
    assert FIELDS[-3:] == ("git_sha", "slurm_job", "machine")


def test_walk_rows_carry_the_farm_cost_columns() -> None:
    """A cost comparison needs the same measurements on both sides."""
    farm_cols = FARM_CSV.read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in ("wall_s", "peak_rss_bytes", "ndof", "ranks", "git_sha", "machine"):
        assert column in farm_cols
        assert column in FIELDS


def test_walk_peak_rss_means_the_same_as_the_farm_column() -> None:
    """Farm rows sum ru_maxrss over ranks (driver/run.py). A same-named
    column with a rank-local value would read 4x too small on 4 ranks and
    no reader could tell. The reduction is collective, so it must sit
    outside the rank-0 block; a serial test can only inspect the source."""
    src = DRIVER.read_text(encoding="utf-8")
    assert "comm.allreduce(int(_peak_rss_bytes()), op=MPI.SUM)" in src
    assert "peak_rss_total," in src
    reduce_at = src.index("comm.allreduce(int(_peak_rss_bytes())")
    guard_at = src.index("if rank == 0:", reduce_at)
    append_at = src.index("append_row(", reduce_at)
    assert reduce_at < guard_at < append_at
