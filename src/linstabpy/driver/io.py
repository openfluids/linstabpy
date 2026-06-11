# SPDX-License-Identifier: GPL-3.0-only
"""Runtime result archive I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_results(
    *,
    output_npz: Path,
    heartbeat_seconds: float,
    payload: dict[str, object],
    peak_rss_bytes: int,
    stage_wall: dict[str, float] | None = None,
) -> None:
    """Persist the runtime NPZ payload under the save-results stage boundary.

    ``peak_rss_bytes`` is the cross-rank-aggregated peak RSS, computed by the
    caller in the collective region (every rank) before the rank-0-only save
    block. It is threaded in so ``collect_cost_metrics`` performs NO MPI
    collective here (this function runs rank-0-only; a collective would deadlock).

    ``stage_wall`` is the caller's accumulated per-stage wall-time map; it is
    threaded in because the orchestration stages now live in
    ``linstabpy.driver.run`` and update that module's accumulator, not this one.
    """
    from linstabpy.driver.run import (
        _STAGE_WALL,
        PETSc,
        _git_head_sha,
        _log_stage,
        _stage_heartbeat,
        _stage_start,
    )

    stage_start = _stage_start()
    with _stage_heartbeat("save_results", interval_seconds=heartbeat_seconds):
        from linstabpy.utils.cost_metrics import collect_cost_metrics

        cost = collect_cost_metrics(
            _STAGE_WALL if stage_wall is None else stage_wall,
            git_sha=_git_head_sha(),
            peak_rss_bytes=peak_rss_bytes,
        )
        payload = {**payload, **cost}
        np.savez_compressed(output_npz, **payload)
        if PETSc.COMM_WORLD.getRank() == 0:
            import json

            sidecar = output_npz.with_suffix(".metrics.json")
            serializable_cost = {}
            for k, v in cost.items():
                if isinstance(v, float) and np.isnan(v):
                    serializable_cost[k] = None
                elif isinstance(v, np.number):
                    serializable_cost[k] = v.item()
                else:
                    serializable_cost[k] = v
            with open(sidecar, "w") as fh:
                json.dump(serializable_cost, fh, indent=2, sort_keys=True)
    _log_stage("save_results", stage_start)


_stage_write_npz = save_results


def load_results(path: str | Path):
    """Load a runtime ``.npz`` archive without pickle support."""
    return np.load(path, allow_pickle=False)
