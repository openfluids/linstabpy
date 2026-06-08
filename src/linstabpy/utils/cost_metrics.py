# SPDX-License-Identifier: GPL-3.0-only
"""Uniform runtime cost metrics for saved result artifacts."""

from __future__ import annotations

import os
from math import nan

from petsc4py import PETSc


def _gpu_memory_bytes(cost_gpus: int) -> float:
    if cost_gpus <= 0:
        return nan
    try:
        import cupy as cp  # type: ignore[import-not-found]

        free_bytes, total_bytes = cp.cuda.Device().mem_info
        return float(total_bytes - free_bytes)
    except Exception:
        return nan


def _petsc_options_string() -> str:
    options = PETSc.Options().getAll()
    if not options:
        return ""
    return ";".join(f"{key}={value}" for key, value in sorted(options.items()))


def collect_cost_metrics(
    stage_wall: dict[str, float],
    *,
    git_sha: str,
    peak_rss_bytes: int,
) -> dict[str, object]:
    # peak_rss_bytes is the ALREADY cross-rank-aggregated value, computed by the
    # caller in the collective region (every rank) BEFORE the rank-0-only save
    # block. This function must contain NO MPI collective: it runs inside the
    # rank-0-only save path, so an allreduce here would deadlock every multi-rank
    # run. See feedback_mpi_collective_rank0 / bead 3.3.
    solve_times = [
        float(stage_wall[stage]) for stage in ("stability", "resolvent") if stage in stage_wall
    ]
    cost_gpus = int(os.getenv("LINSTAB2D_GPUS", "0") or 0)

    return {
        "cost_wall_total_s": float(sum(stage_wall.values())),
        "cost_wall_assemble_s": float(stage_wall.get("assemble_operator_bc", nan)),
        "cost_wall_factor_s": float(nan),
        "cost_wall_solve_s": float(sum(solve_times)) if solve_times else float(nan),
        "cost_peak_rss_bytes": int(peak_rss_bytes),
        "cost_peak_gpu_mem_bytes": _gpu_memory_bytes(cost_gpus),
        "cost_mpi_ranks": int(PETSc.COMM_WORLD.getSize()),
        "cost_gpus": cost_gpus,
        "cost_petsc_options": _petsc_options_string(),
        "cost_hardware_tag": str(os.getenv("LINSTAB2D_HARDWARE_TAG", "")),
        "cost_git_sha": str(git_sha),
    }
