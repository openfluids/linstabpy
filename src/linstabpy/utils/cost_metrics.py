# SPDX-License-Identifier: GPL-3.0-only
"""Uniform runtime cost metrics for saved result artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from math import nan

try:
    import resource as _resource
except ImportError:  # pragma: no cover - the resource module is POSIX-only
    _resource = None


def peak_rss_bytes() -> int:
    """Return this process's peak resident set size in bytes.

    Linux reports ``ru_maxrss`` in kilobytes; macOS reports bytes.
    Returns 0 when the ``resource`` module is not available.
    """
    if _resource is None:
        return 0
    rss = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _petsc_comm():
    """Return PETSc.COMM_WORLD, or None when PETSc is not installed."""
    try:
        import mpi4py.MPI  # noqa: F401
        from petsc4py import PETSc

        return PETSc.COMM_WORLD
    except Exception:
        return None


def _gpu_memory_bytes_cupy() -> float:
    """Device-wide used GPU memory via cupy, or nan when cupy is unavailable."""
    try:
        import cupy as cp  # type: ignore[import-not-found]

        free_bytes, total_bytes = cp.cuda.Device().mem_info
        return float(total_bytes - free_bytes)
    except Exception:
        return nan


def _gpu_memory_bytes_nvidia_smi() -> float:
    """Device-wide used GPU memory via nvidia-smi, or nan when it is unusable.

    Fallback for builds without cupy.
    ``nvidia-smi`` ships with the driver, so it is present wherever CUDA runs.
    It exposes only a byte count -- no host name, no device serial.
    That does not weaken the policy that artifacts carry no machine identity
    (see ``cost_hardware_tag`` below).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return nan
    values = [line.strip() for line in out.splitlines() if line.strip()]
    if not values:
        return nan
    try:
        # Device 0, matching cupy's cp.cuda.Device() default. MiB -> bytes.
        return float(values[0]) * 1024.0 * 1024.0
    except ValueError:
        return nan


def _gpu_memory_bytes(cost_gpus: int) -> float:
    """Used GPU memory for a GPU run, nan when cost_gpus is 0.

    NOT a peak, despite the field name this feeds.
    It is a single device-wide reading taken at artifact write.
    It includes memory held by other processes on the same card.
    Treat it as an order-of-magnitude occupancy figure, not this run's
    high-water mark.

    Previously this had a cupy-only implementation wrapped in a bare
    ``except Exception``.
    On any build without cupy, including the reference CPU and GPU
    virtual environments, it returned nan for every run.
    That nan was indistinguishable from a run with no GPU.
    A silent hole that looks like a legitimate value is the worst shape
    for a provenance field.
    So nvidia-smi runs before giving up.
    """
    if cost_gpus <= 0:
        return nan
    value = _gpu_memory_bytes_cupy()
    if value == value:  # not nan
        return value
    return _gpu_memory_bytes_nvidia_smi()


def _petsc_options_string() -> str:
    try:
        from petsc4py import PETSc
    except Exception:
        return ""
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
    # block. The body must contain NO MPI collective: it runs inside the
    # rank-0-only save path, so an allreduce here would deadlock every multi-rank
    # run.
    solve_times = [
        float(stage_wall[stage]) for stage in ("stability", "resolvent") if stage in stage_wall
    ]
    cost_gpus = int(os.getenv("LINSTABPY_GPUS") or 0)

    return {
        "cost_wall_total_s": float(sum(stage_wall.values())),
        "cost_wall_assemble_s": float(stage_wall.get("assemble_operator_bc", nan)),
        "cost_wall_factor_s": float(nan),
        "cost_wall_solve_s": float(sum(solve_times)) if solve_times else float(nan),
        "cost_peak_rss_bytes": int(peak_rss_bytes),
        "cost_peak_gpu_mem_bytes": _gpu_memory_bytes(cost_gpus),
        "cost_mpi_ranks": int(_petsc_comm().getSize()) if _petsc_comm() is not None else 1,
        "cost_gpus": cost_gpus,
        "cost_petsc_options": _petsc_options_string(),
        "cost_hardware_tag": str(os.getenv("LINSTABPY_HARDWARE_TAG") or "unknown"),
        "cost_git_sha": str(git_sha),
    }
