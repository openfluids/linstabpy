#!/usr/bin/env python3
"""Tiny-grid Blasius resolvent smoke test (PETSc / MPI / assembly timing).

Uses ``example_1_blasius_smoke.jsonc`` and writes ``example_1_blasius_smoke.*``
outputs only, so the full Example 1 artifacts are not clobbered.

Usage:
  .venv/bin/python -m examples.example_1_blasius_smoke
  mpiexec.mpich -n 1 .venv/bin/python -m examples.example_1_blasius_smoke
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np

import examples.example_1_blasius as bl

_SMOKE = Path(__file__).resolve()


def _comm():
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except Exception:
        return None


def _rank_size() -> tuple[int, int]:
    comm = _comm()
    if comm is None:
        return 0, 1
    return int(comm.Get_rank()), int(comm.Get_size())


def _broadcast_flag(flag: bool) -> bool:
    comm = _comm()
    if comm is None or comm.Get_size() == 1:
        return bool(flag)
    return bool(comm.bcast(bool(flag), root=0))


def _barrier() -> None:
    comm = _comm()
    if comm is not None and comm.Get_size() > 1:
        comm.Barrier()


def _rank_npz_path(rank: int) -> Path:
    return _SMOKE.with_name(f"{_SMOKE.stem}_rank{rank:04d}.npz")


def _patch_paths(rank: int = 0) -> None:
    bl.SCRIPT = _SMOKE
    bl.OUTPUT_NPZ = _SMOKE.with_suffix(".npz") if rank == 0 else _rank_npz_path(rank)
    bl.OUTPUT_PNG = _SMOKE.with_suffix(".png")
    bl.OUTPUT_BASEFLOW = bl.OUTPUT_PNG.with_name(f"{bl.OUTPUT_PNG.stem}_baseflow.png")
    bl.OUTPUT_SUPPORT = bl.OUTPUT_PNG.with_name(f"{bl.OUTPUT_PNG.stem}_support_components.png")
    bl.OUTPUT_SUPPORT_SHARED = bl.OUTPUT_PNG.with_name(
        f"{bl.OUTPUT_PNG.stem}_support_components_shared_limits.png"
    )


def main() -> None:
    rank, _ = _rank_size()
    _patch_paths(rank)
    cfg = bl.load_jsonc(bl.SCRIPT.with_suffix(".jsonc"))
    need_recompute = False
    data = None
    if rank == 0:
        try:
            data = np.load(_SMOKE.with_suffix(".npz"))
        except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
            print(f"Cached results unavailable or invalid ({exc}), recomputing.")
            need_recompute = True
    need_recompute = _broadcast_flag(need_recompute)
    if need_recompute:
        bl.compute(cfg)
        _barrier()
    if rank != 0:
        try:
            os.remove(_rank_npz_path(rank))
        except FileNotFoundError:
            pass
        return
    if data is None or need_recompute:
        data = np.load(_SMOKE.with_suffix(".npz"))
    bl.plot(data)


if __name__ == "__main__":
    main()
