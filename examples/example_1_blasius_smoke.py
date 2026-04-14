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

import numpy as np

import examples.example_1_blasius as bl

_SMOKE = Path(__file__).resolve()


def _patch_paths() -> None:
    bl.SCRIPT = _SMOKE
    bl.OUTPUT_NPZ = _SMOKE.with_suffix(".npz")
    bl.OUTPUT_PNG = _SMOKE.with_suffix(".png")
    bl.OUTPUT_BASEFLOW = bl.OUTPUT_PNG.with_name(f"{bl.OUTPUT_PNG.stem}_baseflow.png")
    bl.OUTPUT_SUPPORT = bl.OUTPUT_PNG.with_name(f"{bl.OUTPUT_PNG.stem}_support_components.png")
    bl.OUTPUT_SUPPORT_SHARED = bl.OUTPUT_PNG.with_name(
        f"{bl.OUTPUT_PNG.stem}_support_components_shared_limits.png"
    )


def main() -> None:
    _patch_paths()
    cfg = bl.load_jsonc(bl.SCRIPT.with_suffix(".jsonc"))
    try:
        data = np.load(bl.OUTPUT_NPZ)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        bl.compute(cfg)
        data = np.load(bl.OUTPUT_NPZ)
    bl.plot(data)


if __name__ == "__main__":
    main()
