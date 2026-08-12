# SPDX-License-Identifier: GPL-3.0-only
"""Lui 2022 suction-side SWBLI on the packed stride-2 mesh.

Usage (from the repository root):
    mpiexec -n 4 python -m examples.sawtooth.sawtooth
"""

from __future__ import annotations

import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "sawtooth_evp.yaml"
DEFAULT_OUTPUT = HERE / "sawtooth_evp.npz"
LOG_PATH = HERE / "sawtooth.log"


def _prepare_env() -> None:
    """Point the shared driver at the sawtooth YAML and output npz."""
    os.environ.setdefault("LINSTABPY_CASE_CONFIG", str(DEFAULT_CONFIG))
    os.environ.setdefault("LINSTABPY_CASE_OUTPUT_NPZ", str(DEFAULT_OUTPUT))
    os.environ.setdefault("LINSTABPY_CASE_LOG_PREFIX", "sawtooth")


def _log(line: str) -> None:
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def compute() -> None:
    """Run the EVP through the shared driver and record wall time."""
    _prepare_env()
    from linstabpy.driver.run import run

    t0 = time.perf_counter()
    run()
    elapsed = time.perf_counter() - t0
    _log(f"sawtooth EVP  {elapsed:.3f} s")
    _log(f"config  {os.environ['LINSTABPY_CASE_CONFIG']}")
    _log(f"output  {os.environ['LINSTABPY_CASE_OUTPUT_NPZ']}")


def main() -> None:
    """Cache-aware entry: recompute only when the output npz is missing."""
    _prepare_env()
    from linstabpy.driver.cli import main as _runtime_main

    t0 = time.perf_counter()
    _runtime_main()
    elapsed = time.perf_counter() - t0
    _log(f"sawtooth driver  {elapsed:.3f} s")


if __name__ == "__main__":
    main()
