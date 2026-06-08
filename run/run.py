#!/usr/bin/env python3
"""Run-kit entrypoint over ``linstabpy.results_runtime``."""

from __future__ import annotations

import os
from pathlib import Path

from linstabpy.results_runtime import compute as _runtime_compute
from linstabpy.results_runtime import main as _runtime_main

SCRIPT = Path(__file__).resolve()


def _prepare_env() -> None:
    """Runtime env defaults for ``results_runtime``."""
    config_path = os.getenv("LINSTABPY_CASE_CONFIG", str(SCRIPT.parent / "parity_config.jsonc"))
    output_npz = os.getenv("LINSTABPY_CASE_OUTPUT_NPZ", str(SCRIPT.with_suffix(".npz")))
    os.environ.setdefault("LINSTABPY_CASE_LOG_PREFIX", "run")
    os.environ.setdefault("LINSTABPY_CASE_CONFIG", config_path)
    os.environ.setdefault("LINSTABPY_CASE_OUTPUT_NPZ", output_npz)


def compute() -> None:
    """Run the configured LSA/resolvent pipeline once."""
    _prepare_env()
    _runtime_compute()


def main() -> None:
    """Cache-aware entrypoint."""
    _prepare_env()
    _runtime_main()


if __name__ == "__main__":
    main()
