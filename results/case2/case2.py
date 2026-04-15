#!/usr/bin/env python3
"""Case 2 runner reusing the Case 1 SWBLI pipeline on the finer baseflow."""

from __future__ import annotations

import os
from pathlib import Path

from linstab2d.results_runtime import compute as _runtime_compute
from linstab2d.results_runtime import main as _runtime_main


SCRIPT = Path(__file__).resolve()


def _prepare_env() -> None:
    """Map Case 2 defaults onto the existing Case 1 runtime knobs."""
    os.environ.setdefault("LINSTAB2D_CASE_LOG_PREFIX", "case2")
    os.environ.setdefault(
        "LINSTAB2D_CASE1_CONFIG",
        os.getenv("LINSTAB2D_CASE2_CONFIG", str(SCRIPT.parent / "parity_config.jsonc")),
    )
    os.environ.setdefault(
        "LINSTAB2D_CASE1_OUTPUT_NPZ",
        os.getenv("LINSTAB2D_CASE2_OUTPUT_NPZ", str(SCRIPT.with_suffix(".npz"))),
    )


def compute() -> None:
    """Run the Case 2 workflow."""
    _prepare_env()
    _runtime_compute()


def main() -> None:
    """Cache-aware entrypoint."""
    _prepare_env()
    _runtime_main()


if __name__ == "__main__":
    main()
