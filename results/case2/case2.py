#!/usr/bin/env python3
"""Case 2 runner reusing the Case 1 SWBLI pipeline on the finer baseflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
CASE1_DIR = SCRIPT.parents[1] / "case1"
if str(CASE1_DIR) not in sys.path:
    sys.path.insert(0, str(CASE1_DIR))

from case1 import compute as _case1_compute
from case1 import main as _case1_main


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
    _case1_compute()


def main() -> None:
    """Cache-aware entrypoint."""
    _prepare_env()
    _case1_main()


if __name__ == "__main__":
    main()
