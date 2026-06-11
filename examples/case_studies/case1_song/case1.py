#!/usr/bin/env python3
"""Case 1 wrapper over the shared results runtime.

The case file owns only working-directory and environment defaults. Numerical
stage ordering, solver policy, postchecks, and NPZ persistence stay in
``linstabpy.driver``.
"""

from __future__ import annotations

import os
from pathlib import Path

from linstabpy.driver.cli import main as _runtime_main
from linstabpy.driver.run import run as _runtime_compute

SCRIPT = Path(__file__).resolve()


def _prepare_env() -> None:
    """Install Case 1 defaults expected by the shared runtime."""
    os.chdir(SCRIPT.parent)
    os.environ.setdefault("LINSTABPY_CASE_LOG_PREFIX", "case1")
    os.environ.setdefault(
        "LINSTABPY_CASE1_CONFIG", str(SCRIPT.parent / "configs" / "parity_config.jsonc")
    )
    os.environ.setdefault("LINSTABPY_CASE1_OUTPUT_NPZ", str(SCRIPT.with_suffix(".npz")))


def compute() -> None:
    """Run the Case 1 workflow."""
    _prepare_env()
    _runtime_compute()


def main() -> None:
    """Cache-aware entrypoint."""
    _prepare_env()
    _runtime_main()


if __name__ == "__main__":
    main()
