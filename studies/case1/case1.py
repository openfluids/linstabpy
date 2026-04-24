#!/usr/bin/env python3
"""Case 1 wrapper over the shared results runtime."""

from __future__ import annotations

import os
from pathlib import Path

from studies.common.results_runtime import compute as _runtime_compute
from studies.common.results_runtime import main as _runtime_main


SCRIPT = Path(__file__).resolve()


def _prepare_env() -> None:
    """Install Case 1 defaults expected by the shared runtime."""
    os.chdir(SCRIPT.parent)
    os.environ.setdefault("LINSTAB2D_CASE_LOG_PREFIX", "case1")
    os.environ.setdefault("LINSTAB2D_CASE1_CONFIG", str(SCRIPT.parent / "configs" / "parity_config.jsonc"))
    os.environ.setdefault("LINSTAB2D_CASE1_OUTPUT_NPZ", str(SCRIPT.with_suffix(".npz")))


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
