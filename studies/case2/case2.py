#!/usr/bin/env python3
"""Case 2 wrapper over the shared results runtime."""

from __future__ import annotations

import os
from pathlib import Path

from linstab2d.results_runtime import compute as _runtime_compute
from linstab2d.results_runtime import main as _runtime_main


SCRIPT = Path(__file__).resolve()


def _prepare_env() -> None:
    """Install Case 2 defaults expected by the shared runtime."""
    os.chdir(SCRIPT.parent)
    config_path = os.getenv("LINSTAB2D_CASE2_CONFIG", str(SCRIPT.parent / "configs" / "parity_config.jsonc"))
    output_npz = os.getenv("LINSTAB2D_CASE2_OUTPUT_NPZ", str(SCRIPT.with_suffix(".npz")))
    os.environ.setdefault("LINSTAB2D_CASE_LOG_PREFIX", "case2")
    os.environ.setdefault("LINSTAB2D_CASE2_CONFIG", config_path)
    os.environ.setdefault("LINSTAB2D_CASE2_OUTPUT_NPZ", output_npz)
    os.environ.setdefault("LINSTAB2D_CASE_CONFIG", config_path)
    os.environ.setdefault("LINSTAB2D_CASE_OUTPUT_NPZ", output_npz)


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
