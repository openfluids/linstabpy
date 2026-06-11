# SPDX-License-Identifier: GPL-3.0-only
"""CLI smoke tests."""

from __future__ import annotations

import subprocess
import sys
from importlib import metadata


def test_driver_cli_version_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "linstabpy.driver.cli", "--version"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == metadata.version("linstabpy")
    assert result.stderr == ""
