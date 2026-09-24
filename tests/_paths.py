# SPDX-License-Identifier: GPL-3.0-only
"""Canonical repository paths for tests and validation fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from examples.song.reference_paths import BASEFLOW_NPZ

REPO_ROOT = Path(__file__).resolve().parents[1]
MARTINI_ROOT = REPO_ROOT / "examples" / "linstab2d"
SONG_ROOT = REPO_ROOT / "examples" / "song"
SAVARINO_ROOT = REPO_ROOT / "examples" / "savarino"
MARTINI_FIXTURE_ROOT = MARTINI_ROOT


def require_existing(path: Path, description: str) -> Path:
    if not path.exists():
        pytest.skip(f"Missing {description}: {path}", allow_module_level=True)
    return path


def martini_fixture(filename: str) -> Path:
    """Return the canonical Martini parity reference path."""
    subdir = MARTINI_FIXTURE_ROOT / "LinStab2DResults" / filename
    if subdir.is_file():
        return subdir
    return MARTINI_FIXTURE_ROOT / filename


def martini_export(filename: str) -> Path:
    """Return the optional raw Martini MATLAB export path."""
    return REPO_ROOT / "martini_examples" / filename


def unpack_song_baseflow() -> Path:
    """Write song_32deg.npz fields to a temporary npy directory."""
    src = BASEFLOW_NPZ
    require_existing(src, "Song 32 deg base flow")
    root = Path(tempfile.mkdtemp(prefix="song32_"))
    packed = np.load(src)
    for name in ("x", "y", "rho", "u", "v", "w", "T", "sponge"):
        np.save(root / f"{name}.npy", packed[name])
    return root
