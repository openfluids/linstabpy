# SPDX-License-Identifier: GPL-3.0-only
"""Canonical paths for the self-contained Savarino resolvent example (θ = 30.8°)."""

from __future__ import annotations

import os
from pathlib import Path

SAVARINO_EXAMPLE_DIR = Path(__file__).resolve().parent

REFERENCE_DIR = SAVARINO_EXAMPLE_DIR / "reference"

# Shipped reference data (git).
BASEFLOW_NAME = "state_atcenter_mesh300_y300_xend195_angle308_T157_Mach215.dat"
BASEFLOW_DAT = REFERENCE_DIR / "baseflow" / BASEFLOW_NAME

# Local outputs (mostly gitignored; see .gitignore).
OUTPUT_DIR = SAVARINO_EXAMPLE_DIR / "output"
OUTPUT_FIGURES_DIR = OUTPUT_DIR / "figures"
# 200-point map of record (per-point jobs); savarino_map3_lines_200pt.csv is the
# line-driver run of the same points. map2 is the superseded earlier campaign that
# the line driver's farm_gain / rel_err columns compare against (see README step 6).
GAIN_MAP_CSV = SAVARINO_EXAMPLE_DIR / "savarino_map3_200pt.csv"
GAIN_MAP_LINES_CSV = SAVARINO_EXAMPLE_DIR / "savarino_map3_lines_200pt.csv"
GAIN_MAP2_CSV = OUTPUT_DIR / "savarino_map2_200pt.csv"
GAIN_MAP_317_CSV = OUTPUT_DIR / "savarino_map317pt.csv"
GAIN_MAP_PNG = OUTPUT_FIGURES_DIR / "savarino_gain_map.png"
COMPARE_MODE_MACK_PNG = OUTPUT_FIGURES_DIR / "compare_mode_mack.png"
COMPARE_MODE_LIFTUP_PNG = OUTPUT_FIGURES_DIR / "compare_mode_liftup.png"

GRID_CSV = REFERENCE_DIR / "savarino_grid.csv"
FORCING_MASK_CSV = REFERENCE_DIR / "savarino_forcing_mask_measured.csv"

SAVARINO_RESPONSE_MACK_NPZ = REFERENCE_DIR / "savarino_response_mack.npz"
SAVARINO_RESPONSE_LIFTUP_NPZ = REFERENCE_DIR / "savarino_response_liftup.npz"

# Local converted base flow for case_dir (gitignored; see savarino_to_npy.py).
BASEFLOW_NPY_DIR = SAVARINO_EXAMPLE_DIR / "output" / "baseflow"

# Local linstabpy resolvent outputs for mode comparison (gitignored).
OUTPUT_COMPARE_DIR = SAVARINO_EXAMPLE_DIR / "output" / "compare"
MACK_OUTPUT_NPZ = OUTPUT_COMPARE_DIR / "mack" / "sav_map_b45p0_f2.npz"
LIFTUP_OUTPUT_NPZ = OUTPUT_COMPARE_DIR / "liftup" / "sav_map_b163p0_f0p0001.npz"

# Optional extra Zenodo downloads (gitignored; see fetch_zenodo.py).
ZENODO_DIR = REFERENCE_DIR / "zenodo"
ZENODO_MODES_DIR = ZENODO_DIR / "modes"

# Legacy aliases.
CONVERTED_DIR = BASEFLOW_NPY_DIR
REFERENCE_FIGURES_DIR = OUTPUT_FIGURES_DIR
MODES_SHIP_DIR = REFERENCE_DIR


def zenodo_root() -> Path:
    """Root for optional Zenodo mode archives; override with ``$SAVARINO_DATA``."""
    return Path(os.environ.get("SAVARINO_DATA", ZENODO_DIR))


def zenodo_baseflow_dat() -> Path:
    """Published Tecplot base flow (downloaded by ``fetch_zenodo.py baseflow``)."""
    override = os.environ.get("SAVARINO_BASE_DAT")
    if override:
        return Path(override)
    return BASEFLOW_DAT


def zenodo_modes_dir() -> Path:
    return Path(os.environ.get("SAVARINO_MODES", zenodo_root() / "modes"))
