# SPDX-License-Identifier: GPL-3.0-only
"""Canonical paths for the self-contained Song (2023) example (θ = 32°)."""

from __future__ import annotations

from pathlib import Path

SONG_EXAMPLE_DIR = Path(__file__).resolve().parent

REFERENCE_DIR = SONG_EXAMPLE_DIR / "reference"
BASEFLOW_NPZ = REFERENCE_DIR / "baseflow" / "song_32deg.npz"
LEADING_MODE_NPZ = REFERENCE_DIR / "eigenmode" / "song_32deg_leading_mode.npz"
SPECTRUM_EIGENVALUES_DAT = REFERENCE_DIR / "spectrum" / "eigenvalues.dat"
SPECTRUM_OMEGA_I_CSV = REFERENCE_DIR / "spectrum" / "omega_i.csv"

INPUTS_DIR = SONG_EXAMPLE_DIR / "inputs"
INPUTS_RAW_DIR = INPUTS_DIR / "raw"
INPUTS_BASEFLOW_DIR = INPUTS_DIR / "baseflow"

OUTPUT_DIR = SONG_EXAMPLE_DIR / "output"
FD4_NPZ = OUTPUT_DIR / "fd4_n30" / "song_2023.npz"

FIGURES_DIR = SONG_EXAMPLE_DIR / "figures"
FIGURES_BASEFLOW_DIR = FIGURES_DIR / "baseflow"
FIGURES_VALIDATION_DIR = FIGURES_DIR / "validation"

CONFIGS_DIR = SONG_EXAMPLE_DIR / "configs"
RUN_FD4_N30_YAML = CONFIGS_DIR / "run_fd4_n30.yaml"

# Song (2023) length scale [mm] applied to eigenvalues.dat columns.
LENGTH_SCALE = 80.0
# Hungarian |Δω| pairing tolerance for Song spectrum comparison (not operator dissipation).
SPECTRUM_MATCH_DELTA_OMEGA_DEFAULT = 0.01
# Back-compat alias used by plot/report scripts.
SPECTRUM_MATCH_EPS_DEFAULT = SPECTRUM_MATCH_DELTA_OMEGA_DEFAULT
