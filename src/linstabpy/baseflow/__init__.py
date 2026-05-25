# SPDX-License-Identifier: GPL-3.0-only
"""Base flow infrastructure: data loading, viscosity models."""

from .baseflow import BaseFlow
from .case1 import load_hamada_case
from .loaders import (
    load_blasius,
    load_cavity,
    load_jet,
    load_paraboloid,
    load_streaky_bl,
)
from .sponge import CASE1_SPONGE_PRESETS, case1_sponge_from_config, generate_case1_sponge
from .viscosity import sutherland_air
