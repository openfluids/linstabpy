# SPDX-License-Identifier: GPL-3.0-only
"""Base-flow object, viscosity, and field loaders.

Use ``load_grid_fields`` for a npy directory or one npz. ``load_hamada_case``
keeps the curved-mesh stride guard. Martini examples use ``load_blasius`` and
the other named MATLAB loaders.
"""

from .baseflow import BaseFlow
from .case1 import load_hamada_case
from .grid_fields import load_grid_fields, read_grid_fields
from .loaders import (
    load_blasius,
    load_cavity,
    load_jet,
    load_paraboloid,
    load_streaky_bl,
)
from .sponge import CASE1_SPONGE_PRESETS, case1_sponge_from_config, generate_case1_sponge
from .viscosity import sutherland_air
