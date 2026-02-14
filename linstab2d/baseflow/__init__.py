"""Base flow infrastructure: data loading, viscosity models."""

from .baseflow import BaseFlow
from .viscosity import sutherland_air
from .loaders import (
    load_blasius,
    load_cavity,
    load_jet,
    load_paraboloid,
    load_streaky_bl,
)
