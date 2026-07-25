# SPDX-License-Identifier: GPL-3.0-only
"""Configuration validation helpers."""

from linstabpy.config.defaults import DISSIPATION_APPLY_DEFAULTS
from linstabpy.config.schema import ConfigValidationError, validate_config
from linstabpy.config.typed import (
    Config,
    EigenCfg,
    LinSolveCfg,
    MeshCfg,
    MpiCfg,
    ResolventCfg,
    ResolventEpsCfg,
    SpatialCfg,
)

__all__ = [
    "DISSIPATION_APPLY_DEFAULTS",
    "Config",
    "ConfigValidationError",
    "EigenCfg",
    "LinSolveCfg",
    "MeshCfg",
    "MpiCfg",
    "ResolventCfg",
    "ResolventEpsCfg",
    "SpatialCfg",
    "validate_config",
]
