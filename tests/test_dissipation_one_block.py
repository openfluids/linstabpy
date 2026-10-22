# SPDX-License-Identifier: GPL-3.0-only
"""Every path reads the dissipation strength from numerics.operator_dissipation."""

from __future__ import annotations

import math
import re
from importlib import import_module
from typing import Literal

import numpy as np
import pytest

from linstabpy.config.defaults import stage_dissipation_epsilon
from linstabpy.config.schema import ConfigValidationError, validate_config

# Import the module itself: linstabpy.solvers.router exports a function of the same name.
rm = import_module("linstabpy.solvers.router.resolvent_map")
BLOCK_MESSAGE = re.escape("numerics.operator_dissipation")


@pytest.mark.parametrize(
    ("block", "stage", "expected"),
    [
        (None, "resolvent", 0.0),
        ({"enabled": False, "epsilon": 0.1}, "resolvent", 0.0),
        ({"enabled": True, "epsilon": 0.1}, "resolvent", 0.1),
        ({"enabled": True, "epsilon": 0.1}, "stability", 0.1),
        ({"enabled": True, "epsilon": 0.1, "apply_to_resolvent": False}, "resolvent", 0.0),
        ({"enabled": True, "epsilon": 0.1, "apply_to_resolvent": False}, "stability", 0.1),
    ],
)
def test_stage_dissipation_epsilon(
    block: dict[str, bool | float] | None,
    stage: Literal["stability", "resolvent"],
    expected: float,
) -> None:
    assert math.isclose(stage_dissipation_epsilon(block, stage), expected, abs_tol=0.0)


def test_router_reads_strength_from_the_block() -> None:
    cfg = {"numerics": {"operator_dissipation": {"enabled": True, "epsilon": 0.07}}}
    assert math.isclose(rm._assembly_kwargs(cfg)["dissipation_epsilon"], 0.07, abs_tol=0.0)
    assert math.isclose(rm._assembly_kwargs({})["dissipation_epsilon"], 0.0, abs_tol=0.0)


@pytest.mark.parametrize("key", ["dissipation_epsilon", "eps_resolv"])
def test_router_refuses_the_old_top_level_key(key: str) -> None:
    with pytest.raises(ValueError, match=BLOCK_MESSAGE):
        rm._assembly_kwargs({key: 0.1})


@pytest.mark.parametrize("key", ["dissipation_epsilon", "eps_resolv"])
def test_validate_config_names_the_block_for_the_old_key(key: str) -> None:
    with pytest.raises(ConfigValidationError, match=BLOCK_MESSAGE):
        validate_config({key: 0.1})


def test_router_refuses_strength_on_a_mesh_without_q_diss() -> None:
    """The assembly refuses the strength, so the map cannot run undamped."""
    from linstabpy.baseflow import BaseFlow, sutherland_air
    from linstabpy.mesh import Mesh

    cfg = {"numerics": {"operator_dissipation": {"enabled": True, "epsilon": 0.1}}}
    mesh = Mesh.create(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=8, ny=8, fd_order=4)
    ones = np.ones(mesh.X.shape)
    base_flow = BaseFlow(
        Re=1000.0,
        Ma=0.5,
        Pr=0.7,
        kappa=1.4,
        T_0=300.0,
        RHO=ones,
        U=ones,
        V=0 * ones,
        W=0 * ones,
        T=ones,
    )
    sutherland_air(base_flow)
    with pytest.raises(ValueError, match="carries no Q_diss"):
        rm._assemble_beta_operator(mesh, base_flow, 1.0, cfg)
