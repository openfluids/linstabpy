# SPDX-License-Identifier: GPL-3.0-only
"""Flat-plate filter-off guard for near-wall two-delta saw-tooth artifacts."""

from __future__ import annotations

import numpy as np

from linstabpy.baseflow import BaseFlow, sutherland_air
from linstabpy.mesh import Mesh
from linstabpy.operators import build_lnse_operator


def _flat_plate_boundary_layer_problem(nx: int = 32, ny: int = 20):
    mesh = Mesh.create(
        xrange=(0.0, 4.0),
        yrange=(0.0, 1.0),
        nx=nx,
        ny=ny,
        fd_order=4,
        alpha_filter="none",
    )

    y = mesh.Y
    ones = np.ones_like(y)
    zeros = np.zeros_like(y)
    # Smooth synthetic flat-plate profile: no geometric curvature, no filter.
    u_profile = 1.0 - np.exp(-((y / 0.15) ** 2))
    baseflow = BaseFlow(
        Re=1.0e4,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=ones.copy(),
        U=u_profile,
        V=zeros.copy(),
        W=zeros.copy(),
        T=ones.copy(),
    )
    sutherland_air(baseflow)
    return mesh, baseflow


def _row_nyquist_gain(operator, idx, var: str, row: int, nx: int, ny: int) -> float:
    probe = np.zeros(operator.shape[1], dtype=complex)
    mesh_indices = np.array([row + col * ny for col in range(nx)], dtype=np.int64)
    alternating = (-1.0) ** np.arange(nx)
    probe[idx[var][mesh_indices]] = alternating / np.linalg.norm(alternating)
    return float(np.linalg.norm(operator @ probe))


def test_flat_plate_filter_off_has_no_near_wall_two_delta_growth():
    """Front-D: curved-closure saw-tooth uses Path (a) fd4 stability; flat plate is clean with filter off."""
    mesh, baseflow = _flat_plate_boundary_layer_problem()
    assert mesh.alpha_filter == "none"

    operator, idx = build_lnse_operator(mesh, baseflow, kx=0.0)
    ny, nx = mesh.X.shape

    variables = ("rho_j", "u_j", "v_j", "w_j", "T_j")
    ratios = {}
    for var in variables:
        near_wall_gain = max(
            _row_nyquist_gain(operator, idx, var, row, nx, ny) for row in (0, 1, 2)
        )
        interior_gains = np.array(
            [_row_nyquist_gain(operator, idx, var, row, nx, ny) for row in range(4, ny - 4)]
        )
        interior_reference = float(np.median(interior_gains))
        ratios[var] = near_wall_gain / interior_reference

    # A clean rectangular flat-plate operator should not turn a wall-row
    # two-delta probe into an anomalously larger response than the same
    # physical Nyquist probe in the interior.
    assert max(ratios.values()) < 2.25, ratios
