# SPDX-License-Identifier: GPL-3.0-only
"""Tests for finite-difference weight helpers."""

from __future__ import annotations

import numpy as np

from linstabpy.mesh import fdweights


def test_fdweights_known_centered_stencils() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    w1 = fdweights(0.0, x, 1)
    w2 = fdweights(0.0, x, 2)
    assert np.allclose(w1, np.array([-0.5, 0.0, 0.5]))
    assert np.allclose(w2, np.array([1.0, -2.0, 1.0]))


def test_fdweights_interpolation_reproduces_polynomial() -> None:
    x = np.array([0.0, 0.5, 1.5, 2.0])
    xi = 0.8
    w = fdweights(xi, x, 0)
    values = x**3
    assert np.isclose(w @ values, xi**3)


def test_fdweights_unequal_spacing_matches_analytic_derivative() -> None:
    x = np.array([0.0, 0.2, 0.7, 1.4])
    xi = 0.5
    w = fdweights(xi, x, 1)
    values = x**2
    assert np.isclose(w @ values, 2.0 * xi, rtol=1e-10)


def test_fdweights_single_node_edge_case() -> None:
    x = np.array([1.2])
    assert np.allclose(fdweights(1.2, x, 0), np.array([1.0]))
    assert np.allclose(fdweights(1.2, x, 1), np.array([0.0]))
