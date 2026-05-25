# SPDX-License-Identifier: GPL-3.0-only
"""Tests for mesh utility ports."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, identity

from linstab2d.mesh import fdweights, interp_mat_2d, min_distance, segmspace


def test_fdweights_known_centered_stencils() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    w1 = fdweights(0.0, x, 1)
    w2 = fdweights(0.0, x, 2)
    assert np.allclose(w1, np.array([-0.5, 0.0, 0.5]))
    assert np.allclose(w2, np.array([1.0, -2.0, 1.0]))


def test_fdweights_interpolation_reproduces_polynomial() -> None:
    x = np.array([0.0, 0.25, 1.0, 1.5, 2.5])
    xi = 0.5
    w = fdweights(xi, x, 0)
    values = x**4
    assert np.isclose(np.dot(w, values), xi**4)


def test_fdweights_unequal_spacing_matches_analytic_derivative() -> None:
    x = np.array([0.0, 0.25, 1.0, 1.5, 2.5])
    xi = 0.5
    w = fdweights(xi, x, 1)
    approx = np.dot(w, np.exp(x))
    assert np.isclose(approx, np.exp(xi), rtol=5e-2, atol=5e-2)


def test_fdweights_single_node_edge_case() -> None:
    x = np.array([1.2])
    assert np.allclose(fdweights(1.2, x, 0), np.array([1.0]))
    assert np.allclose(fdweights(1.2, x, 1), np.array([0.0]))


def test_segmspace_single_segment_equals_linspace() -> None:
    x = segmspace([0.0, 1.0], [5], 0)
    assert np.allclose(x, np.linspace(0.0, 1.0, 5))


def test_segmspace_two_segments_without_smoothing() -> None:
    x = segmspace([0.0, 1.0, 2.0], [4, 4], 0)
    expected = np.array(
        [
            0.0,
            0.2916666666666667,
            0.5833333333333334,
            0.875,
            1.125,
            1.4166666666666667,
            1.7083333333333335,
            2.0,
        ]
    )
    assert np.allclose(x, expected)


def test_segmspace_smoothing_preserves_endpoints() -> None:
    raw = segmspace([0.0, 1.0, 2.0], [8, 8], 0)
    smoothed = segmspace([0.0, 1.0, 2.0], [8, 8], 4)
    assert np.isclose(smoothed[0], raw[0])
    assert np.isclose(smoothed[-1], raw[-1])


def test_segmspace_smoothing_keeps_monotonicity() -> None:
    smoothed = segmspace([0.0, 1.0, 2.0, 4.0], [6, 6, 6], 5)
    assert np.all(np.diff(smoothed) > 0.0)


def test_min_distance_single_reference_point() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 3.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    d = min_distance(X, Y, [0])
    assert np.allclose(d, np.sqrt(X**2 + Y**2))


def test_min_distance_all_points_are_references() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    d = min_distance(X, Y, np.arange(X.size))
    assert np.allclose(d, np.zeros_like(X))


def test_min_distance_symmetry_for_symmetric_layout() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-1.0, 0.0, 1.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    d = min_distance(X, Y, [0, 2, 6, 8])
    assert np.allclose(d, d[::-1, :])
    assert np.allclose(d, d[:, ::-1])


def test_interp_mat_2d_identity_grid_gives_identity_matrix() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-1.0, 0.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    W = interp_mat_2d(x, y, X, Y, method="linear")
    I = identity(x.size * y.size, format="csc", dtype=np.complex128)
    assert isinstance(W, csc_matrix)
    assert W.dtype == np.complex128
    assert np.allclose((W - I).toarray(), 0.0)


def test_interp_mat_2d_preserves_constant_field() -> None:
    x = np.linspace(0.0, 2.0, 5)
    y = np.linspace(-1.0, 1.0, 4)
    Xd, Yd = np.meshgrid(np.linspace(0.2, 1.8, 6), np.linspace(-0.8, 0.8, 7), indexing="ij")
    W = interp_mat_2d(x, y, Xd, Yd, method="linear")
    const = np.full(x.size * y.size, 3.25)
    out = W @ const
    assert np.allclose(out, 3.25)


def test_interp_mat_2d_matches_sine_cosine_field() -> None:
    x = np.linspace(0.0, 1.0, 7)
    y = np.linspace(-0.5, 0.5, 6)
    Xs, Ys = np.meshgrid(x, y, indexing="ij")
    f_src = np.sin(Xs) * np.cos(Ys)
    Xd, Yd = np.meshgrid(np.linspace(0.05, 0.95, 9), np.linspace(-0.45, 0.45, 8), indexing="ij")
    W = interp_mat_2d(x, y, Xd, Yd, method="linear")
    interp_values = (W @ f_src.ravel(order="F")).reshape(Xd.shape, order="F").real
    truth = np.sin(Xd) * np.cos(Yd)
    assert np.max(np.abs(interp_values - truth)) < 5e-2


def test_interp_mat_2d_linear_has_at_most_four_nonzeros_per_row() -> None:
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 1.0, 5)
    Xd, Yd = np.meshgrid(np.linspace(0.1, 0.9, 8), np.linspace(0.1, 0.9, 7), indexing="ij")
    W = interp_mat_2d(x, y, Xd, Yd, method="linear")
    row_nnz = np.diff(W.tocsr().indptr)
    assert np.max(row_nnz) <= 4
