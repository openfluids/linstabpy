# SPDX-License-Identifier: GPL-3.0-only
"""Wavemaker input checks that do not import PETSc."""

from __future__ import annotations

import numpy as np
import pytest

from linstabpy.stability.sensitivity import wavemaker


def test_wavemaker_rejects_empty_modes() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        wavemaker(
            np.zeros((0, 1), dtype=complex),
            np.zeros((0, 1), dtype=complex),
            np.array([1.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
            np.ones(0),
            grid_shape=(1, 1),
        )


def test_wavemaker_rejects_nonfinite_eigs() -> None:
    modes = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="finite"):
        wavemaker(
            modes,
            modes,
            np.array([1.0, 2.0, 3.0, np.nan], dtype=complex),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=complex),
            np.ones(4),
            grid_shape=(2, 2),
        )


def test_wavemaker_rejects_off_diagonal_weight() -> None:
    modes = np.eye(4, dtype=complex)
    weight = np.eye(4)
    weight[0, 1] = 0.1
    with pytest.raises(ValueError, match="diagonal weight"):
        wavemaker(
            modes,
            modes,
            np.array([1.0, 2.0, 3.0, 4.0], dtype=complex),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=complex),
            weight,
            grid_shape=(2, 2),
        )


def test_wavemaker_rejects_grid_that_does_not_divide_ndof() -> None:
    modes = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="integer multiple"):
        wavemaker(
            modes,
            modes,
            np.array([1.0, 2.0, 3.0, 4.0], dtype=complex),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=complex),
            np.ones(4),
            grid_shape=(3, 2),
        )


def test_wavemaker_accepts_a_1d_mode_vector() -> None:
    mode = np.ones(4, dtype=complex)
    result = wavemaker(
        mode,
        mode,
        np.array([1.0 + 0.0j]),
        np.array([1.0 + 0.0j]),
        np.ones(4),
        grid_shape=(2, 2),
    )
    assert result["zeta"].shape == (1, 2, 2)


def test_wavemaker_rejects_bad_weight_and_grid_shapes() -> None:
    modes = np.eye(4, dtype=complex)
    eigs = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
    with pytest.raises(ValueError, match="length must match"):
        wavemaker(modes, modes, eigs[:2], eigs, np.ones(4), (2, 2))
    with pytest.raises(ValueError, match="weight length"):
        wavemaker(modes, modes, eigs, eigs, np.ones(3), (2, 2))
    with pytest.raises(ValueError, match="weight matrix shape"):
        wavemaker(modes, modes, eigs, eigs, np.eye(3), (2, 2))
    with pytest.raises(ValueError, match="1-D diagonal or a 2-D"):
        wavemaker(modes, modes, eigs, eigs, np.ones((2, 2, 2)), (2, 2))
    with pytest.raises(ValueError, match="\\(Ny, Nx\\)"):
        wavemaker(modes, modes, eigs, eigs, np.ones(4), (2,))
    with pytest.raises(ValueError, match="positive"):
        wavemaker(modes, modes, eigs, eigs, np.ones(4), (0, 2))
    with pytest.raises(ValueError, match="1-D mode vector or 2-D"):
        wavemaker(np.ones((2, 2, 2)), np.ones((2, 2, 2)), eigs[:1], eigs[:1], np.ones(8), (2, 2))


def test_wavemaker_rejects_nonpositive_weight_and_zero_norm() -> None:
    modes = np.eye(4, dtype=complex)
    eigs = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
    with pytest.raises(ValueError, match="positive"):
        wavemaker(modes, modes, eigs, eigs, np.array([-1.0, 1.0, 1.0, 1.0]), (2, 2))
    zeros = np.zeros((4, 1), dtype=complex)
    with pytest.raises(ValueError, match="non-positive weighted mode norm"):
        wavemaker(zeros, zeros, np.array([1.0 + 0.0j]), np.array([1.0 + 0.0j]), np.ones(4), (2, 2))


def test_wavemaker_rejects_bad_pairing_tolerance_and_reused_adjoint() -> None:
    modes = np.eye(4, dtype=complex)
    eigs = np.array([1.0, 1.05], dtype=complex)
    adj = np.array([1.0, 10.0], dtype=complex)
    two = modes[:, :2]
    with pytest.raises(ValueError, match="pairing_tol"):
        wavemaker(two, two, eigs, adj, np.ones(4), (2, 2), pairing_tol=0.0)
    with pytest.raises(ValueError, match="paired more than once"):
        wavemaker(two, two, eigs, adj, np.ones(4), (2, 2), pairing_tol=0.1)


def test_wavemaker_rejects_shape_mismatch_and_zero_inner_product() -> None:
    direct = np.eye(4, 1, dtype=complex)
    adjoint = np.eye(4, 2, dtype=complex)
    with pytest.raises(ValueError, match="same shape"):
        wavemaker(direct, adjoint, np.array([1.0 + 0.0j]), np.array([1.0, 2.0]), np.ones(4), (2, 2))
    q_dir = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    q_adj = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    with pytest.raises(ValueError, match="normalization"):
        wavemaker(
            q_dir.reshape(-1, 1),
            q_adj.reshape(-1, 1),
            np.array([1.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
            np.ones(4),
            (2, 2),
        )
