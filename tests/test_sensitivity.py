# SPDX-License-Identifier: GPL-3.0-only
"""Tests for structural-sensitivity/wavemaker post-processing."""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
import scipy.linalg

from linstabpy import wavemaker
from linstabpy.stability._eigenvalues import adjoint_eigenvalues


def _dense_oracle_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a small non-normal W-adjoint eigenproblem with separated roots."""
    rng = np.random.default_rng(20260612)
    ndof = 8
    transform = rng.normal(size=(ndof, ndof)) + 1j * rng.normal(size=(ndof, ndof))
    while np.linalg.cond(transform) > 25.0:
        transform = rng.normal(size=(ndof, ndof)) + 1j * rng.normal(size=(ndof, ndof))
    spectrum = np.array(
        [
            -0.40 + 0.15j,
            -0.15 - 0.35j,
            0.20 + 0.55j,
            0.75 - 0.10j,
            1.20 + 0.30j,
            1.80 - 0.65j,
            2.40 + 0.05j,
            3.10 - 0.40j,
        ],
        dtype=complex,
    )
    operator = transform @ np.diag(spectrum) @ np.linalg.inv(transform)
    weight = np.linspace(0.7, 2.3, ndof)
    adjoint_operator = (operator.conj().T * weight) / weight[:, None]
    direct_eigs, direct_modes = scipy.linalg.eig(operator)
    adjoint_eigs, adjoint_modes = scipy.linalg.eig(adjoint_operator)
    order = np.argsort(direct_eigs.real)
    direct_eigs = direct_eigs[order]
    direct_modes = direct_modes[:, order]
    adj_order = np.array(
        [np.argmin(np.abs(direct_eigs[i] - np.conj(adjoint_eigs))) for i in range(ndof)]
    )
    return direct_eigs, direct_modes, adjoint_eigs[adj_order], adjoint_modes[:, adj_order], weight


def _w_normalize(modes: np.ndarray, weight: np.ndarray) -> np.ndarray:
    out = np.array(modes, dtype=complex, copy=True)
    for i in range(out.shape[1]):
        out[:, i] /= np.sqrt(np.real(np.vdot(out[:, i], weight * out[:, i])))
    return out


def test_adjoint_eigenvalues_return_weighted_adjoint_modes_not_euclidean_left_modes():
    """Pin the source convention consumed by wavemaker: W^-1 L0^H W modes."""
    operator = np.array([[1.0 + 0.2j, 2.5 - 0.4j], [0.0, 3.0 - 0.7j]], dtype=complex)
    weight = np.array([1.0, 2.0])
    inv_weight = 1.0 / weight
    adjoint_eigs, adjoint_modes = adjoint_eigenvalues(
        operator,
        n_modes=operator.shape[0],
        target=np.conj(1.0 + 0.2j),
        W_diag=weight,
        invW_diag=inv_weight,
    )
    weighted_adjoint_operator = (operator.conj().T * weight) / weight[:, None]
    euclidean_adjoint_operator = operator.conj().T
    weighted_errors = []
    euclidean_errors = []
    for value, mode in zip(adjoint_eigs, adjoint_modes.T, strict=True):
        weighted_residual = weighted_adjoint_operator @ mode - value * mode
        euclidean_residual = euclidean_adjoint_operator @ mode - value * mode
        weighted_errors.append(np.linalg.norm(weighted_residual) / np.linalg.norm(mode))
        euclidean_errors.append(np.linalg.norm(euclidean_residual) / np.linalg.norm(mode))
    assert max(weighted_errors) < 1e-12
    # Non-constant W means at least one W-adjoint eigenvector is not a
    # Euclidean L0^H eigenvector; wavemaker must not apply W^-1 again.
    assert max(euclidean_errors) > 1e-3


def test_biorthogonality_oracle_for_well_separated_eigenvalues():
    """Scaled direct/W-adjoint eigenvectors satisfy <q_adj_i, W q_dir_j> = I."""
    _, direct_modes, _, adjoint_modes, weight = _dense_oracle_problem()
    direct_unit = _w_normalize(direct_modes, weight)
    adjoint_unit = _w_normalize(adjoint_modes, weight)
    for i in range(adjoint_unit.shape[1]):
        inner = np.vdot(adjoint_unit[:, i], weight * direct_unit[:, i])
        adjoint_unit[:, i] /= np.conj(inner)
    biorthogonality = adjoint_unit.conj().T @ (weight[:, None] * direct_unit)
    np.testing.assert_allclose(biorthogonality, np.eye(biorthogonality.shape[0]), atol=1e-8)


def test_wavemaker_matches_independent_dense_oracle():
    """Package wavemaker agrees with independent scipy.linalg.eig post-processing."""
    direct_eigs, direct_modes, adjoint_eigs, adjoint_modes, weight = _dense_oracle_problem()
    result = wavemaker(
        direct_modes,
        adjoint_modes,
        direct_eigs,
        adjoint_eigs,
        weight,
        grid_shape=(2, 2),
        pairing_tol=1e-8,
    )

    expected = []
    n_grid = 4
    for i, direct_value in enumerate(direct_eigs):
        distances = np.abs(direct_value - np.conj(adjoint_eigs))
        j = int(np.argmin(distances))
        assert distances[j] < 1e-8
        qd = direct_modes[:, i] / np.sqrt(
            np.real(np.vdot(direct_modes[:, i], weight * direct_modes[:, i]))
        )
        qa = adjoint_modes[:, j] / np.sqrt(
            np.real(np.vdot(adjoint_modes[:, j], weight * adjoint_modes[:, j]))
        )
        divisor = abs(np.vdot(qa, weight * qd))
        mag_d = np.zeros((2, 2), dtype=float)
        mag_a = np.zeros((2, 2), dtype=float)
        for component in range(2):
            direct_block = qd[component * n_grid : (component + 1) * n_grid].reshape(
                (2, 2), order="F"
            )
            adjoint_block = qa[component * n_grid : (component + 1) * n_grid].reshape(
                (2, 2), order="F"
            )
            mag_d += np.abs(direct_block) ** 2
            mag_a += np.abs(adjoint_block) ** 2
        expected.append(np.sqrt(mag_d) * np.sqrt(mag_a) / divisor)
    expected_zeta = np.asarray(expected)
    np.testing.assert_allclose(result["zeta"], expected_zeta, rtol=1e-10, atol=1e-12)


def test_pairing_loud_fail_lists_offending_eigenvalues():
    direct_modes = np.eye(4, dtype=complex)
    adjoint_modes = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="ambiguous or mismatched.*direct_eigs=.*adjoint_eigs="):
        wavemaker(
            direct_modes,
            adjoint_modes,
            direct_eigs=np.array([1.0, 2.0, 3.0, 4.0]),
            adjoint_eigs=np.array([10.0, 20.0, 30.0, 40.0]),
            weight=np.ones(4),
            grid_shape=(2, 2),
            pairing_tol=1e-8,
        )


def test_logging_contains_per_mode_pairing_table(caplog: pytest.LogCaptureFixture):
    direct_eigs, direct_modes, adjoint_eigs, adjoint_modes, weight = _dense_oracle_problem()
    caplog.set_level(logging.INFO, logger="linstabpy.stability.sensitivity")
    wavemaker(
        direct_modes[:, :2], adjoint_modes[:, :2], direct_eigs[:2], adjoint_eigs[:2], weight, (2, 2)
    )
    text = caplog.text
    assert "lambda_dir=" in text
    assert "lambda_adj=" in text
    assert "distance=" in text
    assert "normalization=" in text


def test_fortran_order_reshape_guard():
    """A non-square grid catches accidental C-order reshaping."""
    direct = np.arange(1, 7, dtype=float).astype(complex).reshape(6, 1)
    adjoint = np.ones((6, 1), dtype=complex)
    result = wavemaker(
        direct,
        adjoint,
        direct_eigs=np.array([2.0 + 0.0j]),
        adjoint_eigs=np.array([2.0 + 0.0j]),
        weight=np.ones(6),
        grid_shape=(3, 2),
    )
    qd = direct[:, 0] / np.linalg.norm(direct[:, 0])
    qa = adjoint[:, 0] / np.linalg.norm(adjoint[:, 0])
    divisor = abs(np.vdot(qa, qd))
    expected_fortran = (
        np.abs(qd.reshape((3, 2), order="F")) * np.abs(qa.reshape((3, 2), order="F")) / divisor
    )
    expected_c = (
        np.abs(qd.reshape((3, 2), order="C")) * np.abs(qa.reshape((3, 2), order="C")) / divisor
    )
    np.testing.assert_allclose(result["zeta"][0], expected_fortran)
    assert not np.allclose(result["zeta"][0], expected_c)
