# SPDX-License-Identifier: GPL-3.0-only
"""Structural-sensitivity post-processing utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _as_modes(name: str, modes: np.ndarray) -> np.ndarray:
    arr = np.asarray(modes, dtype=complex)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1-D mode vector or 2-D column matrix")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _as_eigs(name: str, eigs: np.ndarray, n_modes: int) -> np.ndarray:
    arr = np.asarray(eigs, dtype=complex).reshape(-1)
    if arr.size != n_modes:
        raise ValueError(f"{name} length must match the number of mode columns ({n_modes})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_weight(weight: np.ndarray, ndof: int) -> np.ndarray:
    arr = np.asarray(weight, dtype=complex)
    if arr.ndim == 1:
        if arr.size != ndof:
            raise ValueError(f"weight length must match mode length ({ndof})")
        return arr
    if arr.ndim == 2:
        if arr.shape != (ndof, ndof):
            raise ValueError(f"weight matrix shape must be ({ndof}, {ndof})")
        if np.count_nonzero(arr - np.diag(np.diag(arr))) != 0:
            raise ValueError("wavemaker currently expects a diagonal weight")
        return np.diag(arr)
    raise ValueError("weight must be a 1-D diagonal or a 2-D diagonal matrix")


def _validate_grid_shape(grid_shape: tuple[int, int], ndof: int) -> tuple[int, int, int]:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must be a (Ny, Nx) tuple")
    ny, nx = (int(grid_shape[0]), int(grid_shape[1]))
    if ny <= 0 or nx <= 0:
        raise ValueError("grid_shape entries must be positive")
    n_grid = ny * nx
    if ndof % n_grid != 0:
        raise ValueError("mode length must be an integer multiple of Ny * Nx")
    return ny, nx, ndof // n_grid


def _weighted_norm(vector: np.ndarray, weight_diag: np.ndarray) -> float:
    norm_sq = np.vdot(vector, weight_diag * vector)
    norm = float(np.sqrt(np.real_if_close(norm_sq).real))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("encountered a non-positive weighted mode norm")
    return norm


def _pair_modes(
    direct_eigs: np.ndarray,
    adjoint_eigs: np.ndarray,
    pairing_tol: float,
) -> list[dict[str, Any]]:
    if pairing_tol <= 0.0 or not np.isfinite(pairing_tol):
        raise ValueError("pairing_tol must be a positive finite value")

    pairs: list[dict[str, Any]] = []
    used_adjoint: set[int] = set()
    distances = np.abs(direct_eigs[:, None] - np.conj(adjoint_eigs[None, :]))
    for direct_index, direct_value in enumerate(direct_eigs):
        row = distances[direct_index]
        candidates = np.flatnonzero(row <= pairing_tol)
        if candidates.size != 1:
            nearest = int(np.argmin(row))
            raise ValueError(
                "ambiguous or mismatched adjoint/direct eigenvalue pairing: "
                f"direct_index={direct_index}, lambda_dir={direct_value!r}, "
                f"candidates_within_tol={candidates.tolist()}, nearest_adjoint_index={nearest}, "
                f"lambda_adj_nearest={adjoint_eigs[nearest]!r}, distance={row[nearest]:.3e}, "
                f"pairing_tol={pairing_tol:.3e}, direct_eigs={direct_eigs!r}, "
                f"adjoint_eigs={adjoint_eigs!r}"
            )
        adjoint_index = int(candidates[0])
        if adjoint_index in used_adjoint:
            raise ValueError(
                "ambiguous or mismatched adjoint/direct eigenvalue pairing: "
                f"adjoint_index={adjoint_index} is paired more than once, "
                f"direct_eigs={direct_eigs!r}, adjoint_eigs={adjoint_eigs!r}"
            )
        used_adjoint.add(adjoint_index)
        pairs.append(
            {
                "direct_index": direct_index,
                "adjoint_index": adjoint_index,
                "lambda_dir": direct_value,
                "lambda_adj": adjoint_eigs[adjoint_index],
                "distance": float(row[adjoint_index]),
            }
        )
    return pairs


def _pointwise_magnitude(
    vector: np.ndarray, grid_shape: tuple[int, int], n_components: int
) -> np.ndarray:
    ny, nx = grid_shape
    n_grid = ny * nx
    magnitude_sq = np.zeros((ny, nx), dtype=float)
    for component in range(n_components):
        block = vector[component * n_grid : (component + 1) * n_grid]
        field = block.reshape((ny, nx), order="F")
        magnitude_sq += np.abs(field) ** 2
    return np.sqrt(magnitude_sq)


def wavemaker(
    direct_modes: np.ndarray,
    adjoint_modes: np.ndarray,
    direct_eigs: np.ndarray,
    adjoint_eigs: np.ndarray,
    weight: np.ndarray,
    grid_shape: tuple[int, int],
    pairing_tol: float = 1e-8,
) -> dict[str, Any]:
    """Compute Giannetti--Luchini wavemaker fields from direct/adjoint modes.

    The structural sensitivity is evaluated as
    ``zeta(x, y) = |q_dir(x, y)| |q_adj(x, y)| / |<q_adj, q_dir>_W|``.
    Both input mode columns are first normalized to unit weighted norm, and the
    biorthogonal inner-product divisor ``<q_adj, W q_dir>`` is retained
    explicitly in the returned ``normalization_constants`` so figures can be
    reproduced without relying on hidden scaling conventions.

    Adjoint convention: :func:`linstabpy.stability._eigenvalues.adjoint_eigenvalues`
    returns eigenvectors of the Chu-weighted adjoint operator
    ``W^{-1} L0^H W`` and W-normalizes those vectors.  Therefore wavemaker
    expects physical weighted-adjoint modes already; it does **not** apply an
    additional ``W^{-1}`` transform.  Supplying Euclidean ``L0^H`` left modes
    requires converting them before calling this routine.

    Direct and adjoint modes are paired by the minimum distance
    ``|lambda_dir - conj(lambda_adj)|`` with a tolerance gate.  Pairing failures
    are loud: ambiguous or missing matches raise :class:`ValueError` with the
    offending eigenvalues.  Each accepted pair is logged at INFO level as a
    table row containing ``lambda_dir``, ``lambda_adj``, pairing distance, and
    normalization constant.

    The spatial reshape is Fortran-order by construction.  For each state
    component block, this routine uses ``vec.reshape((Ny, Nx), order="F")``;
    C-order reshaping would scramble wall-normal/grid-line locations.
    """
    direct = _as_modes("direct_modes", direct_modes)
    adjoint = _as_modes("adjoint_modes", adjoint_modes)
    if direct.shape != adjoint.shape:
        raise ValueError("direct_modes and adjoint_modes must have the same shape")

    ndof, n_modes = direct.shape
    direct_values = _as_eigs("direct_eigs", direct_eigs, n_modes)
    adjoint_values = _as_eigs("adjoint_eigs", adjoint_eigs, n_modes)
    weight_diag = _as_weight(weight, ndof)
    if np.any(np.real(weight_diag) <= 0.0):
        raise ValueError("weight entries must be positive")
    ny, nx, n_components = _validate_grid_shape(grid_shape, ndof)

    pairs = _pair_modes(direct_values, adjoint_values, pairing_tol)
    zeta = np.empty((n_modes, ny, nx), dtype=float)
    normalizations = np.empty(n_modes, dtype=complex)

    for out_index, pair in enumerate(pairs):
        direct_index = pair["direct_index"]
        adjoint_index = pair["adjoint_index"]
        q_dir = direct[:, direct_index] / _weighted_norm(direct[:, direct_index], weight_diag)
        q_adj = adjoint[:, adjoint_index] / _weighted_norm(adjoint[:, adjoint_index], weight_diag)
        normalization = np.vdot(q_adj, weight_diag * q_dir)
        if not np.isfinite(normalization) or np.abs(normalization) == 0.0:
            raise ValueError(
                "encountered a zero/non-finite direct-adjoint normalization for "
                f"pair {direct_index}->{adjoint_index}"
            )
        normalizations[out_index] = normalization
        zeta[out_index] = (
            _pointwise_magnitude(q_dir, (ny, nx), n_components)
            * _pointwise_magnitude(q_adj, (ny, nx), n_components)
            / np.abs(normalization)
        )
        pair["normalization"] = normalization
        logger.info(
            "wavemaker pairing mode=%d direct_index=%d adjoint_index=%d "
            "lambda_dir=%s lambda_adj=%s distance=%.6e normalization=%s",
            out_index,
            direct_index,
            adjoint_index,
            pair["lambda_dir"],
            pair["lambda_adj"],
            pair["distance"],
            normalization,
        )

    return {
        "zeta": zeta,
        "pairing": pairs,
        "normalization_constants": normalizations,
        "grid_shape": (ny, nx),
        "n_components": n_components,
    }
