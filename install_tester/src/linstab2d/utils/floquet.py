"""Floquet reconstruction helpers."""

from __future__ import annotations

import numpy as np


def reconstruct_floquet(
    X: np.ndarray,
    Y: np.ndarray,
    V: np.ndarray,
    idx: dict[str, np.ndarray],
    used_ind: np.ndarray,
    floquet_exp: float,
) -> np.ndarray:
    """Reconstruct a Floquet mode in physical space.

    Parameters
    ----------
    X, Y : ndarray
        Coordinates of the computational domain.
    V : ndarray
        Mode values on active grid points. Accepts:

        - 1-D array of length ``len(used_ind)``.
        - 2-D array where one axis has length ``len(used_ind)``; the
          other axis is treated as an ensemble axis.
    idx : dict
        Boundary index map from the mesh (used for compatibility; currently
        accepted but not required for reconstruction).
    used_ind : ndarray
        Active-point indices into ``np.ravel(X, order='F')``.
    floquet_exp : float
        Floquet exponent ``β`` used in the phase ``exp(i β x)``.

    Returns
    -------
    ndarray
        Reconstructed array shaped like ``X`` (for 1-D input) or
        ``(num_fields, ny, nx)`` for ensemble input.
    """
    _ = Y  # retained for signature parity with MATLAB reconstruction helpers
    _ = idx

    x_phase = np.exp(1j * floquet_exp * np.asarray(X, dtype=float))
    used_ind = np.asarray(used_ind, dtype=int)
    flat_active = np.asarray(V, dtype=complex)
    ngp = used_ind.size

    if flat_active.ndim == 1:
        if flat_active.size != ngp:
            raise ValueError("For 1-D V, size must match len(used_ind)")
        full_flat = np.zeros(X.size, dtype=complex)
        full_flat[used_ind] = flat_active
        return x_phase * full_flat.reshape(X.shape, order="F")

    if flat_active.ndim != 2:
        raise ValueError("V must be 1-D or 2-D")

    if flat_active.shape[0] == ngp:
        n_fields = flat_active.shape[1]
        full_flat = np.zeros((n_fields, X.size), dtype=complex)
        full_flat[:, used_ind] = flat_active.T
        out = np.empty((n_fields,) + X.shape, dtype=complex)
        for i in range(n_fields):
            out[i, ...] = x_phase * full_flat[i, :].reshape(X.shape, order="F")
        return out

    if flat_active.shape[1] == ngp:
        n_fields = flat_active.shape[0]
        full_flat = np.zeros((n_fields, X.size), dtype=complex)
        full_flat[:, used_ind] = flat_active
        out = np.empty((n_fields,) + X.shape, dtype=complex)
        for i in range(n_fields):
            out[i, ...] = x_phase * full_flat[i, :].reshape(X.shape, order="F")
        return out

    raise ValueError("For 2-D V, one dimension must have length len(used_ind)")
