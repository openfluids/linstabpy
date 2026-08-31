# SPDX-License-Identifier: GPL-3.0-only
"""Loader for the prescribed-forcing npz contract.

One npz file holds one complex forcing vector at one ``(beta, omega)``
point, produced entirely outside linstabpy (see
``docs/plan_1.1_prescribed_forcing.md``). Required keys: ``f``, ``beta``,
``omega``. Optional keys: ``gamma``, ``mesh``, ``nx``, ``ny``, ``commit``.

Loading here checks the file's shape, not its physics. It never guesses the
unknown order. It never interpolates data onto a mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REQUIRED_KEYS = ("f", "beta", "omega")
_OPTIONAL_EXTRA_KEYS = ("mesh", "nx", "ny", "commit")


@dataclass(frozen=True)
class PrescribedForcing:
    """One loaded prescribed-forcing point.

    Attributes
    ----------
    f : ndarray
        Complex forcing vector, length ``expected_ndof``.
    beta : float
        Spanwise wavenumber, matching ``build_lnse_system``.
    omega : float
        Frequency, in the case's own nondimensional units.
    gamma : float or None
        Discount from the npz file, if present.
    extras : dict
        Whichever of ``mesh``, ``nx``, ``ny``, ``commit`` the file carried.
    """

    f: np.ndarray
    beta: float
    omega: float
    gamma: float | None
    extras: dict[str, object] = field(default_factory=dict)


def _npz_scalar(value: np.ndarray) -> object:
    """Unwrap a NumPy scalar or 0-d array to a plain Python object.

    Returns
    -------
    object
        ``value.item()`` when possible, otherwise ``value`` unchanged.
    """
    if hasattr(value, "item"):
        return value.item()
    return value


def load_prescribed_forcing(
    path: str | Path,
    *,
    expected_ndof: int,
    config_gamma: float | None = None,
    allow_zero_bin: bool = False,
) -> PrescribedForcing:
    """Load and validate one prescribed-forcing npz file.

    Parameters
    ----------
    path : str or Path
        Location of the npz file.
    expected_ndof : int
        Required length of the ``f`` vector, from the target operator.
    config_gamma : float, optional
        Discount from the caller's config. Compared against an ``f``
        npz ``gamma`` when both are present.
    allow_zero_bin : bool, optional
        Set True to accept ``(beta, omega) == (0.0, 0.0)``. Default False
        rejects that bin, since it is mean-flow distortion, not a
        forcing point.

    Returns
    -------
    PrescribedForcing
        The validated forcing point.
    """
    with np.load(path, allow_pickle=False) as npz:
        _check_required_keys(npz, path)
        f = np.asarray(npz["f"]).astype(complex, copy=False)
        beta = float(npz["beta"])
        omega = float(npz["omega"])
        npz_gamma = float(npz["gamma"]) if "gamma" in npz else None
        extras = {key: _npz_scalar(npz[key]) for key in _OPTIONAL_EXTRA_KEYS if key in npz}

    _check_f_shape(f, expected_ndof, path)
    _check_zero_bin(beta, omega, path, allow_zero_bin=allow_zero_bin)
    gamma = _reconcile_gamma(npz_gamma, config_gamma, path)
    return PrescribedForcing(f=f, beta=beta, omega=omega, gamma=gamma, extras=extras)


def _check_required_keys(npz: np.lib.npyio.NpzFile, path: str | Path) -> None:
    """Raise loudly when a required npz key is absent.

    Raises
    ------
    ValueError
        If any of ``f``, ``beta``, ``omega`` is missing.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in npz]
    if missing:
        message = f"{path}: missing required key(s) {missing}, need {list(_REQUIRED_KEYS)}"
        raise ValueError(message)


def _check_f_shape(f: np.ndarray, expected_ndof: int, path: str | Path) -> None:
    """Raise loudly when ``f``'s shape, length, or values fail validation.

    Raises
    ------
    ValueError
        If ``f`` is not 1-D. Also if its length differs from
        ``expected_ndof``, or if it holds a NaN or Inf entry.
    """
    if f.ndim != 1:
        message = f"{path}: f must be a 1-D vector, got shape {f.shape}"
        raise ValueError(message)
    if f.shape[0] != expected_ndof:
        message = f"{path}: f length {f.shape[0]} does not match expected ndof {expected_ndof}"
        raise ValueError(message)
    if not np.all(np.isfinite(f)):
        message = f"{path}: f contains NaN or Inf"
        raise ValueError(message)


def _check_zero_bin(beta: float, omega: float, path: str | Path, *, allow_zero_bin: bool) -> None:
    """Raise loudly on the rejected ``(0, 0)`` bin.

    The Fourier bin at (beta, omega) = (0, 0) is the mean-flow component of
    the disturbance field. A non-zero mean-flow distortion would be a physical
    statement about the base-flow correction; a zero-bin forcing is a different
    question from the prescribed-forcing resolvent problem. Rejecting (0, 0)
    by default prevents a caller from silently asking the wrong question.

    Raises
    ------
    ValueError
        If ``(beta, omega) == (0.0, 0.0)`` and ``allow_zero_bin`` is False.
    """
    if not allow_zero_bin and (beta, omega) == (0.0, 0.0):
        message = f"{path}: (beta, omega) = (0, 0) is the mean-flow bin; set allow_zero_bin=True to accept it"
        raise ValueError(message)


def _reconcile_gamma(
    npz_gamma: float | None,
    config_gamma: float | None,
    path: str | Path,
) -> float | None:
    """Reconcile the npz and config ``gamma`` values.

    A mismatch between the two discount values is an error: the npz file
    encodes the physics of the external forcing (a deliberate choice at
    generation time), while config_gamma comes from the user's solver
    setup. A silent disagreement would poison every row of a gain map with
    the wrong discount—the resolvent response would be solved for one
    gamma but reported under another. The comparison is exact bitwise
    equality, not approximate: the only valid outcome is that both carry
    the same value or only one is present.

    Returns
    -------
    float or None
        ``npz_gamma`` when present, else ``config_gamma``.

    Raises
    ------
    ValueError
        If both are present and unequal.
    """
    if npz_gamma is not None and config_gamma is not None and npz_gamma != config_gamma:
        message = f"{path}: npz gamma={npz_gamma!r} does not match config gamma={config_gamma!r}"
        raise ValueError(message)
    return npz_gamma if npz_gamma is not None else config_gamma
