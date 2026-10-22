# SPDX-License-Identifier: GPL-3.0-only
"""EVP-distance marking for resolvent forcing-frequency bands.

The marker accepts temporal EVP values in the ``omega`` plane used by
:func:`linstabpy.stability.eigenvalues`.
The resolvent solve forms ``L_sigma = (gamma - i*omega)*I - L0``.
Singular shifts occur when ``-i*omega = lambda(L0)`` (``gamma = 0``).
The resolvent pole is ``omega_pole = i*lambda(L0)``.
The EVP returns ``omega_eig = eig(i*L0) = i*lambda(L0)``.
The pole associated with a temporal EVP value is therefore
``omega_pole = +omega_eig``.
Compare a requested real forcing frequency ``omega_req`` with
``min(abs(omega_req - omega_eig))``.
With ``solver.resolvent_sign_convention: matlab_parity`` the solve forms
``L0 - (gamma + i*omega)*I`` and the pole is ``-omega_eig``.
See ``docs/resolvent_sign_convention.md``.
:func:`label_band` performs no shifted linear solve or resolvent solve.

Configuration.
The danger radius is read, in order, from one of these equivalent keys.

* ``cfg["danger_radius"]``.
* ``cfg["band_marking"]["danger_radius"]``.
* ``cfg["solver"]["router"]["band_marking"]["danger_radius"]``.

When omitted, the default is a physical sweep-spacing radius,
``0.5 * median(diff(sorted(unique(finite(omegas)))))``.
This marks a requested frequency as dangerous when a pole lies within
about half of a local sweep cell.
It uses the user's requested omega-plane spacing rather than an
unrelated linear-solve residual scale.
The least-damped ``abs(Im omega_eig)`` scale is intentionally unused.
It collapses toward zero near a stability threshold, exactly when a
near-real pole most needs a no-go band.
If fewer than two distinct finite requested frequencies exist, no
sweep scale exists.
The default radius is ``0.0`` (no ``"near-pole"`` labels).
NaN and infinite frequencies drop out before the spacing computation.
Non-uniform grids use the median successive spacing so one large gap
does not set the radius.
Any explicit ``danger_radius`` key above overrides this default.
It is still validated as a finite non-negative omega-plane distance.

Eigenvalue search.
:func:`mark_unstable_band` runs one shift-invert eigensolve per target and
merges the eigenvalues, dropping repeats.
Without ``target`` the targets are spread over the requested range: both
ends and the centre at least, more when the range is wide compared with
``n_modes`` danger radii, up to 9.
Poles at the edges of the range are then found, not only those near the
median.
Each target costs one factorisation.
An explicit ``target`` (or ``evp_target``) gives that single target.

Smooth regions are opt-in.
By default every non-danger point gets the label ``"stable"``.
A point gets ``"smooth"`` only if it sits outside the danger radius.
It must also lie in a configured interval from ``smooth_regions`` (or
``solver.router.band_marking.smooth_regions``).
Each interval is either a ``(lo, hi)`` pair or a mapping with
``min``/``max`` keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from linstabpy.config import Config

STABLE = "stable"
NEAR_POLE = "near-pole"
SMOOTH = "smooth"


@dataclass(frozen=True)
class BandMarks:
    """Per-frequency EVP-distance labels for a beta slice."""

    omegas: np.ndarray
    labels: np.ndarray
    distances: np.ndarray
    danger_radius: float


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _band_cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    root = _as_mapping(cfg)
    direct = _as_mapping(root.get("band_marking"))
    if direct:
        return direct
    solver = _as_mapping(root.get("solver"))
    router = _as_mapping(solver.get("router"))
    return _as_mapping(router.get("band_marking"))


def _sweep_spacing_radius(omegas: np.ndarray) -> float:
    finite_unique = np.unique(omegas[np.isfinite(omegas)])
    if finite_unique.size < 2:
        return 0.0
    return 0.5 * float(np.median(np.diff(finite_unique)))


def _danger_radius(cfg: Mapping[str, Any] | None, omegas: np.ndarray) -> float:
    root = _as_mapping(cfg)
    direct = _as_mapping(root.get("band_marking"))
    solver = _as_mapping(root.get("solver"))
    router = _as_mapping(solver.get("router"))
    nested = _as_mapping(router.get("band_marking"))
    if "danger_radius" in root:
        value = root["danger_radius"]
    elif "danger_radius" in direct:
        value = direct["danger_radius"]
    elif "danger_radius" in nested:
        value = nested["danger_radius"]
    else:
        value = _sweep_spacing_radius(omegas)
    radius = float(value)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError(f"danger_radius must be a finite non-negative float, got {value!r}")
    return radius


def _smooth_regions(cfg: Mapping[str, Any] | None) -> tuple[tuple[float, float], ...]:
    root = _as_mapping(cfg)
    band = _band_cfg(root)
    raw = root.get("smooth_regions", band.get("smooth_regions", ()))
    regions: list[tuple[float, float]] = []
    for item in raw or ():
        if isinstance(item, Mapping):
            lo = float(item["min"])
            hi = float(item["max"])
        else:
            lo = float(item[0])
            hi = float(item[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
            raise ValueError(f"invalid smooth region {item!r}")
        regions.append((lo, hi))
    return tuple(regions)


def label_band(
    eigs: Sequence[complex],
    omegas: Sequence[float],
    cfg: Mapping[str, Any] | None = None,
    *,
    matlab_parity: bool = False,
) -> BandMarks:
    """Label requested real frequencies by distance to a supplied EVP spectrum.

    Parameters
    ----------
    eigs:
        Complex temporal eigenvalues in the ``omega = eig(i*L0)`` plane.
        The compared resolvent poles are ``+eigs``, or ``-eigs`` when
        ``matlab_parity`` is true.
    omegas:
        Requested real forcing frequencies in that same omega normalization.
    cfg:
        Optional band-marking configuration.  See the module docstring for keys.
    matlab_parity:
        True when the resolvent solve uses the LinStab2D sign convention.

    Returns
    -------
    BandMarks
        Frequencies, labels, nearest-EVP distances, and the radius used.
    """
    eig_arr = np.asarray(eigs, dtype=np.complex128).ravel()
    omega_arr = np.asarray(omegas, dtype=float).ravel()
    radius = _danger_radius(cfg, omega_arr)

    if eig_arr.size == 0:
        distances = np.full(omega_arr.shape, np.inf, dtype=float)
    else:
        poles = -eig_arr if matlab_parity else eig_arr
        distances = np.min(np.abs(omega_arr[:, None] - poles[None, :]), axis=1)

    labels = np.full(omega_arr.shape, STABLE, dtype=object)
    labels[distances < radius] = NEAR_POLE

    if omega_arr.size:
        smooth_mask = labels != NEAR_POLE
        for lo, hi in _smooth_regions(cfg):
            smooth_mask &= ~np.isnan(omega_arr)
            labels[smooth_mask & (omega_arr >= lo) & (omega_arr <= hi)] = SMOOTH

    return BandMarks(
        omegas=omega_arr,
        labels=labels,
        distances=distances,
        danger_radius=radius,
    )


_TARGET_OFFSET = 1.0e-6
_MAX_TARGETS = 9  # each target is one full-size shift-invert factorisation


def _evp_targets(omega_arr: np.ndarray, radius: float, n_modes: int, sign: float) -> list[complex]:
    """Return EVP targets that cover the requested range, ends and centre first.

    ``sign`` is ``-1`` for the LinStab2D convention, whose poles sit at
    ``-omega_eig``, and ``+1`` otherwise.
    """
    finite = omega_arr[np.isfinite(omega_arr)]
    if finite.size == 0:
        return [0j]
    lo, hi = float(finite.min()), float(finite.max())
    count = 3
    if radius > 0.0:
        count = max(count, int(np.ceil((hi - lo) / (n_modes * radius))) + 1)
    points = np.unique(sign * np.linspace(lo, hi, min(count, _MAX_TARGETS)))
    # A small imaginary offset keeps a target that sits exactly on an eigenvalue
    # (a range end on a pole) from making the shifted matrix singular.
    return [complex(point, _TARGET_OFFSET * (1.0 + abs(point))) for point in points]


def _merge_eigenvalues(groups: Sequence[Any]) -> np.ndarray:
    """Join eigenvalue arrays and drop values that repeat within 1e-8 (relative)."""
    kept: list[complex] = []
    for value in np.concatenate([np.asarray(g, dtype=np.complex128).ravel() for g in groups]):
        if not any(abs(value - k) <= 1.0e-8 * (1.0 + abs(k)) for k in kept):
            kept.append(complex(value))
    return np.asarray(kept, dtype=np.complex128)


def _get_first(cfg: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in cfg:
            return cfg[name]
    return default


def _evp_eigenvalues(
    L0_beta: Any,
    *,
    n_modes: int,
    target: complex,
    ncv: int | None,
    solver_cfg: Any,
) -> tuple[Any, Any]:
    from linstabpy.stability import eigenvalues

    return eigenvalues(
        L0_beta,
        n_modes=n_modes,
        target=target,
        ncv=ncv,
        solver_cfg=solver_cfg,
    )


def mark_unstable_band(
    L0_beta: Any,
    f_range: Sequence[float],
    cfg: Mapping[str, Any] | None = None,
    *,
    matlab_parity: bool | None = None,
) -> BandMarks:
    """Compute EVP values near a beta slice and label the requested band.

    This is intentionally a thin wrapper around :func:`linstabpy.stability.eigenvalues`;
    it does not assemble operators or run a resolvent sweep.
    ``matlab_parity=None`` reads ``solver.resolvent_sign_convention`` from ``cfg``.
    """
    root = _as_mapping(cfg)
    band = _band_cfg(root)
    omega_arr = np.asarray(f_range, dtype=float).ravel()
    if matlab_parity is None:
        matlab_parity = Config.load(root).resolvent.matlab_parity
    n_modes = int(_get_first(band, "n_modes", "evp_n_modes", default=8))
    explicit = _get_first(band, "target", "evp_target", default=None)
    if explicit is not None:
        targets = [complex(explicit)]
    else:
        # Physical poles sit at +omega_eig, so the eigenvalues relevant to the requested
        # real frequencies live in the range itself; under matlab_parity, in its mirror.
        radius = _danger_radius(root, omega_arr)
        targets = _evp_targets(omega_arr, radius, n_modes, -1.0 if matlab_parity else 1.0)
    ncv = _get_first(band, "ncv", "evp_ncv", default=None)
    solver_cfg = root.get("solver", root if root else None)

    groups = [
        _evp_eigenvalues(
            L0_beta,
            n_modes=n_modes,
            target=target,
            ncv=None if ncv is None else int(ncv),
            solver_cfg=solver_cfg,
        )[0]
        for target in targets
    ]
    eigs = _merge_eigenvalues(groups)
    return label_band(eigs, omega_arr, root, matlab_parity=matlab_parity)


__all__ = ["BandMarks", "NEAR_POLE", "SMOOTH", "STABLE", "label_band", "mark_unstable_band"]
