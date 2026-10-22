# SPDX-License-Identifier: GPL-3.0-only
"""Acceptance gates for stability eigenvalues and postcheck residuals."""

from __future__ import annotations

import numpy as np


def _postcheck_target_distance(
    omega: np.ndarray,
    target: complex,
    *,
    conjugate_aware: bool,
) -> np.ndarray:
    """Per-mode distance from the shift-invert target, optionally conjugate-aware.

    Background
    ----------
    For a real-valued baseflow, the linearised Navier-Stokes operator ``L0`` has
    real matrix entries, and ``omega = eig(i*L0)``.
    If ``(omega, v)`` is an eigenpair, conjugate both sides of
    ``i L0 v = omega v``.
    That gives ``i L0 conj(v) = -conj(omega) conj(v)``.
    So ``(-conj(omega), conj(v))`` is also an eigenpair: the same growth rate
    ``Im(omega)`` with the opposite frequency.
    The two are the same physical instability wave in two complex
    representations.

    When ``conjugate_aware=True``, the distance is
    ``min(|omega - target|, |omega + conj(target)|)``.
    The conjugate partner then passes the same target-distance gate as
    the primary branch.
    Use ``conjugate_aware=False`` for complex operators where
    the real-``L`` conjugate symmetry no longer applies.
    """
    omega_arr = np.asarray(omega, dtype=complex)
    direct = np.abs(omega_arr - complex(target))
    if not conjugate_aware:
        return direct
    conj_dist = np.abs(omega_arr + np.conjugate(complex(target)))
    return np.minimum(direct, conj_dist)


def evaluate_stability_acceptance(
    omega: np.ndarray,
    postcheck_residual_rel: np.ndarray,
    postcheck_target_distance: np.ndarray,
    *,
    postcheck_rel_max: float,
    target_distance_max: float,
    omega_real_abs_max: float = float("nan"),
    omega_imag_abs_max: float = float("nan"),
    conjugate_aware: bool = True,
) -> dict[str, object]:
    """Apply optional acceptance gates to returned stability modes.

    Temporal eigenvalue convention
    ------------------------------
    ``omega`` stores complex temporal eigenvalues directly.

    - ``real(omega)``  = oscillation frequency.
    - ``imag(omega)``  = growth/decay rate (positive = unstable).

    All thresholds are optional. When every threshold is NaN, the
    function returns an empty dictionary so older workflows remain unchanged.

    Gate semantics
    --------------
    A mode passes acceptance iff every enabled gate passes.

    - postcheck_rel_max: explicit full-operator residual, relative.
    - target_distance_max: complex distance from the solver shift target
      (see ``conjugate_aware`` below).
    - omega_real_abs_max: cap on |Re(omega)|.
      Physically a filter on oscillation frequency magnitude.
    - omega_imag_abs_max: cap on |Im(omega)|.
      Physically a filter on growth/decay rate magnitude.

    The ``conjugate_aware`` flag (default ``True``)
    -----------------------------------------------
    For a real-valued baseflow the linearised operator ``L0`` is real.
    With ``omega = eig(i*L0)`` the spectrum is symmetric under
    ``omega -> -conj(omega)``.
    ``(omega, v)`` and ``(-conj(omega), conj(v))`` are both eigenpairs of ``i*L0``.
    These two eigenpairs represent the SAME physical instability wave.
    Their sum is the real, observable perturbation field.
    A shift-invert EVP with a complex ``omega_target`` that has non-zero
    real part sits near ONE branch of this pair.
    The other branch lives near ``-conj(omega_target)``, with the same
    growth rate and the opposite frequency.

    Why this is the default.
    For the SWBLI mode families tracked in this project (see ``SWBLI.md``).

    * BG-S (bubble global stationary mode): ``Re(omega) ~ 0``.
      Target and its partner coincide on the imaginary axis.
      The metric is a no-op here. Steady modes pass cleanly.
    * OM-1 (oblique first-Mack wave): a pair
      ``(omega_r + 1j*omega_i, -omega_r + 1j*omega_i)``.
      Without conjugate awareness the Euclidean ``|omega - target|``
      metric would silently drop the mirror partner.
      That would lose half the physical mode.
    * LU-S (lift-up streak response): near-zero frequency, same as BG-S.
      No effect either way.

    With ``conjugate_aware=True``, the distance metric is
    ``min(|omega - target|, |omega + conj(target)|)``.
    The conjugate partner then passes the same gate as the primary.

    When to disable
    ~~~~~~~~~~~~~~~
    Set ``conjugate_aware=False`` only when the underlying operator is no
    longer real.
    Examples: a spanwise wavenumber ``beta != 0`` (the operator carries
    ``i*beta`` terms), Floquet-shifted problems, complex base-flow
    perturbation tracking, or any formulation that explicitly breaks the
    real-``L`` symmetry.
    In those cases, admitting ``-conj(target)`` would over-accept
    unphysical partners.

    Provenance
    ~~~~~~~~~~
    Upstream, the solver already uses this metric in
    ``_postcheck_target_distance`` when ``conjugate_aware=True``.
    Acceptance only decides the gate.
    The result dict stores the flag as ``acceptance_conjugate_aware``.
    The downstream ``.npz`` then keeps the metric that produced the verdicts.

    Back-compat note.
    Flipping the default from ``False`` to ``True`` widens acceptance
    on real-baseflow runs.
    The old metric silently discarded the conjugate partner.
    Saved ``.npz`` artefacts produced before commit
    ``d{conjugate-default}`` may show a different accepted set on rerun.
    That is a correction, not a regression.

    Additive return keys. Always present when any gate is active.
    -------------------------------------------------------------
    - ``rejected_indices``: int32 array of mode indices the gates reject.
    - ``reject_reason_code``: int32 array of length ``n_modes``.
      Bitmask per mode encoding which gates failed (0 = accepted).
      bit 0 (=1): postcheck_rel gate failed.
      bit 1 (=2): target_distance gate failed.
      bit 2 (=4): omega_real gate failed.
      bit 3 (=8): omega_imag gate failed.
    - ``acceptance_conjugate_aware``: bool.
      Records whether conjugate-aware target distance produced the verdicts.
    """

    use_postcheck_gate = np.isfinite(postcheck_rel_max)
    use_target_gate = np.isfinite(target_distance_max)
    use_omega_real_gate = np.isfinite(omega_real_abs_max)
    use_omega_imag_gate = np.isfinite(omega_imag_abs_max)
    if not (use_postcheck_gate or use_target_gate or use_omega_real_gate or use_omega_imag_gate):
        return {}

    omega_arr = np.asarray(omega, dtype=complex).reshape(-1)
    n_modes = int(omega_arr.size)
    accepted = np.ones(n_modes, dtype=bool)
    reject_reason_code = np.zeros(n_modes, dtype=np.int32)

    post_rel = np.asarray(postcheck_residual_rel, dtype=float).reshape(-1)
    if use_postcheck_gate:
        post_rel_mask = np.zeros(n_modes, dtype=bool)
        n_rel = min(n_modes, post_rel.size)
        if n_rel > 0:
            vals = post_rel[:n_rel]
            post_rel_mask[:n_rel] = np.isfinite(vals) & (vals <= postcheck_rel_max)
        failed = ~post_rel_mask
        reject_reason_code[failed] |= np.int32(1)
        accepted &= post_rel_mask

    target_dist = np.asarray(postcheck_target_distance, dtype=float).reshape(-1)
    if use_target_gate:
        target_mask = np.zeros(n_modes, dtype=bool)
        n_dist = min(n_modes, target_dist.size)
        if n_dist > 0:
            vals = target_dist[:n_dist]
            target_mask[:n_dist] = np.isfinite(vals) & (vals <= target_distance_max)
        failed = ~target_mask
        reject_reason_code[failed] |= np.int32(2)
        accepted &= target_mask

    omega_real = np.real(omega_arr)
    if use_omega_real_gate:
        real_mask = np.isfinite(omega_real) & (np.abs(omega_real) <= omega_real_abs_max)
        failed = ~real_mask
        reject_reason_code[failed] |= np.int32(4)
        accepted &= real_mask

    omega_imag = np.imag(omega_arr)
    if use_omega_imag_gate:
        imag_mask = np.isfinite(omega_imag) & (np.abs(omega_imag) <= omega_imag_abs_max)
        failed = ~imag_mask
        reject_reason_code[failed] |= np.int32(8)
        accepted &= imag_mask

    accepted_indices = np.flatnonzero(accepted).astype(np.int32)
    rejected_indices = np.flatnonzero(~accepted).astype(np.int32)
    result: dict[str, object] = {
        "acceptance_postcheck_rel_max": float(postcheck_rel_max),
        "acceptance_target_distance_max": float(target_distance_max),
        "acceptance_omega_real_abs_max": float(omega_real_abs_max),
        "acceptance_omega_imag_abs_max": float(omega_imag_abs_max),
        "acceptance_conjugate_aware": bool(conjugate_aware),
        "accepted_count": int(accepted_indices.size),
        "accepted_indices": accepted_indices,
        "rejected_indices": rejected_indices,
        "reject_reason_code": reject_reason_code,
    }

    if accepted_indices.size and post_rel.size:
        accepted_post_rel = post_rel[accepted_indices]
        finite = accepted_post_rel[np.isfinite(accepted_post_rel)]
        if finite.size:
            result["accepted_postcheck_rel_min"] = float(np.min(finite))
            result["accepted_postcheck_rel_median"] = float(np.median(finite))

    if accepted_indices.size and target_dist.size:
        accepted_target = target_dist[accepted_indices]
        finite = accepted_target[np.isfinite(accepted_target)]
        if finite.size:
            result["accepted_target_distance_min"] = float(np.min(finite))
            result["accepted_target_distance_median"] = float(np.median(finite))

    if accepted_indices.size:
        accepted_omega = omega_arr[accepted_indices]
        result["accepted_omega_real_min"] = float(np.min(np.real(accepted_omega)))
        result["accepted_omega_real_max"] = float(np.max(np.real(accepted_omega)))
        result["accepted_omega_imag_min"] = float(np.min(np.imag(accepted_omega)))
        result["accepted_omega_imag_max"] = float(np.max(np.imag(accepted_omega)))

    return result


def check_conjugate_pair_completeness(
    omega: np.ndarray,
    accepted_indices: np.ndarray,
    *,
    pair_tol_real: float = 1e-6,
    pair_tolerance: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Diagnostic: check whether accepted complex modes have their conjugate partners.

    For each accepted mode with ``|Re(omega)| > pair_tol_real``, search all
    returned omegas.
    Look for the partner ``-conj(omega)`` within ``pair_tolerance``.
    A mode with ``Re(omega) = 0`` is its own partner.

    Does NOT modify the accepted/rejected sets. Pure diagnostic.

    Parameters
    ----------
    omega : np.ndarray
        All returned complex eigenvalues (not just accepted ones).
    accepted_indices : np.ndarray
        Integer indices into ``omega`` of the accepted modes.
    pair_tol_real : float
        Minimum |Re(omega)| for a conjugate-pair candidate.
    pair_tolerance : float
        Maximum distance at which the expected conjugate partner counts as found.

    Returns
    -------
    dict with keys:
    - ``missing_partner_indices`` : int32 array of accepted-mode indices whose
      conjugate partner is absent from ``omega``.
    - ``partner_rejected_indices`` : int32 array of accepted-mode indices whose
      conjugate partner exists in ``omega`` but was not accepted.
    """
    omega_arr = np.asarray(omega, dtype=complex).reshape(-1)
    accepted_idx = np.asarray(accepted_indices, dtype=np.int32).reshape(-1)
    accepted_set = set(int(i) for i in accepted_idx)

    missing: list[int] = []
    partner_rejected: list[int] = []

    for ai in accepted_idx:
        oi = omega_arr[int(ai)]
        if abs(np.real(oi)) <= pair_tol_real:
            # On the imaginary axis the mode is its own partner.
            continue
        conj_oi = -np.conj(oi)
        # Search ALL omegas for the conjugate partner.
        dists = np.abs(omega_arr - conj_oi)
        # Exclude self (the mode itself is not its own partner when off-axis).
        close = np.flatnonzero(dists <= pair_tolerance)
        # Remove self from candidates.
        close = close[close != int(ai)]
        if close.size == 0:
            missing.append(int(ai))
        elif not any(int(c) in accepted_set for c in close):
            partner_rejected.append(int(ai))
        # else: partner found AND accepted — pair complete, nothing to record.

    return {
        "missing_partner_indices": np.array(missing, dtype=np.int32),
        "partner_rejected_indices": np.array(partner_rejected, dtype=np.int32),
    }
