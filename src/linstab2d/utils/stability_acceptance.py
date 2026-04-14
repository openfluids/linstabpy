from __future__ import annotations

import numpy as np


def evaluate_stability_acceptance(
    omega: np.ndarray,
    postcheck_residual_rel: np.ndarray,
    postcheck_target_distance: np.ndarray,
    *,
    postcheck_rel_max: float,
    target_distance_max: float,
    omega_real_abs_max: float = float("nan"),
    omega_imag_abs_max: float = float("nan"),
) -> dict[str, object]:
    """Apply optional acceptance gates to returned stability modes.

    The solver stores complex temporal eigenvalues directly in ``omega``.
    In the current convention:
    - ``real(omega)`` is the oscillation frequency
    - ``imag(omega)`` is the growth/decay rate

    All thresholds are optional. When every threshold is disabled (NaN), the
    function returns an empty dictionary so older workflows remain unchanged.
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

    post_rel = np.asarray(postcheck_residual_rel, dtype=float).reshape(-1)
    if use_postcheck_gate:
        post_rel_mask = np.zeros(n_modes, dtype=bool)
        n_rel = min(n_modes, post_rel.size)
        if n_rel > 0:
            vals = post_rel[:n_rel]
            post_rel_mask[:n_rel] = np.isfinite(vals) & (vals <= postcheck_rel_max)
        accepted &= post_rel_mask

    target_dist = np.asarray(postcheck_target_distance, dtype=float).reshape(-1)
    if use_target_gate:
        target_mask = np.zeros(n_modes, dtype=bool)
        n_dist = min(n_modes, target_dist.size)
        if n_dist > 0:
            vals = target_dist[:n_dist]
            target_mask[:n_dist] = np.isfinite(vals) & (vals <= target_distance_max)
        accepted &= target_mask

    omega_real = np.real(omega_arr)
    if use_omega_real_gate:
        real_mask = np.isfinite(omega_real) & (np.abs(omega_real) <= omega_real_abs_max)
        accepted &= real_mask

    omega_imag = np.imag(omega_arr)
    if use_omega_imag_gate:
        imag_mask = np.isfinite(omega_imag) & (np.abs(omega_imag) <= omega_imag_abs_max)
        accepted &= imag_mask

    accepted_indices = np.flatnonzero(accepted).astype(np.int32)
    result: dict[str, object] = {
        "acceptance_postcheck_rel_max": float(postcheck_rel_max),
        "acceptance_target_distance_max": float(target_distance_max),
        "acceptance_omega_real_abs_max": float(omega_real_abs_max),
        "acceptance_omega_imag_abs_max": float(omega_imag_abs_max),
        "accepted_count": int(accepted_indices.size),
        "accepted_indices": accepted_indices,
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
