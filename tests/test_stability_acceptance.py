# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import numpy as np

from linstab2d.utils.stability_acceptance import evaluate_stability_acceptance


def test_acceptance_disabled_returns_empty_dict() -> None:
    omega = np.array([0.0 + 0.0j, 0.5 - 0.2j], dtype=complex)
    result = evaluate_stability_acceptance(
        omega,
        np.array([1.0e-6, 2.0e-6], dtype=float),
        np.array([0.1, 0.2], dtype=float),
        postcheck_rel_max=float("nan"),
        target_distance_max=float("nan"),
    )

    assert result == {}


def test_acceptance_combines_postcheck_target_and_spectral_box() -> None:
    omega = np.array(
        [
            0.0 + 0.0j,
            2.0 * np.pi - 1.0j,
            0.5 - 0.7j,
            7.0 + 0.0j,
            0.0 + 1.2j,
            0.2 - 0.3j,
        ],
        dtype=complex,
    )
    post_rel = np.array([1.0e-6, 1.0e-6, 2.0e-4, 1.0e-6, 1.0e-6, 1.0e-6], dtype=float)
    target_dist = np.array([0.1, 0.4, 0.2, 0.1, 0.1, 0.9], dtype=float)

    result = evaluate_stability_acceptance(
        omega,
        post_rel,
        target_dist,
        postcheck_rel_max=1.0e-5,
        target_distance_max=0.5,
        omega_real_abs_max=2.0 * np.pi,
        omega_imag_abs_max=1.0,
    )

    assert result["accepted_count"] == 2
    np.testing.assert_array_equal(result["accepted_indices"], np.array([0, 1], dtype=np.int32))
    assert result["accepted_omega_real_min"] == 0.0
    assert result["accepted_omega_real_max"] == 2.0 * np.pi
    assert result["accepted_omega_imag_min"] == -1.0
    assert result["accepted_omega_imag_max"] == 0.0


def test_acceptance_spectral_box_only_filters_outside_interest_region() -> None:
    omega = np.array(
        [
            -0.8 - 0.9j,
            0.0 - 0.7j,
            6.4 - 0.1j,
            0.3 - 1.1j,
        ],
        dtype=complex,
    )

    result = evaluate_stability_acceptance(
        omega,
        np.array([], dtype=float),
        np.array([], dtype=float),
        postcheck_rel_max=float("nan"),
        target_distance_max=float("nan"),
        omega_real_abs_max=2.0 * np.pi,
        omega_imag_abs_max=1.0,
    )

    assert result["accepted_count"] == 2
    np.testing.assert_array_equal(result["accepted_indices"], np.array([0, 1], dtype=np.int32))


def test_reject_reason_code_bitmask() -> None:
    """Front A: verify per-mode bitmask encoding of failed gates."""
    # Mode 0: all gates pass -> code 0
    # Mode 1: postcheck fails only -> code 1 (bit 0)
    # Mode 2: postcheck AND target both fail -> code 3 (bits 0+1)
    omega = np.array([0.1 - 0.2j, 0.1 - 0.2j, 0.1 - 0.2j], dtype=complex)
    post_rel = np.array([1.0e-6, 1.0e-3, 1.0e-3], dtype=float)   # mode 1,2 fail threshold 1e-5
    target_dist = np.array([0.05, 0.05, 0.8], dtype=float)        # mode 2 fails threshold 0.3

    result = evaluate_stability_acceptance(
        omega,
        post_rel,
        target_dist,
        postcheck_rel_max=1.0e-5,
        target_distance_max=0.3,
    )

    np.testing.assert_array_equal(
        result["reject_reason_code"],
        np.array([0, 1, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(result["rejected_indices"], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(result["accepted_indices"], np.array([0], dtype=np.int32))
    # Default in this project is conjugate_aware=True (real-baseflow assumption).
    # The test passes conjugate_aware implicitly -> default True.
    assert result["acceptance_conjugate_aware"] is True


def test_conjugate_aware_distance_accepts_conjugate() -> None:
    """Front B: verify conjugate-aware distance accepts the physical conjugate of the target."""
    target = 0.5 - 1.1j
    # Mode 0: exactly on target, Mode 1: conjugate of target, Mode 2: far from both
    omega = np.array([0.5 - 1.1j, 0.5 + 1.1j, 2.0 + 2.0j], dtype=complex)
    post_rel = np.array([1.0e-6, 1.0e-6, 1.0e-6], dtype=float)
    # Pre-compute distances against target for reference; gate uses conjugate_aware path
    target_dist_direct = np.abs(omega - target)

    # Without conjugate_aware: only mode 0 (distance 0) passes target_distance_max=0.3
    result_plain = evaluate_stability_acceptance(
        omega,
        post_rel,
        target_dist_direct,
        postcheck_rel_max=1.0e-5,
        target_distance_max=0.3,
        conjugate_aware=False,
    )
    np.testing.assert_array_equal(result_plain["accepted_indices"], np.array([0], dtype=np.int32))
    assert result_plain["acceptance_conjugate_aware"] is False

    # With conjugate_aware=True but target_dist_direct still passed in:
    # The conjugate_aware flag in evaluate_stability_acceptance controls the metadata record;
    # the actual distance array comes pre-computed from _postcheck_stability_modes.
    # So to test the full end-to-end B2/B3 path, we simulate what results_runtime does:
    # re-compute target_dist with conjugate awareness before calling evaluate_stability_acceptance.
    from linstab2d.utils.stability_acceptance import _postcheck_target_distance

    target_dist_conj = _postcheck_target_distance(omega, target, conjugate_aware=True)
    result_conj = evaluate_stability_acceptance(
        omega,
        post_rel,
        target_dist_conj,
        postcheck_rel_max=1.0e-5,
        target_distance_max=0.3,
        conjugate_aware=True,
    )
    # Mode 0 (distance 0) and mode 1 (conjugate, distance 0) both pass; mode 2 rejected
    np.testing.assert_array_equal(result_conj["accepted_indices"], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(result_conj["rejected_indices"], np.array([2], dtype=np.int32))
    assert result_conj["acceptance_conjugate_aware"] is True


def test_conjugate_pair_completeness() -> None:
    """Fix #5: conjugate-pair completeness diagnostic."""
    from linstab2d.utils.stability_acceptance import check_conjugate_pair_completeness

    # Omega list: modes 0 and 1 are a conjugate pair, mode 2 has no partner in the list.
    omega = np.array([0.5 + 1.0j, 0.5 - 1.0j, 3.0 + 2.0j], dtype=complex)
    # Accepted: modes 0 and 2 only (mode 1 — the partner of mode 0 — was rejected).
    accepted_indices = np.array([0, 2], dtype=np.int32)

    result = check_conjugate_pair_completeness(omega, accepted_indices)

    # Mode 0's partner (mode 1, conj = 0.5 - 1.0j) is found in omega but was rejected.
    np.testing.assert_array_equal(
        result["partner_rejected_indices"],
        np.array([0], dtype=np.int32),
    )
    # Mode 2's partner (conj = 3.0 - 2.0j) does not exist in omega at all.
    np.testing.assert_array_equal(
        result["missing_partner_indices"],
        np.array([2], dtype=np.int32),
    )
