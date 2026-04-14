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
