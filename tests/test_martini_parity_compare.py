"""Focused tests for Martini parity comparison helpers."""

from __future__ import annotations

import numpy as np

from tools.parity.compare_martini_examples import _compare_cavity_case, compare_resolvent_case


def test_compare_resolvent_case_handles_matlab_mode_rows_and_phase() -> None:
    gains = np.array([3.0, 2.0, 1.0], dtype=float)

    responses_py = np.array(
        [
            [1.0 + 0.0j, 0.0 + 1.0j, 1.0 - 1.0j],
            [2.0 + 1.0j, 3.0 + 0.0j, 0.5 + 0.5j],
            [0.5 - 0.5j, 4.0 + 1.0j, 2.0 + 0.0j],
            [1.5 + 0.5j, 2.0 - 1.0j, 3.0 + 0.0j],
        ],
        dtype=complex,
    )
    forcings_py = np.array(
        [
            [0.5 + 0.5j, 1.0 + 0.0j, 1.5 - 0.5j],
            [1.0 + 0.0j, 0.5 + 0.5j, 2.0 + 1.0j],
            [1.5 - 0.5j, 2.0 + 0.0j, 0.5 + 0.5j],
            [2.0 + 1.0j, 1.5 - 0.5j, 1.0 + 0.0j],
        ],
        dtype=complex,
    )

    responses_mat = (responses_py * np.array([1j, -1j, np.exp(0.3j)])).T
    forcings_mat = (forcings_py * np.array([-1j, np.exp(-0.2j), 1j])).T

    report = compare_resolvent_case(
        gains_py=gains,
        responses_py=responses_py,
        forcings_py=forcings_py,
        gains_mat=gains,
        responses_mat=responses_mat,
        forcings_mat=forcings_mat,
        n_compare=3,
    )

    assert report["gains"]["max_abs_error"] == 0.0
    assert report["gains"]["max_rel_error"] == 0.0
    assert report["modes"]["max_forcing_rel_l2"] < 1e-12
    assert report["modes"]["max_response_rel_l2"] < 1e-12


def test_compare_cavity_case_uses_global_best_spectrum_pairs() -> None:
    matlab_direct_omega = np.array(
        [
            1.50 + 0.10j,
            0.86 - 0.32j,
            0.45 - 0.04j,
            0.46 - 0.07j,
        ],
        dtype=complex,
    )
    python_direct_omega = np.array(
        [
            1.45 + 0.95j,
            0.855 + 0.0j,
            0.8557 - 0.3196j,
            0.4526 - 0.0365j,
            0.4605 - 0.0650j,
        ],
        dtype=complex,
    )

    ndof = 300
    direct_modes = np.zeros((ndof, 4), dtype=complex)
    adjoint_modes = np.zeros((ndof, 4), dtype=complex)
    for i in range(4):
        direct_modes[i, i] = 1.0
        adjoint_modes[i, i] = np.exp(0.1j)

    python_npz = {
        "omega": python_direct_omega,
        # Python stores adjoint eigenvalues as i * conj(omega_direct).
        "omega_adj": 1j * np.conj(matlab_direct_omega),
        "direct_modes": np.column_stack(
            [
                np.full(ndof, 5.0 + 1.0j),
                direct_modes[:, 0],
                direct_modes[:, 1],
                direct_modes[:, 2],
                direct_modes[:, 3],
            ]
        ),
        "adjoint_modes": adjoint_modes,
    }
    matlab_direct = {
        "results_Cavity_direct__omega": matlab_direct_omega,
        "results_Cavity_direct__V": direct_modes,
    }
    matlab_adjoint = {
        "results_Cavity_adjoint__omegaAdj": np.conj(matlab_direct_omega),
        "results_Cavity_adjoint__VAdj": adjoint_modes,
    }

    report = _compare_cavity_case(
        python_npz=python_npz,
        matlab_direct=matlab_direct,
        matlab_adjoint=matlab_adjoint,
        n_compare=3,
        omega_target=1.5 + 1.0j,
    )

    assert report["direct"]["selection_strategy"] == "global_best_match"
    assert report["adjoint"]["selection_strategy"] == "global_best_match"
    assert report["direct"]["spectrum_max_abs_error"] < 1e-2
    assert report["adjoint"]["spectrum_max_abs_error"] < 1e-2
    assert report["direct"]["mode_max_rel_l2"] < 1e-12
    assert report["adjoint"]["mode_max_rel_l2"] < 1e-12
