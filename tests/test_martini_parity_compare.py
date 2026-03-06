"""Focused tests for Martini parity comparison helpers."""

from __future__ import annotations

import numpy as np

from tools.parity.compare_martini_examples import compare_resolvent_case


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
