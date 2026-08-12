# SPDX-License-Identifier: GPL-3.0-only
"""Payload tests for resolvent residual flag."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from linstabpy.resolvent.analysis import resolvent_analysis


def test_resolvent_analysis_residuals_flag_appends_rho() -> None:
    diag = -np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    L0 = sp.diags(diag, format="csc", dtype=complex)
    W_diag = np.array([1.0, 2.0, 0.75, 1.5])
    solver_cfg = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}

    gains, responses, forcings = resolvent_analysis(
        L0,
        omega=0.25,
        n_modes=2,
        W_diag=W_diag,
        solver_cfg=solver_cfg,
        residuals=False,
    )
    flagged = resolvent_analysis(
        L0,
        omega=0.25,
        n_modes=2,
        W_diag=W_diag,
        solver_cfg=solver_cfg,
        residuals=True,
    )
    assert len(flagged) == 5
    np.testing.assert_array_equal(flagged[0], gains)
    np.testing.assert_array_equal(flagged[1], responses)
    np.testing.assert_array_equal(flagged[2], forcings)
