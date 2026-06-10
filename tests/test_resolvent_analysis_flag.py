# SPDX-License-Identifier: GPL-3.0-only
"""Payload-equivalence tests for resolvent residual flag."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

import linstabpy.solvers.resolvent as resolvent_shim
from linstabpy.resolvent.analysis import resolvent_analysis


def test_resolvent_analysis_residuals_flag_matches_legacy_shim_payload_exactly() -> None:
    diag = -np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    L0 = sp.diags(diag, format="csc", dtype=complex)
    W_diag = np.array([1.0, 2.0, 0.75, 1.5])
    solver_cfg = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}

    flagged_tuple = resolvent_analysis(
        L0,
        omega=0.25,
        n_modes=2,
        W_diag=W_diag,
        solver_cfg=solver_cfg,
        residuals=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_tuple = resolvent_shim.resolvent_analysis_with_residuals(
            L0,
            omega=0.25,
            n_modes=2,
            W_diag=W_diag,
            solver_cfg=solver_cfg,
        )

    assert any(item.category is DeprecationWarning for item in caught)
    names = ("gains", "responses", "forcings", "rho_fwd", "rho_adj")
    flagged = dict(zip(names, flagged_tuple, strict=True))
    legacy = dict(zip(names, legacy_tuple, strict=True))

    assert set(flagged) == set(legacy) == set(names)
    assert len(flagged_tuple) == len(legacy_tuple) == len(names)
    for name in names:
        np.testing.assert_array_equal(flagged[name], legacy[name])
