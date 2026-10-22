# SPDX-License-Identifier: GPL-3.0-only
"""The raw-eigenvalue diagnostic path reports omega = eig(i*L0) like the main path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
import importlib

from linstabpy.utils.petsc_core import scipy_to_petsc

driver_run = importlib.import_module("linstabpy.driver.run")

NGP = 8
OMEGAS = 0.1 * np.arange(1, 5 * NGP + 1) + 0.01j * np.arange(1, 5 * NGP + 1)
TARGET = 0.52 + 0.05j
N_MODES = 3


def _run_raw(L0: object) -> np.ndarray:
    empty = np.empty(0, dtype=np.int64)
    result = driver_run._stage_run_stability(
        run_stability=True,
        use_raw_eigs=True,
        reduce_dirichlet=False,
        use_root_operator=True,
        reduce_dirichlet_set="column_zeroed",
        constrained_dirichlet_indices=empty,
        effective_dirichlet_indices=empty,
        effective_neumann_indices=empty,
        overlap_indices=empty,
        L0=L0,
        W_diag=np.ones(5 * NGP),
        omega_target=TARGET,
        n_modes=N_MODES,
        mesh=SimpleNamespace(ngp=NGP),
        solver_cfg={"eigen": {"st_solver": "mumps"}},
        total_bc_indices=empty,
        stability_acceptance_conjugate_aware=False,
        heartbeat_seconds=0.0,
    )
    return result.omega


def _expected() -> np.ndarray:
    return OMEGAS[np.argsort(np.abs(OMEGAS - TARGET))[:N_MODES]]


def test_raw_eigs_report_omega_near_target_scipy() -> None:
    """L0 = -i*diag(omega) has eig(i*L0) = omega; the raw path must return those."""
    L0 = sp.diags(-1j * OMEGAS).tocsr()
    np.testing.assert_allclose(_run_raw(L0), _expected(), atol=1e-8)


def test_raw_eigs_accept_petsc_matrix() -> None:
    """A PETSc L0 (the driver's default layout) no longer fails on .tocsc()."""
    L0 = scipy_to_petsc(sp.diags(-1j * OMEGAS).tocsr())
    np.testing.assert_allclose(_run_raw(L0), _expected(), atol=1e-8)
