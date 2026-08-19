# SPDX-License-Identifier: GPL-3.0-only
"""Tests for fail-fast shifted-KSP backend accuracy probes."""

from __future__ import annotations

import numpy as np
import pytest
from petsc4py import PETSc

from linstabpy.linalg.backends import (
    backend_probe_is_preconditioner_role,
    shifted_ksp_probe_residual,
    should_probe_backend,
)
from linstabpy.solvers import eigenvalue
from linstabpy.stability import eigenvalues as stability_eigenvalues


def _create_aij(diagonal: list[complex]) -> PETSc.Mat:
    mat = PETSc.Mat().createAIJ(size=(len(diagonal), len(diagonal)), comm=PETSc.COMM_WORLD)
    rstart, rend = mat.getOwnershipRange()
    for row in range(rstart, rend):
        mat.setValue(row, row, diagonal[row])
    mat.assemble()
    return mat


def _create_ksp(mat: PETSc.Mat, *, ksp_type: str, pc_type: str) -> PETSc.KSP:
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(mat)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    if pc_type == PETSc.PC.Type.LU or pc_type == "lu":
        pc.setFactorSolverType("mumps")
    ksp.setUp()
    return ksp


def test_shifted_ksp_probe_residual_preonly_lu_is_near_machine_precision() -> None:
    mat = _create_aij([2.0 + 0.5j, -3.0 + 0.25j, 4.0 - 1.0j])
    ksp = _create_ksp(mat, ksp_type=PETSc.KSP.Type.PREONLY, pc_type=PETSc.PC.Type.LU)
    try:
        probe = shifted_ksp_probe_residual(ksp)
    finally:
        ksp.destroy()
        mat.destroy()

    assert probe["residual_abs"] >= 0.0
    assert probe["residual_rel"] < 1.0e-13
    assert probe["ksp_type"] == PETSc.KSP.Type.PREONLY
    assert probe["pc_type"] == PETSc.PC.Type.LU
    assert probe["pc_factor_solver_type"] == "mumps"


def test_shifted_ksp_probe_residual_weak_pc_reports_large_true_residual() -> None:
    mat = _create_aij([1.0e12 + 0j, 1.0 + 0j, 2.0 + 0j])
    ksp = _create_ksp(mat, ksp_type=PETSc.KSP.Type.RICHARDSON, pc_type=PETSc.PC.Type.NONE)
    ksp.setTolerances(rtol=0.0, atol=0.0, max_it=1)
    try:
        probe = shifted_ksp_probe_residual(ksp)
    finally:
        ksp.destroy()
        mat.destroy()

    assert probe["residual_rel"] > 1.0e5
    assert probe["ksp_iterations"] == 1
    assert probe["ksp_type"] == PETSc.KSP.Type.RICHARDSON
    assert probe["pc_type"] == PETSc.PC.Type.NONE


@pytest.mark.parametrize(
    ("resolved_backend", "probe_cfg", "expected"),
    [
        ("mumps", None, False),
        ("mumps", {}, False),
        ("mumps", {"enabled": True}, False),
        ("strumpack", None, True),
        ("strumpack", {}, True),
        ("strumpack", {"enabled": True}, True),
        ("strumpack", {"enabled": False}, False),
        ("superlu_dist", {"enabled": False}, False),
        (None, None, False),
    ],
)
def test_should_probe_backend_truth_table(resolved_backend, probe_cfg, expected) -> None:
    assert should_probe_backend(resolved_backend, probe_cfg) is expected


@pytest.mark.parametrize(
    "st_solver", ["iterative", "krylov", "slepc", "fgmres_lu", "strumpack_fgmres"]
)
def test_should_probe_backend_treats_iterative_outer_as_preconditioner_role(st_solver) -> None:
    assert backend_probe_is_preconditioner_role(st_solver) is True
    assert should_probe_backend("mumps", {}, st_solver=st_solver) is True
    assert should_probe_backend("mumps", {"enabled": False}, st_solver=st_solver) is False


@pytest.mark.parametrize("st_solver", ["mumps", "direct", "lu", None])
def test_should_probe_backend_keeps_direct_default_mumps_exemption(st_solver) -> None:
    assert backend_probe_is_preconditioner_role(st_solver) is False
    assert should_probe_backend("mumps", {}, st_solver=st_solver) is False


def test_temporal_default_mumps_path_does_not_invoke_probe(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_FACTOR_BACKEND", raising=False)

    def fail_probe(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("MUMPS default path must not run backend probe")

    monkeypatch.setattr(eigenvalue, "shifted_ksp_probe_residual", fail_probe)
    operator = np.diag([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])

    omega, _ = stability_eigenvalues(operator, n_modes=1, target=2.2j)

    assert omega.shape == (1,)


def test_temporal_backend_probe_failure_raises_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_FACTOR_BACKEND", raising=False)
    # solver_cfg.eigen.factor_backend below selects the resolved backend
    # (cfg.eigen.st_factor_backend == "strumpack").
    # Production no longer reads a module-level eigenvalue.resolve_factor_backend,
    # so this check needs no monkeypatch there.
    monkeypatch.setattr(eigenvalue, "apply_factor_backend", lambda *args, **kwargs: "strumpack")
    monkeypatch.setattr(
        eigenvalue,
        "shifted_ksp_probe_residual",
        lambda ksp: {
            "residual_abs": 1.0e-6,
            "residual_rel": 1.0e-6,
            "ksp_iterations": 1,
            "ksp_reason": 1,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_solver_type": "strumpack",
        },
    )
    operator = np.diag([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])

    with pytest.raises(RuntimeError) as excinfo:
        stability_eigenvalues(
            operator,
            n_modes=1,
            target=2.2j,
            solver_cfg={
                "eigen": {
                    "factor_backend": "strumpack",
                    "backend_probe": {"residual_rel_max": 1.0e-8},
                }
            },
        )

    message = str(excinfo.value)
    assert "factor backend 'strumpack' failed accuracy probe" in message
    assert "||Kx-b||/||b|| = 1.000e-06 > 1.000e-08" in message
    assert "set eigen.backend_probe.enabled=false to override" in message


@pytest.mark.parametrize(
    "st_solver", ["iterative", "krylov", "slepc", "fgmres_lu", "strumpack_fgmres"]
)
def test_temporal_backend_probe_preconditioner_role_does_not_reject_moderate_residual(
    monkeypatch, st_solver
) -> None:
    monkeypatch.delenv("LINSTABPY_FACTOR_BACKEND", raising=False)
    monkeypatch.setattr(eigenvalue, "apply_factor_backend", lambda *args, **kwargs: "strumpack")
    monkeypatch.setattr(
        eigenvalue,
        "shifted_ksp_probe_residual",
        lambda ksp: {
            "residual_abs": 1.0e-6,
            "residual_rel": 1.0e-6,
            "ksp_iterations": 1,
            "ksp_reason": 1,
            "ksp_type": "fgmres",
            "pc_type": "lu",
            "pc_factor_solver_type": "strumpack",
        },
    )
    operator = np.diag([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
    eigen_cfg = {
        "st_solver": st_solver,
        "factor_backend": "strumpack",
        "pc_type": "lu",
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-12,
        "ksp_max_it": 20,
        "backend_probe": {"residual_rel_max": 1.0e-8},
    }

    omega, _ = stability_eigenvalues(
        operator,
        n_modes=1,
        target=2.2j,
        solver_cfg={"eigen": eigen_cfg},
    )

    assert omega.shape == (1,)


@pytest.mark.parametrize(
    "st_solver", ["iterative", "krylov", "slepc", "fgmres_lu", "strumpack_fgmres"]
)
def test_temporal_backend_probe_preconditioner_role_still_rejects_catastrophic_residual(
    monkeypatch, st_solver
) -> None:
    # The relaxed preconditioner-role ceiling is a SANITY bound, not a free pass:
    # a residual above PRECONDITIONER_PROBE_RESIDUAL_REL_MAX (~1e-4) means the
    # factor backend is doing essentially nothing as a preconditioner and must
    # still fail fast. This pins the safety claim behind the relaxation.
    monkeypatch.delenv("LINSTABPY_FACTOR_BACKEND", raising=False)
    monkeypatch.setattr(eigenvalue, "apply_factor_backend", lambda *args, **kwargs: "strumpack")
    monkeypatch.setattr(
        eigenvalue,
        "shifted_ksp_probe_residual",
        lambda ksp: {
            "residual_abs": 1.0e-1,
            "residual_rel": 1.0e-1,
            "ksp_iterations": 1,
            "ksp_reason": 1,
            "ksp_type": "fgmres",
            "pc_type": "lu",
            "pc_factor_solver_type": "strumpack",
        },
    )
    operator = np.diag([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
    eigen_cfg = {
        "st_solver": st_solver,
        "factor_backend": "strumpack",
        "pc_type": "lu",
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-12,
        "ksp_max_it": 20,
        "backend_probe": {"residual_rel_max": 1.0e-8},
    }

    with pytest.raises(RuntimeError) as excinfo:
        stability_eigenvalues(
            operator,
            n_modes=1,
            target=2.2j,
            solver_cfg={"eigen": eigen_cfg},
        )

    assert "failed accuracy probe" in str(excinfo.value)
