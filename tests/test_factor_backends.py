# SPDX-License-Identifier: GPL-3.0-only
"""Tests for PETSc sparse-direct factor backend selection."""

from __future__ import annotations

import importlib
import re
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from petsc4py import PETSc

import linstabpy.linalg.backends as backends
from linstabpy.config import Config
from linstabpy.linalg import Backend
from linstabpy.linalg.backends import available, ensure, select
from linstabpy.linalg.factorize import factorize
from linstabpy.solvers import eigenvalue as eigen_core
from linstabpy.stability import eigenvalues as stability_eigenvalues


def test_linalg_backend_registry_imports_and_availability() -> None:
    availability = available()

    assert Backend.MUMPS in availability
    assert availability[Backend.MUMPS] == backends.factor_backend_available("mumps")
    ensure(None)


def test_linalg_select_matches_solver_backend_resolution(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_FACTOR_BACKEND", "superlu")
    monkeypatch.setenv("LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE", "mumps")
    cfgs = [
        None,
        {"factor_backend": "petsc"},
        {"factor_backend": "MUMPS"},
        {"factor_backend": "strumpack"},
        {"factor_backend": "superlu_dist"},
        {"factor_backend": "cudss"},
    ]

    for cfg in cfgs:
        resolved = backends.resolve_factor_backend(
            cfg,
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
            default="mumps",
        )
        selected = select(
            cfg,
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
            default=Backend.MUMPS,
        )
        assert (None if selected is None else str(selected)) == resolved


def test_resolve_factor_backend_precedence(monkeypatch) -> None:
    monkeypatch.setenv("LINSTABPY_FACTOR_BACKEND", "superlu")
    monkeypatch.setenv("LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE", "mumps")

    assert backends.resolve_factor_backend(None) == "superlu"
    assert (
        backends.resolve_factor_backend(
            None,
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
        )
        == "mumps"
    )
    assert (
        backends.resolve_factor_backend(
            {"factor_backend": "petsc"},
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
            default="mumps",
        )
        is None
    )
    assert (
        backends.resolve_factor_backend(
            {"factor_backend": "MUMPS"},
            env_var="LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
        )
        == "mumps"
    )


def test_resolve_factor_precision_precedence_and_config_surface(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_FACTOR_PRECISION", raising=False)

    assert backends.resolve_factor_precision(None) == "double"
    assert backends.resolve_factor_precision({"factor_precision": "single"}) == "single"

    monkeypatch.setenv("LINSTABPY_FACTOR_PRECISION", "double")
    assert backends.resolve_factor_precision({"factor_precision": "single"}) == "double"

    cfg = Config.load({"solver": {"linsolve": {"factor_precision": "single"}}}, env={})
    assert cfg.linsolve.factor_precision == "single"
    assert Config.load({}, env={}).linsolve.factor_precision == "double"


class _FakeOptions:
    def __init__(self) -> None:
        self.set_calls: list[tuple[str, str]] = []
        self.del_calls: list[str] = []

    def setValue(self, key: str, value: str) -> None:  # noqa: N802
        self.set_calls.append((key, value))

    def delValue(self, key: str) -> None:  # noqa: N802
        self.del_calls.append(key)


class _FakePC:
    def __init__(self) -> None:
        self.from_options_calls = 0
        self.factor_solver_type: str | None = None

    def setFromOptions(self) -> None:  # noqa: N802
        self.from_options_calls += 1

    def setFactorSolverType(self, backend: str) -> None:  # noqa: N802
        self.factor_solver_type = backend


def _apply_with_fake_options(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    *,
    factor_precision: str | None = None,
    mumps_blr: bool | None = None,
    mumps_blr_tol: float | None = None,
) -> tuple[_FakeOptions, _FakePC]:
    options = _FakeOptions()
    pc = _FakePC()
    monkeypatch.setattr(backends, "ensure_factor_backend_available", lambda backend: None)
    monkeypatch.setattr(backends.PETSc, "Options", lambda: options)
    backends.apply_factor_backend(
        pc,
        backend,
        factor_precision=factor_precision,
        mumps_blr=mumps_blr,
        mumps_blr_tol=mumps_blr_tol,
    )
    return options, pc


def test_factor_precision_double_is_inert_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_FACTOR_PRECISION", raising=False)

    absent_options, absent_pc = _apply_with_fake_options(monkeypatch, "mumps")
    explicit_options, explicit_pc = _apply_with_fake_options(
        monkeypatch, "mumps", factor_precision="double"
    )

    assert absent_options.set_calls == explicit_options.set_calls == [
        ("pc_factor_mat_solver_type", "mumps")
    ]
    assert absent_options.del_calls == explicit_options.del_calls == [
        "mat_mumps_icntl_35",
        "mat_mumps_cntl_7",
        "pc_factor_mat_solver_type",
    ]
    assert absent_pc.factor_solver_type == explicit_pc.factor_solver_type == "mumps"


def test_factor_precision_single_superlu_dist_sets_pc_precision(monkeypatch) -> None:
    options, pc = _apply_with_fake_options(
        monkeypatch, "superlu_dist", factor_precision="single"
    )

    assert options.set_calls == [
        ("pc_factor_mat_solver_type", "superlu_dist"),
        ("pc_precision", "single"),
    ]
    assert options.del_calls == ["pc_precision", "pc_factor_mat_solver_type"]
    assert pc.factor_solver_type == "superlu_dist"


def test_mumps_blr_default_off_keeps_options_db_byte_identical(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_MUMPS_BLR", raising=False)
    monkeypatch.delenv("LINSTABPY_MUMPS_BLR_TOL", raising=False)

    absent_options, absent_pc = _apply_with_fake_options(monkeypatch, "mumps")
    explicit_off_options, explicit_off_pc = _apply_with_fake_options(
        monkeypatch, "mumps", mumps_blr=False, mumps_blr_tol=1.0e-2
    )

    assert absent_options.set_calls == explicit_off_options.set_calls == [
        ("pc_factor_mat_solver_type", "mumps")
    ]
    # The off path clears any leftover BLR options (no-op when absent) so a
    # prior BLR-enabled setup cannot leak into this one, then removes the
    # solver-type key as before. Nothing is ever *set* besides the backend.
    assert absent_options.del_calls == explicit_off_options.del_calls == [
        "mat_mumps_icntl_35",
        "mat_mumps_cntl_7",
        "pc_factor_mat_solver_type",
    ]
    assert absent_pc.factor_solver_type == explicit_off_pc.factor_solver_type == "mumps"


def test_mumps_blr_enabled_sets_icntl_35_and_cntl_7(monkeypatch) -> None:
    options, pc = _apply_with_fake_options(
        monkeypatch, "mumps", mumps_blr=True, mumps_blr_tol=5.0e-3
    )

    assert options.set_calls == [
        ("pc_factor_mat_solver_type", "mumps"),
        ("mat_mumps_icntl_35", "1"),
        ("mat_mumps_cntl_7", "0.005"),
    ]
    assert options.del_calls == ["pc_factor_mat_solver_type"]
    assert pc.factor_solver_type == "mumps"


@pytest.mark.parametrize("backend", ["petsc", None, "strumpack", "superlu_dist", "cudss"])
def test_mumps_blr_rejects_non_mumps_backend(monkeypatch, backend) -> None:
    monkeypatch.setattr(backends, "ensure_factor_backend_available", lambda backend: None)

    with pytest.raises(ValueError) as excinfo:
        backends.apply_factor_backend(_FakePC(), backend, mumps_blr=True)

    message = str(excinfo.value)
    assert repr(backend) in message
    assert "LINSTABPY_MUMPS_BLR" in message
    assert "mumps" in message


def test_mumps_blr_env_beats_jsonc_and_default(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_MUMPS_BLR", raising=False)
    monkeypatch.delenv("LINSTABPY_MUMPS_BLR_TOL", raising=False)

    assert backends.resolve_mumps_blr(None) is False
    assert backends.resolve_mumps_blr_tol(None) == pytest.approx(1.0e-4)
    assert backends.resolve_mumps_blr({"mumps_blr": True}) is True
    assert backends.resolve_mumps_blr_tol({"mumps_blr_tol": 2.5e-3}) == pytest.approx(2.5e-3)

    monkeypatch.setenv("LINSTABPY_MUMPS_BLR", "off")
    monkeypatch.setenv("LINSTABPY_MUMPS_BLR_TOL", "3e-2")
    assert backends.resolve_mumps_blr({"mumps_blr": True}) is False
    assert backends.resolve_mumps_blr_tol({"mumps_blr_tol": 2.5e-3}) == pytest.approx(3.0e-2)

    cfg = Config.load(
        {"solver": {"linsolve": {"mumps_blr": False, "mumps_blr_tol": 1.0e-2}}},
        env={"LINSTABPY_MUMPS_BLR": "yes", "LINSTABPY_MUMPS_BLR_TOL": "4e-3"},
    )
    assert cfg.linsolve.mumps_blr is True
    assert cfg.linsolve.mumps_blr_tol == pytest.approx(4.0e-3)


@pytest.mark.parametrize("bad_tol", [-1.0, 0.0, "nan", "inf", "not-a-number"])
def test_mumps_blr_bad_tolerance_raises_typed_error(bad_tol) -> None:
    with pytest.raises(ValueError) as excinfo:
        Config.load({"solver": {"linsolve": {"mumps_blr_tol": bad_tol}}}, env={})

    message = str(excinfo.value)
    assert "LINSTABPY_MUMPS_BLR_TOL" in message
    assert "positive finite" in message


@pytest.mark.parametrize("backend", ["mumps", "superlu", "strumpack", "cudss"])
def test_factor_precision_single_rejects_non_superlu_dist(monkeypatch, backend) -> None:
    monkeypatch.setattr(backends, "ensure_factor_backend_available", lambda backend: None)

    with pytest.raises(ValueError) as excinfo:
        backends.apply_factor_backend(_FakePC(), backend, factor_precision="single")

    message = str(excinfo.value)
    assert backend in message
    assert "LINSTABPY_FACTOR_PRECISION" in message
    assert "superlu_dist" in message


def _mumps_blr_probe_ksp(monkeypatch, *, enabled: bool, tol: float = 2.5e-3):  # noqa: ANN001
    if not backends.factor_backend_available("mumps"):
        pytest.skip("SKIPPED: missing MUMPS factor backend")

    monkeypatch.setenv("LINSTABPY_MUMPS_BLR", "1" if enabled else "0")
    monkeypatch.setenv("LINSTABPY_MUMPS_BLR_TOL", f"{tol:g}")
    A = PETSc.Mat().createAIJ(size=(2, 2), comm=PETSc.COMM_SELF)
    values = {
        (0, 0): 2.0 + 0.5j,
        (0, 1): 0.1 - 0.2j,
        (1, 0): 0.3 + 0.1j,
        (1, 1): 3.0 - 0.25j,
    }
    for (row, col), value in values.items():
        A.setValue(row, col, value)
    A.assemble()
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    try:
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        backends.apply_factor_backend(pc, "mumps")
        ksp.setUp()
        return ksp, A
    except Exception:
        ksp.destroy()
        A.destroy()
        raise


def test_mumps_blr_real_wiring_reaches_mumps_after_ksp_setup(monkeypatch) -> None:
    tol = 2.5e-3
    ksp, A = _mumps_blr_probe_ksp(monkeypatch, enabled=True, tol=tol)
    try:
        factor = ksp.getPC().getFactorMatrix()
        assert factor.getMumpsIcntl(35) == 1
        assert factor.getMumpsCntl(7) == pytest.approx(tol)
    finally:
        PETSc.Options().delValue("mat_mumps_cntl_7")
        PETSc.Options().delValue("mat_mumps_icntl_35")
        ksp.destroy()
        A.destroy()


def test_mumps_blr_default_off_real_wiring_leaves_mumps_blr_disabled(monkeypatch) -> None:
    ksp, A = _mumps_blr_probe_ksp(monkeypatch, enabled=False)
    try:
        assert ksp.getPC().getFactorMatrix().getMumpsIcntl(35) == 0
    finally:
        PETSc.Options().delValue("mat_mumps_cntl_7")
        PETSc.Options().delValue("mat_mumps_icntl_35")
        ksp.destroy()
        A.destroy()


def test_mumps_blr_off_after_enabled_setup_is_not_sticky(monkeypatch) -> None:
    ksp_on, A_on = _mumps_blr_probe_ksp(monkeypatch, enabled=True, tol=2.5e-3)
    try:
        assert ksp_on.getPC().getFactorMatrix().getMumpsIcntl(35) == 1
        ksp_off, A_off = _mumps_blr_probe_ksp(monkeypatch, enabled=False)
        try:
            assert ksp_off.getPC().getFactorMatrix().getMumpsIcntl(35) == 0
        finally:
            ksp_off.destroy()
            A_off.destroy()
    finally:
        PETSc.Options().delValue("mat_mumps_cntl_7")
        PETSc.Options().delValue("mat_mumps_icntl_35")
        ksp_on.destroy()
        A_on.destroy()


def test_mumps_probe_and_apply_configure_lu_without_cuda_imports() -> None:
    before_cuda_modules = {name for name in sys.modules if "cuda" in name.lower()}

    assert backends.resolve_factor_backend(None) == "mumps"
    assert backends.factor_backend_available("mumps")

    A = PETSc.Mat().createAIJ(size=(1, 1), comm=PETSc.COMM_SELF)
    A.setValue(0, 0, 1.0 + 0.0j)
    A.assemble()
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    try:
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        selected = backends.apply_factor_backend(pc, "mumps")
        ksp.setUp()

        assert selected == "mumps"
        assert ksp.getType() == PETSc.KSP.Type.PREONLY
        assert pc.getType() == PETSc.PC.Type.LU
        assert pc.getFactorSolverType() == "mumps"
    finally:
        ksp.destroy()
        A.destroy()

    after_cuda_modules = {name for name in sys.modules if "cuda" in name.lower()}
    assert after_cuda_modules == before_cuda_modules


def test_absent_backend_probe_and_error_message() -> None:
    missing = "linstabpy_absent_factor_backend"

    assert not backends.factor_backend_available(missing)
    with pytest.raises(RuntimeError) as excinfo:
        backends.ensure_factor_backend_available(missing)

    message = str(excinfo.value)
    assert missing in message
    assert "PETSc" in message
    assert "Rebuild PETSc/petsc4py" in message
    assert "docs/COMPUTE_BACKENDS.md" in message


@pytest.mark.parametrize("backend", ["strumpack", "superlu_dist", "cudss"])
def test_future_backend_knob_fails_loudly_on_cpu_only_build(monkeypatch, backend) -> None:
    monkeypatch.setenv("LINSTABPY_FACTOR_BACKEND", backend)
    operator = sp.eye(2, format="csr", dtype=complex)

    assert backends.resolve_factor_backend(None) == backend
    assert not backends.factor_backend_available(backend)
    with pytest.raises(RuntimeError) as excinfo:
        factorize(operator, solver_cfg={"linsolve": {"type": "direct"}})

    message = str(excinfo.value)
    assert f"PETSc factor backend '{backend}' is not available" in message
    assert "Rebuild PETSc/petsc4py" in message
    assert "docs/COMPUTE_BACKENDS.md" in message


def test_direct_linsolve_uses_mumps_backend(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_FACTOR_BACKEND", raising=False)
    operator = sp.eye(2, format="csr", dtype=complex)

    solver = factorize(operator, solver_cfg={"linsolve": {"type": "direct"}})
    try:
        assert solver.backend == "MUMPS"
        assert solver._ksp is not None
        assert solver._ksp.getType() == PETSc.KSP.Type.PREONLY
        assert solver._ksp.getPC().getType() == PETSc.PC.Type.LU
        assert solver._ksp.getPC().getFactorSolverType() == "mumps"
        np.testing.assert_allclose(solver.solve(np.array([1.0, 2.0])), [1.0, 2.0])
    finally:
        solver.destroy()


def _tiny_shifted_ksp(st_solver: str, backend: str | None = None) -> tuple[PETSc.KSP, PETSc.Mat]:
    A = PETSc.Mat().createAIJ(size=(3, 3), comm=PETSc.COMM_SELF)
    values = {
        (0, 0): 4.0 + 0.5j,
        (0, 1): 0.25 - 0.1j,
        (1, 0): 0.5 + 0.2j,
        (1, 1): 3.0 - 0.25j,
        (1, 2): -0.3j,
        (2, 1): 0.1 + 0.4j,
        (2, 2): 2.0 + 0.75j,
    }
    for (row, col), value in values.items():
        A.setValue(row, col, value)
    A.assemble()
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(A)
    ecfg: dict[str, object] = {
        "st_solver": st_solver,
        "inexact": {"ksp_rtol": 1.0e-12, "ksp_atol": 1.0e-14, "ksp_max_it": 80},
    }
    if backend is not None:
        ecfg["factor_backend"] = backend
    cfg = Config.load({"solver": {"eigen": ecfg}})
    eigen_core._configure_st_for_eigenvalue(st_solver, ecfg, ksp, ksp.getPC(), None, cfg=cfg)
    ksp.setUp()
    return ksp, A


def _solve_ksp(ksp: PETSc.KSP, values: list[complex]) -> np.ndarray:
    b = ksp.getOperators()[0].createVecRight()
    x = ksp.getOperators()[0].createVecRight()
    try:
        for row, value in enumerate(values):
            b.setValue(row, value)
        b.assemble()
        ksp.solve(b, x)
        return x.getArray(readonly=True).copy()
    finally:
        x.destroy()
        b.destroy()


def test_fgmres_lu_path_with_mumps_matches_plain_mumps_direct() -> None:
    if not backends.factor_backend_available("mumps"):
        pytest.skip("SKIPPED: missing MUMPS factor backend")

    fgmres_ksp, fgmres_A = _tiny_shifted_ksp("fgmres_lu", "mumps")
    direct_ksp, direct_A = _tiny_shifted_ksp("mumps", "mumps")
    try:
        assert fgmres_ksp.getType() == PETSc.KSP.Type.FGMRES
        assert fgmres_ksp.getPC().getType() == PETSc.PC.Type.LU
        assert fgmres_ksp.getPC().getFactorSolverType() == "mumps"

        rhs = [1.0 + 0.25j, -0.5 + 0.75j, 0.125 - 0.5j]
        fgmres_x = _solve_ksp(fgmres_ksp, rhs)
        direct_x = _solve_ksp(direct_ksp, rhs)
        np.testing.assert_allclose(fgmres_x, direct_x, rtol=1.0e-10, atol=1.0e-12)

        probe = backends.shifted_ksp_probe_residual(fgmres_ksp)
        assert float(probe["residual_rel"]) <= 1.0e-8
    finally:
        fgmres_ksp.destroy()
        fgmres_A.destroy()
        direct_ksp.destroy()
        direct_A.destroy()


def test_strumpack_fgmres_path_matches_plain_mumps_direct_when_available() -> None:
    if not backends.factor_backend_available("strumpack"):
        pytest.skip("SKIPPED: missing STRUMPACK factor backend")
    if not backends.factor_backend_available("mumps"):
        pytest.skip("SKIPPED: missing MUMPS factor backend")

    strumpack_ksp, strumpack_A = _tiny_shifted_ksp("strumpack_fgmres")
    direct_ksp, direct_A = _tiny_shifted_ksp("mumps", "mumps")
    try:
        assert strumpack_ksp.getType() == PETSc.KSP.Type.FGMRES
        assert strumpack_ksp.getPC().getType() == PETSc.PC.Type.LU
        assert strumpack_ksp.getPC().getFactorSolverType() == "strumpack"

        rhs = [1.0 + 0.25j, -0.5 + 0.75j, 0.125 - 0.5j]
        strumpack_x = _solve_ksp(strumpack_ksp, rhs)
        direct_x = _solve_ksp(direct_ksp, rhs)
        np.testing.assert_allclose(strumpack_x, direct_x, rtol=1.0e-8, atol=1.0e-10)
    finally:
        strumpack_ksp.destroy()
        strumpack_A.destroy()
        direct_ksp.destroy()
        direct_A.destroy()


def test_strumpack_fgmres_missing_backend_fails_loudly() -> None:
    if backends.factor_backend_available("strumpack"):
        pytest.skip("STRUMPACK factor backend is available in this PETSc build")

    with pytest.raises(RuntimeError) as excinfo:
        ksp, A = _tiny_shifted_ksp("strumpack_fgmres")
        ksp.destroy()
        A.destroy()

    message = str(excinfo.value)
    assert "PETSc factor backend 'strumpack' is not available" in message
    assert "Rebuild PETSc/petsc4py" in message
    assert "docs/COMPUTE_BACKENDS.md" in message


@pytest.mark.parametrize("st_solver", ["fgmres_lu", "strumpack_fgmres"])
def test_fgmres_selectors_honor_eigen_factor_solver_env_override(
    monkeypatch: pytest.MonkeyPatch, st_solver: str
) -> None:
    monkeypatch.setenv("LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE", "superlu_dist")
    cfg = Config.load({"solver": {"eigen": {"st_solver": st_solver}}})
    assert cfg.eigen.st_factor_backend == "superlu_dist"


def test_strumpack_fgmres_cp5_probe_policy_is_active() -> None:
    assert backends.should_probe_backend("strumpack", st_solver="strumpack_fgmres")
    # The fgmres families are inexact outer solves: even the default MUMPS
    # factor backend must be probed there, while plain direct MUMPS stays exempt.
    assert backends.should_probe_backend("mumps", st_solver="fgmres_lu")
    assert backends.should_probe_backend("mumps", st_solver="strumpack_fgmres")
    assert not backends.should_probe_backend("mumps")
    # An explicit probe_cfg switch still wins over the family default.
    assert not backends.should_probe_backend(
        "mumps", {"enabled": False}, st_solver="fgmres_lu"
    )


def _tiny_stability_operator() -> np.ndarray:
    return np.array(
        [
            [1.0 + 0.0j, 0.2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 2.0 + 0.0j, 0.1 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j, 0.3 + 0.0j],
            [0.1 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 4.0 + 0.0j],
        ],
        dtype=complex,
    )


def test_mumps_shift_invert_probe_real_solve_is_below_threshold(monkeypatch) -> None:
    if not backends.factor_backend_available("mumps"):
        pytest.skip("SKIPPED: missing MUMPS factor backend")

    real_probe = eigen_core.shifted_ksp_probe_residual
    calls: list[dict[str, float | int | str | None]] = []

    def force_probe(resolved_backend, probe_cfg=None, *, st_solver=None):  # noqa: ANN001
        assert resolved_backend == "mumps"
        assert st_solver == "mumps"
        return True

    def spy_probe(ksp):  # noqa: ANN001
        probe = real_probe(ksp)
        calls.append(probe)
        return probe

    monkeypatch.setattr(eigen_core, "should_probe_backend", force_probe)
    monkeypatch.setattr(eigen_core, "shifted_ksp_probe_residual", spy_probe)

    omega, _, info = stability_eigenvalues(
        _tiny_stability_operator(),
        n_modes=1,
        target=2.2 + 0.0j,
        ncv=3,
        solver_cfg={"eigen": {"st_solver": "mumps"}},
        return_info=True,
    )

    assert omega.shape == (1,)
    assert len(calls) == 1
    assert calls[0]["ksp_type"] == PETSc.KSP.Type.PREONLY
    assert calls[0]["pc_type"] == PETSc.PC.Type.LU
    assert calls[0]["pc_factor_solver_type"] == "mumps"
    assert float(calls[0]["residual_rel"]) <= 1.0e-8
    assert info["backend_probe_ran"] is True
    assert float(info["backend_probe_residual_rel"]) == pytest.approx(
        float(calls[0]["residual_rel"])
    )


def test_fgmres_lu_backend_probe_raises_on_degraded_real_solve(monkeypatch) -> None:
    if not backends.factor_backend_available("mumps"):
        pytest.skip("SKIPPED: missing MUMPS factor backend")

    def degrade_factor_backend_to_unpreconditioned_solve(pc, backend, **kwargs):  # noqa: ANN001, ANN003
        assert backend == "mumps"
        pc.setType(PETSc.PC.Type.NONE)
        return backend

    monkeypatch.setattr(
        eigen_core,
        "apply_factor_backend",
        degrade_factor_backend_to_unpreconditioned_solve,
    )

    real_probe = eigen_core.shifted_ksp_probe_residual
    probe_calls: list[dict[str, float | int | str | None]] = []

    def spy_probe(ksp):  # noqa: ANN001
        probe = real_probe(ksp)
        probe_calls.append(probe)
        return probe

    monkeypatch.setattr(eigen_core, "shifted_ksp_probe_residual", spy_probe)

    with pytest.raises(RuntimeError) as excinfo:
        stability_eigenvalues(
            _tiny_stability_operator(),
            n_modes=1,
            target=2.2 + 0.0j,
            ncv=3,
            solver_cfg={
                "eigen": {
                    "st_solver": "fgmres_lu",
                    "inexact": {
                        "ksp_rtol": 0.999,
                        "ksp_atol": 0.0,
                        "ksp_max_it": 1,
                    },
                    "backend_probe": {"residual_rel_max": 1.0e-8},
                }
            },
        )

    # The raise must come from the real probe measuring the degraded solve.
    assert len(probe_calls) == 1
    assert float(probe_calls[0]["residual_rel"]) > 1.0e-8

    message = str(excinfo.value)
    assert "factor backend 'mumps' failed accuracy probe" in message
    assert "||Kx-b||/||b||" in message
    assert re.search(r"\|\|Kx-b\|\|/\|\|b\|\| = [0-9.]+e[+-]\d+ >", message)
    assert "1.000e-08" in message


def test_strumpack_fgmres_invokes_backend_probe(monkeypatch) -> None:
    monkeypatch.delenv("LINSTABPY_FACTOR_BACKEND", raising=False)
    calls: list[str] = []

    def fake_apply_factor_backend(pc, backend, **kwargs):  # noqa: ANN001, ANN003
        calls.append(f"apply:{backend}")
        return backend

    def fake_probe(ksp):  # noqa: ANN001
        calls.append(f"probe:{ksp.getType()}")
        return {
            "residual_abs": 0.0,
            "residual_rel": 0.0,
            "ksp_iterations": 1,
            "ksp_reason": 1,
            "ksp_type": ksp.getType(),
            "pc_type": ksp.getPC().getType(),
            "pc_factor_solver_type": "strumpack",
        }

    monkeypatch.setattr(eigen_core, "apply_factor_backend", fake_apply_factor_backend)
    monkeypatch.setattr(eigen_core, "shifted_ksp_probe_residual", fake_probe)
    operator = np.diag([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j])

    omega, _ = stability_eigenvalues(
        operator,
        n_modes=1,
        target=2.2 + 0.0j,
        solver_cfg={"eigen": {"st_solver": "strumpack_fgmres"}},
    )

    assert omega.shape == (1,)
    assert calls == ["apply:strumpack", "probe:fgmres"]


def test_iterative_linsolve_does_not_enter_factor_backend_seam(monkeypatch) -> None:
    def fail_apply(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("direct factor seam should not run for iterative linsolve")

    factorize_module = importlib.import_module("linstabpy.linalg.factorize")
    monkeypatch.setattr(factorize_module, "apply_factor_backend", fail_apply)
    operator = sp.eye(2, format="csr", dtype=complex)

    solver = factorize(
        operator,
        solver_cfg={
            "linsolve": {
                "type": "iterative",
                "ksp_type": "gmres",
                "pc_type": "none",
                "ksp_max_it": 10,
            }
        },
    )
    try:
        assert solver.backend == "PETSc-iterative"
        assert solver._ksp is not None
        assert solver._ksp.getType() == "gmres"
        assert solver._ksp.getPC().getType() == "none"
    finally:
        solver.destroy()
