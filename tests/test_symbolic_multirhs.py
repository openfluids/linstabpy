# SPDX-License-Identifier: GPL-3.0-only

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linstabpy.linalg.factorize import factorize
from linstabpy.linalg.shifted import shifted_operator
from linstabpy.solvers.router import stable_sweep


def _operator() -> sp.csc_matrix:
    diagonal = np.array([2.1 + 0.1j, 2.8 - 0.2j, 3.5 + 0.3j, 4.4 - 0.1j, 5.2 + 0.2j])
    lower = np.array([0.02 - 0.01j, -0.03 + 0.02j, 0.01 + 0.02j, -0.02j])
    return sp.diags([lower, diagonal, 0.4 * lower], [-1, 0, 1], format="csc", dtype=complex)


def _cfg(n: int) -> dict[str, object]:
    eye = np.eye(n, dtype=complex)
    return {
        "sigma_tol": 1.0e-7,
        "inv_norm_est": 1.0,
        "actual_solve_tol": 1.0e-10,
        "fgmres_rtol": 1.0e-11,
        "fgmres_maxiter": n,
        "n_modes": 1,
        "oversample": n,
        "seeds": (7, 8, 9),
        "probes_a": (eye[0], eye[-1]),
        "probes_b": (eye[1], eye[-2]),
        "W": np.ones(n, dtype=complex),
        "invW": np.ones(n, dtype=complex),
        "linearity_tol": 1.0e-9,
    }


class _ScipyFactorization:
    def __init__(self, operator: sp.spmatrix) -> None:
        self.matrix = sp.csc_matrix(operator, dtype=complex)
        self.lu = spla.splu(self.matrix)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(rhs, dtype=complex))

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(rhs, dtype=complex), trans="H")


class _RaisingRefactorization(_ScipyFactorization):
    def __init__(self, operator: sp.spmatrix) -> None:
        super().__init__(operator)
        self.refactorize_calls = 0

    def refactorize(self, _operator: sp.spmatrix) -> None:
        self.refactorize_calls += 1
        raise ValueError("forced refactorize failure")


def test_symbolic_once_matches_recompute_and_reuses_analysis() -> None:
    L0 = _operator()
    anchors = np.array([0.8, 1.5], dtype=complex)
    omegas = anchors.copy()
    cfg = _cfg(L0.shape[0])

    recompute = stable_sweep.sweep_stable_regime(
        L0, omegas, anchors, {**cfg, "reuse_symbolic": False}
    )
    symbolic = stable_sweep.sweep_stable_regime(
        L0, omegas, anchors, {**cfg, "reuse_symbolic": True}
    )

    np.testing.assert_allclose(
        symbolic.gain, recompute.gain, rtol=cfg["sigma_tol"], atol=cfg["sigma_tol"]
    )
    assert symbolic.metrics["symbolic_reuse_used"] is True
    assert symbolic.metrics["symbolic_reuse_fallback"] is False
    assert symbolic.metrics["analysis_reuse_count"] == anchors.size - 1
    assert recompute.metrics["analysis_reuse_count"] == 0
    assert symbolic.metrics["solve_block_rhs"] > symbolic.metrics["solve_block_calls"]


def test_factorization_solve_block_matches_single_rhs_loop() -> None:
    L0 = _operator()
    fac = factorize(shifted_operator(L0, 1.1), env_prefix="LINSTABPY_RESOLVENT")
    rhs_block = np.column_stack(
        [
            np.linspace(1.0, 2.0, L0.shape[0]) + 0.2j,
            np.linspace(-0.5, 0.5, L0.shape[0]) - 0.1j,
            np.eye(L0.shape[0], dtype=complex)[2],
        ]
    )
    try:
        block_solution = fac.solve_block(rhs_block)
        loop_solution = np.column_stack(
            [fac.solve(rhs_block[:, j]) for j in range(rhs_block.shape[1])]
        )
    finally:
        fac.destroy()

    np.testing.assert_allclose(block_solution, loop_solution, rtol=1.0e-12, atol=1.0e-12)
    assert block_solution.shape == rhs_block.shape


def test_refactorize_keeps_the_symbolic_factorisation() -> None:
    """refactorize copies values into the KSP's own Mat: MUMPS redoes the numeric phase only.

    Handing the KSP a new Mat redid the analysis at every refactorize.
    Savarino 450k DOF, c9, 2026-08-18: 90.0 s new Mat vs 74.9 s in place.
    PETSc's event log is the witness: MatLUFactorSym must not move,
    MatLUFactorNum must.
    """
    from petsc4py import PETSc

    if not PETSc.Log.isActive():
        PETSc.Log.begin()
    sym = PETSc.Log.Event("MatLUFactorSym")
    num = PETSc.Log.Event("MatLUFactorNum")
    L0 = _operator()
    fac = factorize(shifted_operator(L0, 0.8), env_prefix="LINSTABPY_RESOLVENT")
    try:
        sym0 = int(sym.getPerfInfo()["count"])
        num0 = int(num.getPerfInfo()["count"])
        assert num0 >= 1, (
            "PETSc event logging is not counting; the assertions below would be vacuous"
        )
        fac.refactorize(shifted_operator(L0, 1.5))
        assert int(sym.getPerfInfo()["count"]) == sym0, (
            "refactorize redid the symbolic factorization"
        )
        assert int(num.getPerfInfo()["count"]) == num0 + 1
        rhs = np.linspace(1.0, 2.0, L0.shape[0]).astype(complex)
        x = fac.solve(rhs)
        residual = np.linalg.norm(shifted_operator(L0, 1.5) @ x - rhs) / np.linalg.norm(rhs)
        assert residual <= 1.0e-10
    finally:
        fac.destroy()


def test_refactorize_rejects_different_nonzero_pattern() -> None:
    L0 = _operator()
    different_pattern = sp.diags(
        [np.diag(L0.toarray()), np.array([0.1 + 0.2j, -0.2j, 0.3 + 0.1j])],
        [0, 2],
        shape=L0.shape,
        format="csc",
        dtype=complex,
    )
    fac = factorize(L0, env_prefix="LINSTABPY_RESOLVENT")
    rhs = np.linspace(1.0, 2.0, L0.shape[0]).astype(complex)
    try:
        with pytest.raises(ValueError, match="identical nonzero pattern"):
            fac.refactorize(different_pattern)
        x = fac.solve(rhs)
        assert fac.residual(rhs, x) <= 1.0e-10
    finally:
        fac.destroy()


def test_sweep_falls_back_when_refactorize_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    L0 = _operator()
    anchors = np.array([0.8, 1.5], dtype=complex)
    omegas = anchors.copy()
    cfg = {**_cfg(L0.shape[0]), "use_block_solve": False}

    def normal_factorize(operator: sp.spmatrix, **_kwargs: object) -> _ScipyFactorization:
        return _ScipyFactorization(operator)

    monkeypatch.setattr(stable_sweep, "factorize", normal_factorize)
    recompute = stable_sweep.sweep_stable_regime(
        L0, omegas, anchors, {**cfg, "reuse_symbolic": False}
    )

    calls: list[_ScipyFactorization] = []

    def raising_first_factorize(operator: sp.spmatrix, **_kwargs: object) -> _ScipyFactorization:
        factor: _ScipyFactorization
        factor = _RaisingRefactorization(operator) if not calls else _ScipyFactorization(operator)
        calls.append(factor)
        return factor

    monkeypatch.setattr(stable_sweep, "factorize", raising_first_factorize)
    fallback = stable_sweep.sweep_stable_regime(
        L0, omegas, anchors, {**cfg, "reuse_symbolic": True}
    )

    np.testing.assert_allclose(
        fallback.gain, recompute.gain, rtol=cfg["sigma_tol"], atol=cfg["sigma_tol"]
    )
    assert fallback.metrics["symbolic_reuse_fallback"] is True
    assert fallback.metrics["symbolic_reuse_fallback_anchor_index"] == 1
    assert fallback.metrics["symbolic_reuse_used"] is False
    assert len(calls) == anchors.size
    assert isinstance(calls[0], _RaisingRefactorization)
    assert calls[0].refactorize_calls == 1
