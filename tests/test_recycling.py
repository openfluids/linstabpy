# SPDX-License-Identifier: GPL-3.0-only
"""GCRO-DR-style fixed-subspace recycling tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linstabpy.linalg.backends import factor_backend_available
from linstabpy.linalg.factorize import factorize
from linstabpy.solvers.router import stable_sweep


@dataclass
class _ScipyFactorization:
    matrix: sp.csc_matrix
    lu: spla.SuperLU
    metrics: dict[str, float | int]

    def solve(self, rhs):  # noqa: ANN001, ANN201
        return self.lu.solve(np.asarray(rhs, dtype=complex))

    def solve_adjoint(self, rhs):  # noqa: ANN001, ANN201
        return self.lu.solve(np.asarray(rhs, dtype=complex), trans="H")

    def solve_block(self, rhs_block: np.ndarray) -> np.ndarray:
        block = np.asarray(rhs_block, dtype=complex)
        return np.column_stack([self.solve(block[:, j]) for j in range(block.shape[1])])

    def destroy(self) -> None:
        return None


class _FactorizeRecorder:
    def __init__(self) -> None:
        self.factors: list[_ScipyFactorization] = []

    def __call__(self, operator, **_kwargs):  # noqa: ANN001, ANN003, ANN201
        matrix = sp.csc_matrix(operator, dtype=complex)
        factor = _ScipyFactorization(
            matrix=matrix,
            lu=spla.splu(matrix),
            metrics={
                "refactor_count": 0,
                "refactor_seconds": 0.0,
                "solve_block_calls": 0,
                "solve_block_rhs": 0,
                "solve_block_seconds": 0.0,
            },
        )
        self.factors.append(factor)
        return factor


def _operator(n: int = 6) -> sp.csc_matrix:
    diagonal = 2.7 + 0.22 * np.arange(n) + 1j * (0.17 - 0.025 * np.arange(n))
    lower = 0.045 - 0.025j + 0.003 * np.arange(n - 1)
    upper = -0.035 + 0.02j - 0.002 * np.arange(n - 1)
    far = np.full(n - 2, 0.012 - 0.006j, dtype=complex)
    return sp.diags(
        [far, lower, diagonal, upper, 0.4 * far], [-2, -1, 0, 1, 2], format="csc", dtype=complex
    )


def _cfg(n: int, *, recycling: bool = False) -> dict:
    eye = np.eye(n, dtype=complex)
    cfg = {
        "sigma_tol": 1.0e-6,
        "inv_norm_est": 1.0,
        "actual_solve_tol": 1.0e-10,
        "fgmres_rtol": 1.0e-11,
        "fgmres_maxiter": n,
        "n_modes": 1,
        "oversample": n,
        "seeds": (21, 22, 23),
        "probes_a": (eye[0], eye[n // 2]),
        "probes_b": (eye[1], eye[-1]),
        "W": np.ones(n, dtype=complex),
        "invW": np.ones(n, dtype=complex),
        "linearity_tol": 1.0e-8,
        "seed_gain_tol": 1.0e-6,
        "seed_mode_tol": 1.0e-5,
        "reuse_symbolic": False,
    }
    if recycling:
        cfg["recycling"] = {"enabled": True, "max_dim": 3, "orth_tol": 1.0e-12}
    return cfg


def _direct_mumps_gain_map(L0: sp.spmatrix, omegas: np.ndarray) -> np.ndarray:
    solver_cfg = stable_sweep._with_direct_factor_backend(  # noqa: SLF001
        {"solver": {"resolvent": {"type": "direct", "factor_backend": "mumps"}}},
        "mumps",
    )
    factorize_cfg = stable_sweep._factorize_cfg_with_backend(None, solver_cfg, "mumps")  # noqa: SLF001
    gains: list[float] = []
    eye = np.eye(L0.shape[0], dtype=complex)
    for omega in omegas:
        factor = factorize(
            stable_sweep.shifted_operator(L0, omega),
            env_prefix="LINSTABPY_RESOLVENT",
            solver_cfg=solver_cfg,
            cfg=factorize_cfg,
        )
        try:
            resolvent = np.column_stack([factor.solve(eye[:, j]) for j in range(eye.shape[1])])
        finally:
            factor.destroy()
        gains.append(float(np.linalg.svd(resolvent, compute_uv=False)[0]))
    return np.asarray(gains)


def test_recycling_is_opt_in_and_off_path_records_only_unrecycled_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    L0 = _operator()
    omegas = np.array([0.96, 1.0, 1.04], dtype=complex)
    anchors = np.array([1.0], dtype=complex)
    cfg = _cfg(L0.shape[0])
    recorder = _FactorizeRecorder()
    monkeypatch.setattr(stable_sweep, "factorize", recorder)

    result = stable_sweep.sweep_stable_regime(L0, omegas, anchors, cfg)

    assert result.metrics["recycling_requested"] is False
    assert result.metrics["matvecs_total"] > 0
    assert result.metrics["matvecs_total_unrecycled"] == result.metrics["matvecs_total"]
    assert result.metrics["matvecs_total_actual"] == result.metrics["matvecs_total"]
    assert result.metrics["matvecs_total_recycled"] == 0
    assert (
        result.metrics["gmres_iterations_total_actual"] == result.metrics["gmres_iterations_total"]
    )
    assert result.metrics["gmres_iterations_total_recycled"] == 0
    assert len(recorder.factors) == anchors.size
    assert all(report.passed for report in result.certification_reports)


def test_recycled_sweep_matches_s1_and_direct_mumps_oracle_with_linearity_and_counts() -> None:
    if not factor_backend_available("mumps"):
        pytest.skip("SKIPPED: mumps backend unavailable")

    L0 = _operator()
    omegas = np.array([0.94, 0.98, 1.02, 1.06], dtype=complex)
    anchors = np.array([1.0], dtype=complex)
    baseline_cfg = _cfg(L0.shape[0])
    recycling_cfg = _cfg(L0.shape[0], recycling=True)

    baseline = stable_sweep.sweep_stable_regime(L0, omegas, anchors, baseline_cfg)
    recycled = stable_sweep.sweep_stable_regime(L0, omegas, anchors, recycling_cfg)
    direct = _direct_mumps_gain_map(L0, omegas)

    np.testing.assert_allclose(
        recycled.gain,
        baseline.gain,
        rtol=recycling_cfg["sigma_tol"],
        atol=recycling_cfg["sigma_tol"],
    )
    np.testing.assert_allclose(
        recycled.gain, direct, rtol=recycling_cfg["sigma_tol"], atol=recycling_cfg["sigma_tol"]
    )
    for report in recycled.certification_reports:
        linearity = report.checks["deterministic_tol_linearity"]
        assert linearity.passed
        assert linearity.measured <= recycling_cfg["linearity_tol"]

    metrics = recycled.metrics
    assert metrics["recycling_requested"] is True
    assert metrics["recycle_fixed_per_pass"] is True
    assert metrics["recycle_forward_rank_max"] > 0
    assert metrics["recycle_adjoint_rank_max"] > 0
    assert metrics["recycle_setup_matvecs_total"] > 0
    assert metrics["matvecs_total"] == metrics["matvecs_total_unrecycled"] > 0
    assert metrics["matvecs_total_recycled"] > 0
    assert metrics["matvecs_total_actual"] == metrics["matvecs_total_recycled"]
    assert metrics["gmres_iterations_total"] == metrics["gmres_iterations_total_unrecycled"] > 0
    assert metrics["gmres_iterations_total_recycled"] > 0
    assert metrics["gmres_iterations_total_actual"] == metrics["gmres_iterations_total_recycled"]
    assert (
        metrics["matvecs_delta_recycled_minus_baseline"]
        == metrics["matvecs_total_recycled"] - metrics["matvecs_total"]
    )
    assert metrics["gmres_iterations_delta_recycled_minus_baseline"] == (
        metrics["gmres_iterations_total_recycled"] - metrics["gmres_iterations_total"]
    )
