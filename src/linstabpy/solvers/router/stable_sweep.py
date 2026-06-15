# SPDX-License-Identifier: GPL-3.0-only
"""Anchor-preconditioned stable-regime resolvent frequency sweeps."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from linstabpy.linalg.factorize import Factorization, factorize
from linstabpy.linalg.shifted import shifted_operator
from linstabpy.solvers.router.certification import CertificationReport, certified_rsvd
from linstabpy.utils.petsc_core import _is_petsc_matrix, vec_to_numpy
from linstabpy.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

Array = np.ndarray


@dataclass(frozen=True)
class SweepResult:
    """Results from an anchor-preconditioned frequency sweep."""

    omega: Array
    gain: Array
    iterations: Array
    residuals: Array
    anchor_indices: Array
    responses: Array
    forcings: Array
    certification_reports: tuple[CertificationReport, ...]
    metrics: dict[str, float | int | bool] = field(default_factory=dict)


def _cfg_get(cfg: Mapping[str, Any] | object | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_1d_complex(values: Sequence[complex] | Array, name: str) -> Array:
    arr = np.asarray(values, dtype=complex).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr


def _operator_size(operator: sp.spmatrix | PETSc.Mat) -> int:
    if _is_petsc_matrix(operator):
        rows, cols = operator.getSize()
        if rows != cols:
            raise ValueError("L0_beta must be square")
        return int(rows)
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("L0_beta must be square")
    return int(operator.shape[0])


def _matvec(operator: sp.spmatrix | PETSc.Mat, x: Array) -> Array:
    x_arr = np.asarray(x, dtype=complex).reshape(-1)
    if _is_petsc_matrix(operator):
        b = operator.createVecRight()
        y = operator.createVecLeft()
        r0, r1 = b.getOwnershipRange()
        if r1 > r0:
            rows = np.arange(r0, r1, dtype=PETSc.IntType)
            b.setValues(rows, x_arr[r0:r1], addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        try:
            operator.mult(b, y)
            return vec_to_numpy(y)
        finally:
            y.destroy()
            b.destroy()
    return np.asarray(operator @ x_arr, dtype=complex).reshape(-1)


def _adjoint_matvec(operator: sp.spmatrix | PETSc.Mat, x: Array) -> Array:
    x_arr = np.asarray(x, dtype=complex).reshape(-1)
    if _is_petsc_matrix(operator):
        b = operator.createVecLeft()
        y = operator.createVecRight()
        r0, r1 = b.getOwnershipRange()
        if r1 > r0:
            rows = np.arange(r0, r1, dtype=PETSc.IntType)
            b.setValues(rows, x_arr[r0:r1], addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        try:
            operator.multHermitianTranspose(b, y)
            return vec_to_numpy(y)
        finally:
            y.destroy()
            b.destroy()
    return np.asarray(operator.conjugate().transpose() @ x_arr, dtype=complex).reshape(-1)


def _relative_residual(operator: sp.spmatrix | PETSc.Mat, x: Array, rhs: Array, *, adjoint: bool = False) -> float:
    residual = (_adjoint_matvec(operator, x) if adjoint else _matvec(operator, x)) - rhs
    return float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))


def _right_preconditioned_gmres(
    matvec,
    precondition,
    rhs: Array,
    *,
    rtol: float,
    maxiter: int,
) -> tuple[Array, int, float]:
    """Small right-preconditioned GMRES used when sweeping stable shifts."""
    b = np.asarray(rhs, dtype=complex).reshape(-1)
    n = b.size
    beta = float(np.linalg.norm(b))
    if beta == 0.0:
        return np.zeros(n, dtype=complex), 0, 0.0

    q_vectors = [b / beta]
    z_vectors: list[Array] = []
    h = np.zeros((maxiter + 1, maxiter), dtype=complex)
    g = np.zeros(maxiter + 1, dtype=complex)
    g[0] = beta
    best_y = np.zeros(0, dtype=complex)
    best_residual = 1.0
    best_iter = 0

    for j in range(maxiter):
        z = np.asarray(precondition(q_vectors[j]), dtype=complex).reshape(-1)
        z_vectors.append(z)
        v = np.asarray(matvec(z), dtype=complex).reshape(-1)
        for i, q in enumerate(q_vectors):
            h[i, j] = np.vdot(q, v)
            v = v - h[i, j] * q
        h[j + 1, j] = np.linalg.norm(v)
        if h[j + 1, j] > 0.0 and j + 1 < maxiter:
            q_vectors.append(v / h[j + 1, j])

        y, *_ = np.linalg.lstsq(h[: j + 2, : j + 1], g[: j + 2], rcond=None)
        residual = float(np.linalg.norm(g[: j + 2] - h[: j + 2, : j + 1] @ y) / beta)
        best_y = y
        best_residual = residual
        best_iter = j + 1
        if residual <= rtol:
            break
        if h[j + 1, j] == 0.0:
            break

    x = np.zeros(n, dtype=complex)
    for coeff, z in zip(best_y, z_vectors, strict=False):
        x = x + coeff * z
    return x, best_iter, best_residual


def _nearest_anchor_indices(omega_block: Array, anchors: Array) -> Array:
    distances = np.abs(omega_block[:, None] - anchors[None, :])
    return np.argmin(distances, axis=1).astype(int)


def _default_probes(n: int) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    eye0 = np.zeros(n, dtype=complex)
    eye0[0] = 1.0
    ramp = np.arange(1, n + 1, dtype=float).astype(complex)
    ramp /= np.linalg.norm(ramp)
    alt = ((-1.0) ** np.arange(n)).astype(complex)
    alt /= np.linalg.norm(alt)
    return (eye0, ramp), (ramp, alt)


def _cert_cfg(
    cfg: Mapping[str, Any] | object | None,
    n: int,
    factor: Factorization,
    forward_residuals: Sequence[float],
    adjoint_residuals: Sequence[float],
) -> dict[str, Any]:
    probes_a, probes_b = _cfg_get(cfg, "probes_a"), _cfg_get(cfg, "probes_b")
    if probes_a is None or probes_b is None:
        probes_a, probes_b = _default_probes(n)
    actual_tol = float(_cfg_get(cfg, "actual_solve_tol", _cfg_get(cfg, "fgmres_rtol", 1.0e-10)))
    return {
        "sigma_tol": float(_cfg_get(cfg, "sigma_tol", 1.0e-8)),
        "inv_norm_est": float(_cfg_get(cfg, "inv_norm_est", 1.0)),
        "actual_solve_tol": actual_tol,
        "forward_tol": float(_cfg_get(cfg, "forward_tol", actual_tol)),
        "adjoint_tol": float(_cfg_get(cfg, "adjoint_tol", actual_tol)),
        "forward_residuals": tuple(float(r) for r in forward_residuals),
        "adjoint_residuals": tuple(float(r) for r in adjoint_residuals),
        "approx_solve": factor.solve,
        "approx_solve_adjoint": factor.solve_adjoint,
        "probes_a": probes_a,
        "probes_b": probes_b,
        "n_modes": int(_cfg_get(cfg, "n_modes", 1)),
        "oversample": int(_cfg_get(cfg, "oversample", 4)),
        "seeds": tuple(_cfg_get(cfg, "seeds", (0, 1, 2))),
        "safety": float(_cfg_get(cfg, "safety", 0.1)),
        # Inexact-solve defaults differ from the exact-LU certification defaults:
        # linearity defects of an iterative solve scale with the solve tolerance
        # (not machine eps), and forward/adjoint realized residuals come from
        # independent GMRES runs, so their relative mismatch is O(1) even when
        # both satisfy the per-solve bound (which stays enforced).  The anchor
        # factor itself is an exact LU, so its adjoint check keeps the strict
        # certification default.
        "linearity_tol": float(_cfg_get(cfg, "linearity_tol", 1.0e-10)),
        "adjoint_factor_tol": float(_cfg_get(cfg, "adjoint_factor_tol", 1.0e-12)),
        "consistency_tol": float(_cfg_get(cfg, "consistency_tol", 1.0)),
        "seed_gain_tol": float(_cfg_get(cfg, "seed_gain_tol", _cfg_get(cfg, "sigma_tol", 1.0e-8))),
        "seed_mode_tol": float(_cfg_get(cfg, "seed_mode_tol", 1.0e-6)),
    }


def sweep_stable_regime(
    L0_beta: sp.spmatrix | PETSc.Mat,
    omega_block: Sequence[complex] | Array,
    anchors: Sequence[complex] | Array,
    cfg: Mapping[str, Any] | object | None,
) -> SweepResult:
    """Sweep stable frequencies with nearest-anchor right-preconditioned GMRES.

    Anchor matrices are factored once through :func:`linstabpy.linalg.factorize`.
    Each frequency solve uses a self-contained right-preconditioned GMRES whose
    preconditioner is the nearest anchor factor.  Forward and adjoint actions
    reuse the same :class:`~linstabpy.linalg.factorize.Factorization` object;
    adjoint preconditioning calls ``Factorization.solve_adjoint``.
    """
    omegas = _as_1d_complex(omega_block, "omega_block")
    anchor_values = _as_1d_complex(anchors, "anchors")
    n = _operator_size(L0_beta)
    rtol = float(_cfg_get(cfg, "fgmres_rtol", _cfg_get(cfg, "actual_solve_tol", 1.0e-10)))
    maxiter = int(_cfg_get(cfg, "fgmres_maxiter", n))
    solver_cfg = _cfg_get(cfg, "solver_cfg", None)
    factorize_cfg = _cfg_get(cfg, "factorize_cfg", None)
    W = np.asarray(_cfg_get(cfg, "W", np.ones(n, dtype=complex)), dtype=complex).reshape(-1)
    invW = np.asarray(_cfg_get(cfg, "invW", np.ones(n, dtype=complex)), dtype=complex).reshape(-1)

    anchor_indices = _nearest_anchor_indices(omegas, anchor_values)
    use_symbolic_reuse = bool(_cfg_get(cfg, "reuse_symbolic", True))
    use_block_solve = bool(_cfg_get(cfg, "use_block_solve", True))
    metrics: dict[str, float | int | bool] = {
        "symbolic_reuse_requested": use_symbolic_reuse,
        "symbolic_reuse_used": False,
        "symbolic_reuse_fallback": False,
        "symbolic_reuse_fallback_anchor_index": -1,
        "factorize_calls": 0,
        "factorize_seconds": 0.0,
        "analysis_reuse_count": 0,
    }
    factors_seen: list[Any] = []
    gains = np.empty(omegas.size, dtype=float)
    iterations = np.empty(omegas.size, dtype=int)
    residuals = np.empty(omegas.size, dtype=float)
    responses = np.empty((n, omegas.size), dtype=complex)
    forcings = np.empty((n, omegas.size), dtype=complex)
    reports: list[CertificationReport | None] = [None] * omegas.size

    def factorize_anchor(anchor: complex) -> Any:
        start = time.perf_counter()
        fac = factorize(
            shifted_operator(L0_beta, anchor),
            env_prefix="LINSTABPY_RESOLVENT",
            solver_cfg=solver_cfg,
            cfg=factorize_cfg,
        )
        metrics["factorize_calls"] = int(metrics["factorize_calls"]) + 1
        metrics["factorize_seconds"] = float(metrics["factorize_seconds"]) + time.perf_counter() - start
        factors_seen.append(fac)
        return fac

    def factor_metric_sum(name: str) -> float:
        total = 0.0
        seen: set[int] = set()
        for fac in factors_seen:
            if id(fac) in seen:
                continue
            seen.add(id(fac))
            total += float((getattr(fac, "metrics", None) or {}).get(name, 0.0))
        return total

    def solve_one(pos: int, factor: Any) -> None:
        omega = omegas[pos]
        anchor = anchor_values[int(anchor_indices[pos])]
        operator = shifted_operator(L0_beta, omega)
        iter_log: list[int] = []
        residual_log: list[float] = []
        adj_iter_log: list[int] = []
        adj_residual_log: list[float] = []

        def precondition(rhs: Array) -> Array:
            if use_block_solve and hasattr(factor, "solve_block"):
                block = np.asarray(rhs, dtype=complex).reshape(-1, 1)
                return np.asarray(factor.solve_block(block)[:, 0], dtype=complex)
            return factor.solve(rhs)

        def apply_H(rhs: Array) -> Array:
            x, iters, gmres_residual = _right_preconditioned_gmres(
                lambda z: _matvec(operator, z), precondition, rhs, rtol=rtol, maxiter=maxiter
            )
            iter_log.append(iters)
            residual_log.append(max(gmres_residual, _relative_residual(operator, x, np.asarray(rhs, dtype=complex))))
            return x

        def apply_H_adj(rhs: Array) -> Array:
            x, iters, gmres_residual = _right_preconditioned_gmres(
                lambda z: _adjoint_matvec(operator, z), factor.solve_adjoint, rhs, rtol=rtol, maxiter=maxiter
            )
            adj_iter_log.append(iters)
            adj_residual_log.append(max(gmres_residual, _relative_residual(operator, x, np.asarray(rhs, dtype=complex), adjoint=True)))
            return x

        probes_a, probes_b = _cfg_get(cfg, "probes_a"), _cfg_get(cfg, "probes_b")
        if probes_a is None or probes_b is None:
            probes_a, probes_b = _default_probes(n)
        probe_forward_residuals = []
        probe_adjoint_residuals = []
        used_probe_block = False
        if use_block_solve and hasattr(factor, "solve_block") and omega == anchor and len(probes_a) > 1:
            probe_block = np.column_stack([np.asarray(probe, dtype=complex).reshape(-1) for probe in probes_a])
            probe_x = factor.solve_block(probe_block)
            for probe, x in zip(probes_a, probe_x.T, strict=True):
                probe_forward_residuals.append(_relative_residual(operator, x, np.asarray(probe, dtype=complex)))
            used_probe_block = True
        for probe in probes_a:
            if not used_probe_block:
                x = apply_H(probe)
                probe_forward_residuals.append(_relative_residual(operator, x, np.asarray(probe, dtype=complex)))
            xa = apply_H_adj(probe)
            probe_adjoint_residuals.append(_relative_residual(operator, xa, np.asarray(probe, dtype=complex), adjoint=True))

        cert_cfg = _cert_cfg(cfg, n, factor, probe_forward_residuals, probe_adjoint_residuals)
        cert_cfg["probes_a"] = probes_a
        cert_cfg["probes_b"] = probes_b
        omega_gains, omega_responses, omega_forcings, report = certified_rsvd(apply_H, apply_H_adj, W, invW, cert_cfg)
        gains[pos] = float(omega_gains[0])
        responses[:, pos] = np.asarray(omega_responses[:, 0], dtype=complex)
        forcings[:, pos] = np.asarray(omega_forcings[:, 0], dtype=complex)
        all_iters = iter_log + adj_iter_log
        all_residuals = residual_log + adj_residual_log
        iterations[pos] = max(all_iters) if all_iters else 0
        residuals[pos] = max(all_residuals) if all_residuals else 0.0
        reports[pos] = report

    def solve_anchor_positions(anchor_i: int, factor: Any) -> None:
        for pos in np.flatnonzero(anchor_indices == anchor_i):
            solve_one(int(pos), factor)

    first_factor = factorize_anchor(anchor_values[0])
    refactorize_in_place = getattr(first_factor, "refactorize", None)
    if use_symbolic_reuse and anchor_values.size > 1 and callable(refactorize_in_place):
        for anchor_i, anchor in enumerate(anchor_values):
            if anchor_i > 0:
                try:
                    first_factor.refactorize(shifted_operator(L0_beta, anchor))
                except (PETSc.Error, NotImplementedError, ValueError):
                    metrics["symbolic_reuse_fallback"] = True
                    metrics["symbolic_reuse_fallback_anchor_index"] = anchor_i
                    for fallback_i in range(anchor_i, anchor_values.size):
                        solve_anchor_positions(fallback_i, factorize_anchor(anchor_values[fallback_i]))
                    break
                metrics["symbolic_reuse_used"] = True
            solve_anchor_positions(anchor_i, first_factor)
    else:
        solve_anchor_positions(0, first_factor)
        for anchor_i in range(1, anchor_values.size):
            solve_anchor_positions(anchor_i, factorize_anchor(anchor_values[anchor_i]))

    metrics["analysis_reuse_count"] = int(factor_metric_sum("refactor_count"))
    metrics["refactor_seconds"] = factor_metric_sum("refactor_seconds")
    metrics["solve_block_calls"] = int(factor_metric_sum("solve_block_calls"))
    metrics["solve_block_rhs"] = int(factor_metric_sum("solve_block_rhs"))
    metrics["solve_block_seconds"] = factor_metric_sum("solve_block_seconds")

    return SweepResult(
        omega=omegas,
        gain=gains,
        iterations=iterations,
        residuals=residuals,
        anchor_indices=anchor_indices,
        responses=responses,
        forcings=forcings,
        certification_reports=tuple(report for report in reports if report is not None),
        metrics=metrics,
    )


__all__ = ["SweepResult", "sweep_stable_regime"]
