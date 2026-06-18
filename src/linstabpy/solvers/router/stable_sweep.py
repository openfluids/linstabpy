# SPDX-License-Identifier: GPL-3.0-only
"""Anchor-preconditioned stable-regime resolvent frequency sweeps."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import scipy.sparse as sp

from linstabpy.linalg.backends import factor_backend_available
from linstabpy.linalg.factorize import Factorization, factorize
from linstabpy.linalg.shifted import shifted_operator
from linstabpy.solvers.router.certification import CertificationReport, certified_rsvd
from linstabpy.solvers.router.recycling import (
    combine_recycle_bases,
    empty_recycle_basis,
    prepare_recycle_subspace,
    recycled_right_preconditioned_gmres,
)
from linstabpy.utils.petsc_core import _is_petsc_matrix, vec_to_numpy
from linstabpy.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

Array = np.ndarray
MetricValue = float | int | bool | str


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
    metrics: dict[str, MetricValue] = field(default_factory=dict)


def _cfg_get(cfg: Mapping[str, Any] | object | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return False
    raise ValueError(f"expected boolean-like value, got {value!r}")


def _cfg_lookup(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _compressed_anchor_controls(cfg: Mapping[str, Any] | object | None) -> tuple[bool, dict[str, str], bool]:
    raw = _cfg_get(cfg, "compressed_anchor", None)
    if raw is None:
        raw = _cfg_get(cfg, "compressed_anchors", False)
    if isinstance(raw, Mapping):
        enabled = _as_bool(_cfg_lookup(raw, "enabled", "enable", default=True))
        compression = str(_cfg_lookup(raw, "compression", "compression_type", default="blr")).strip().lower()
        rel_tol = float(_cfg_lookup(raw, "rel_tol", "compression_rel_tol", default=1.0e-4))
        measure_exact_lu = _as_bool(_cfg_lookup(raw, "measure_exact_lu", default=True))
        optional_keys = {
            "abs_tol": "mat_strumpack_compression_abs_tol",
            "compression_abs_tol": "mat_strumpack_compression_abs_tol",
            "leaf_size": "mat_strumpack_compression_leaf_size",
            "min_sep_size": "mat_strumpack_compression_min_sep_size",
            "lossy_precision": "mat_strumpack_compression_lossy_precision",
            "butterfly_levels": "mat_strumpack_compression_butterfly_levels",
        }
        options = {
            "mat_strumpack_compression": compression,
            "mat_strumpack_compression_rel_tol": f"{rel_tol:g}",
        }
        for key, option_name in optional_keys.items():
            if key in raw:
                options[option_name] = str(raw[key])
    else:
        enabled = _as_bool(raw)
        options = {
            "mat_strumpack_compression": "blr",
            "mat_strumpack_compression_rel_tol": "0.0001",
        }
        measure_exact_lu = True
    return enabled, options if enabled else {}, measure_exact_lu


def _recycling_controls(cfg: Mapping[str, Any] | object | None) -> tuple[bool, int, float]:
    raw = _cfg_get(cfg, "recycling", None)
    if raw is None:
        raw = _cfg_get(cfg, "gcrodr_recycling", None)
    if raw is None:
        raw = _cfg_get(cfg, "recycle", False)
    if isinstance(raw, Mapping):
        enabled = _as_bool(_cfg_lookup(raw, "enabled", "enable", default=True))
        max_dim = int(_cfg_lookup(raw, "max_dim", "recycle_dim", "dim", default=8))
        orth_tol = float(_cfg_lookup(raw, "orth_tol", "orthogonalization_tol", default=1.0e-12))
    else:
        enabled = _as_bool(raw)
        max_dim = 8
        orth_tol = 1.0e-12
    if max_dim < 0:
        raise ValueError("recycling max_dim must be nonnegative")
    if enabled and max_dim == 0:
        raise ValueError("recycling max_dim must be positive when recycling is enabled")
    if orth_tol <= 0.0:
        raise ValueError("recycling orth_tol must be positive")
    return enabled, max_dim, orth_tol


@contextmanager
def _temporary_petsc_options(options: Mapping[str, str]):
    opts = PETSc.Options()
    old_values: dict[str, str | None] = {}
    try:
        for key, value in options.items():
            old_values[key] = opts.getString(key) if opts.hasName(key) else None
            opts.setValue(key, value)
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                opts.delValue(key)
            else:
                opts.setValue(key, old)


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


def _with_direct_factor_backend(solver_cfg: Mapping[str, Any] | None, backend: str) -> dict[str, Any]:
    base: dict[str, Any] = {} if solver_cfg is None else deepcopy(dict(solver_cfg))
    if "solver" in base:
        solver = deepcopy(dict(base.get("solver") or {}))
        base["solver"] = solver
    else:
        solver = base
    source = solver.get("resolvent") if isinstance(solver.get("resolvent"), Mapping) else solver.get("linsolve", {})
    section = deepcopy(dict(source)) if isinstance(source, Mapping) else {}
    section.update({"type": "direct", "factor_backend": backend})
    if backend != "mumps":
        section.update({"mumps_blr": False, "mumps_ordering": None})
    solver["resolvent"] = section
    return base if base else {"solver": {"resolvent": section}}


def _factorize_cfg_with_backend(factorize_cfg: Any, solver_cfg: Mapping[str, Any] | None, backend: str) -> Any:
    from linstabpy.config import Config

    if factorize_cfg is None:
        cfg_obj = Config.load(_with_direct_factor_backend(solver_cfg, backend))
    elif isinstance(factorize_cfg, Mapping):
        cfg_obj = Config.load(factorize_cfg)
    else:
        cfg_obj = factorize_cfg
    try:
        linear = replace(
            cfg_obj.resolvent.linear,
            type="direct",
            factor_backend=backend,
            mumps_blr=False,
            mumps_ordering=None,
        )
        return replace(cfg_obj, resolvent=replace(cfg_obj.resolvent, linear=linear))
    except AttributeError as exc:
        raise TypeError("factorize_cfg must be a linstabpy.config.Config-compatible object") from exc


def _factor_info(factor: Any) -> dict[str, float]:
    ksp = getattr(factor, "_ksp", None)
    if ksp is None:
        return {}
    try:
        mat = ksp.getPC().getFactorMatrix()
    except (AttributeError, PETSc.Error):
        return {}
    for info_type in (PETSc.Mat.InfoType.GLOBAL_SUM, PETSc.Mat.InfoType.GLOBAL_MAX, PETSc.Mat.InfoType.LOCAL):
        try:
            return {str(key): float(value) for key, value in mat.getInfo(info_type).items()}
        except (AttributeError, PETSc.Error):
            continue
    return {}


def _factor_metric_number(factor: Any, keys: Sequence[str]) -> float | None:
    metrics = getattr(factor, "metrics", None) or {}
    for key in keys:
        if key in metrics:
            value = float(metrics[key])
            if np.isfinite(value) and value > 0.0:
                return value
    return None


def _factor_memory_bytes(factor: Any, *, allow_fill_estimate: bool) -> tuple[float | None, str]:
    direct = _factor_metric_number(
        factor,
        ("factor_bytes", "compressed_factor_bytes", "memory_bytes", "strumpack_factor_bytes"),
    )
    if direct is not None:
        return direct, "factor.metrics memory_bytes"
    info = _factor_info(factor)
    memory = float(info.get("memory", 0.0))
    if np.isfinite(memory) and memory > 0.0:
        return memory, "PETSc MatInfo.memory"
    if allow_fill_estimate:
        fill_nnz = max(float(info.get("nz_allocated", 0.0)), float(info.get("nz_used", 0.0)))
        if np.isfinite(fill_nnz) and fill_nnz > 0.0:
            return fill_nnz * np.dtype(np.complex128).itemsize, "PETSc MatInfo factor fill estimate"
    return None, "not extractable via current API"


def _factor_effective_rank(factor: Any) -> tuple[float | None, str]:
    rank = _factor_metric_number(
        factor,
        ("effective_rank", "strumpack_effective_rank", "compression_effective_rank"),
    )
    if rank is not None:
        return rank, "factor.metrics effective_rank"
    return None, "effective rank not extractable via current API"


def _exact_lu_fill_bytes(operator: sp.spmatrix | PETSc.Mat, solver_cfg: Mapping[str, Any] | None) -> tuple[float | None, str]:
    if not factor_backend_available("mumps"):
        return None, "exact-LU fill bytes not extractable: MUMPS backend unavailable"
    exact_solver_cfg = _with_direct_factor_backend(solver_cfg, "mumps")
    exact_cfg = _factorize_cfg_with_backend(None, exact_solver_cfg, "mumps")
    exact_factor = factorize(
        operator,
        env_prefix="LINSTABPY_RESOLVENT",
        solver_cfg=exact_solver_cfg,
        cfg=exact_cfg,
    )
    try:
        return _factor_memory_bytes(exact_factor, allow_fill_estimate=True)
    finally:
        exact_factor.destroy()


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
    *,
    compressed_anchor: bool = False,
) -> dict[str, Any]:
    probes_a, probes_b = _cfg_get(cfg, "probes_a"), _cfg_get(cfg, "probes_b")
    if probes_a is None or probes_b is None:
        probes_a, probes_b = _default_probes(n)
    actual_tol = float(_cfg_get(cfg, "actual_solve_tol", _cfg_get(cfg, "fgmres_rtol", 1.0e-10)))
    adjoint_factor_tol_default = actual_tol if compressed_anchor else 1.0e-12
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
        # both satisfy the per-solve bound (which stays enforced).  Exact-LU
        # anchors keep the strict adjoint-factor default; compressed anchors use
        # the approximate factor's own solve tolerance because check #3 verifies
        # factor.solve_adjoint is the adjoint of that approximate factor, not of
        # an ideal exact-LU factor.
        "linearity_tol": float(_cfg_get(cfg, "linearity_tol", 1.0e-10)),
        "adjoint_factor_tol": float(_cfg_get(cfg, "adjoint_factor_tol", adjoint_factor_tol_default)),
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
    compressed_anchor, strumpack_options, measure_exact_lu = _compressed_anchor_controls(cfg)
    recycling_enabled, recycling_max_dim, recycling_orth_tol = _recycling_controls(cfg)
    if recycling_enabled and compressed_anchor:
        raise ValueError("recycling and compressed anchors are not combined in this evaluation path")
    active_solver_cfg = _with_direct_factor_backend(solver_cfg, "strumpack") if compressed_anchor else solver_cfg
    active_factorize_cfg = (
        _factorize_cfg_with_backend(factorize_cfg, active_solver_cfg, "strumpack")
        if compressed_anchor
        else factorize_cfg
    )
    W = np.asarray(_cfg_get(cfg, "W", np.ones(n, dtype=complex)), dtype=complex).reshape(-1)
    invW = np.asarray(_cfg_get(cfg, "invW", np.ones(n, dtype=complex)), dtype=complex).reshape(-1)

    anchor_indices = _nearest_anchor_indices(omegas, anchor_values)
    use_symbolic_reuse = bool(_cfg_get(cfg, "reuse_symbolic", True))
    use_block_solve = bool(_cfg_get(cfg, "use_block_solve", True))
    metrics: dict[str, MetricValue] = {
        "symbolic_reuse_requested": use_symbolic_reuse,
        "symbolic_reuse_used": False,
        "symbolic_reuse_fallback": False,
        "symbolic_reuse_fallback_anchor_index": -1,
        "factorize_calls": 0,
        "factorize_seconds": 0.0,
        "analysis_reuse_count": 0,
        "compressed_anchor_requested": compressed_anchor,
        "compression_backend": "strumpack" if compressed_anchor else "mumps",
        "compression_ratio": float("nan"),
        "effective_rank": -1,
        "compressed_factor_bytes": float("nan"),
        "exact_lu_fill_bytes": float("nan"),
        "compression_note": "compressed anchors disabled",
        "recycling_requested": recycling_enabled,
        "recycle_max_dim": recycling_max_dim if recycling_enabled else 0,
        "recycle_fixed_per_pass": True,
        "recycle_basis_updates": 0,
        "recycle_forward_rank_max": 0,
        "recycle_adjoint_rank_max": 0,
        "recycle_forward_rank_final": 0,
        "recycle_adjoint_rank_final": 0,
        "recycle_setup_matvecs_total": 0,
        "matvecs_total": 0,
        "matvecs_total_unrecycled": 0,
        "matvecs_total_recycled": 0,
        "matvecs_total_actual": 0,
        "matvecs_delta_recycled_minus_baseline": 0,
        "gmres_iterations_total": 0,
        "gmres_iterations_total_unrecycled": 0,
        "gmres_iterations_total_recycled": 0,
        "gmres_iterations_total_actual": 0,
        "gmres_iterations_delta_recycled_minus_baseline": 0,
        "recycling_note": "recycling enabled" if recycling_enabled else "recycling disabled",
    }
    compression_samples: list[tuple[float | None, float | None, float | None, str]] = []
    factors_seen: list[Any] = []
    gains = np.empty(omegas.size, dtype=float)
    iterations = np.empty(omegas.size, dtype=int)
    residuals = np.empty(omegas.size, dtype=float)
    responses = np.empty((n, omegas.size), dtype=complex)
    forcings = np.empty((n, omegas.size), dtype=complex)
    reports: list[CertificationReport | None] = [None] * omegas.size

    def record_compression_sample(operator: sp.spmatrix | PETSc.Mat, factor: Any) -> None:
        if not compressed_anchor:
            return
        compressed_bytes, compressed_source = _factor_memory_bytes(factor, allow_fill_estimate=False)
        exact_bytes, exact_source = (
            _exact_lu_fill_bytes(operator, active_solver_cfg)
            if measure_exact_lu
            else (None, "exact-LU fill-byte measurement disabled")
        )
        effective_rank, rank_source = _factor_effective_rank(factor)
        note = (
            f"compressed={compressed_source}; exact_lu={exact_source}; "
            f"effective_rank={rank_source}"
        )
        compression_samples.append((compressed_bytes, exact_bytes, effective_rank, note))

    def factorize_anchor(anchor: complex) -> Any:
        operator = shifted_operator(L0_beta, anchor)
        start = time.perf_counter()
        with _temporary_petsc_options(strumpack_options):
            fac = factorize(
                operator,
                env_prefix="LINSTABPY_RESOLVENT",
                solver_cfg=active_solver_cfg,
                cfg=active_factorize_cfg,
            )
        metrics["factorize_calls"] = int(metrics["factorize_calls"]) + 1
        metrics["factorize_seconds"] = float(metrics["factorize_seconds"]) + time.perf_counter() - start
        factors_seen.append(fac)
        record_compression_sample(operator, fac)
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

    def add_int_metric(name: str, amount: int) -> None:
        metrics[name] = int(metrics[name]) + int(amount)

    def record_unrecycled_counts(iters: int, *, actual: bool) -> None:
        add_int_metric("matvecs_total", iters)
        add_int_metric("matvecs_total_unrecycled", iters)
        add_int_metric("gmres_iterations_total", iters)
        add_int_metric("gmres_iterations_total_unrecycled", iters)
        if actual:
            add_int_metric("matvecs_total_actual", iters)
            add_int_metric("gmres_iterations_total_actual", iters)

    def record_recycled_setup(matvecs: int) -> None:
        add_int_metric("recycle_setup_matvecs_total", matvecs)
        add_int_metric("matvecs_total_recycled", matvecs)
        add_int_metric("matvecs_total_actual", matvecs)

    def record_recycled_counts(iters: int, matvecs: int) -> None:
        add_int_metric("matvecs_total_recycled", matvecs)
        add_int_metric("matvecs_total_actual", matvecs)
        add_int_metric("gmres_iterations_total_recycled", iters)
        add_int_metric("gmres_iterations_total_actual", iters)

    def solve_one(pos: int, factor: Any, forward_basis: Array, adjoint_basis: Array) -> tuple[Array, Array]:
        omega = omegas[pos]
        anchor = anchor_values[int(anchor_indices[pos])]
        operator = shifted_operator(L0_beta, omega)
        iter_log: list[int] = []
        residual_log: list[float] = []
        adj_iter_log: list[int] = []
        adj_residual_log: list[float] = []
        forward_candidates = empty_recycle_basis(n)
        adjoint_candidates = empty_recycle_basis(n)
        forward_recycle = None
        adjoint_recycle = None
        if recycling_enabled:
            forward_recycle = prepare_recycle_subspace(
                lambda z: _matvec(operator, z),
                forward_basis,
                n=n,
                max_dim=recycling_max_dim,
                orth_tol=recycling_orth_tol,
            )
            adjoint_recycle = prepare_recycle_subspace(
                lambda z: _adjoint_matvec(operator, z),
                adjoint_basis,
                n=n,
                max_dim=recycling_max_dim,
                orth_tol=recycling_orth_tol,
            )
            record_recycled_setup(forward_recycle.setup_matvecs + adjoint_recycle.setup_matvecs)
            metrics["recycle_forward_rank_max"] = max(int(metrics["recycle_forward_rank_max"]), forward_recycle.rank)
            metrics["recycle_adjoint_rank_max"] = max(int(metrics["recycle_adjoint_rank_max"]), adjoint_recycle.rank)

        def precondition(rhs: Array) -> Array:
            if use_block_solve and not compressed_anchor and hasattr(factor, "solve_block"):
                block = np.asarray(rhs, dtype=complex).reshape(-1, 1)
                return np.asarray(factor.solve_block(block)[:, 0], dtype=complex)
            return factor.solve(rhs)

        def apply_H(rhs: Array) -> Array:
            nonlocal forward_candidates
            if recycling_enabled:
                baseline = _right_preconditioned_gmres(
                    lambda z: _matvec(operator, z), precondition, rhs, rtol=rtol, maxiter=maxiter
                )
                record_unrecycled_counts(baseline[1], actual=False)
                recycled = recycled_right_preconditioned_gmres(
                    lambda z: _matvec(operator, z),
                    precondition,
                    rhs,
                    rtol=rtol,
                    maxiter=maxiter,
                    recycle=forward_recycle,
                    candidate_dim=recycling_max_dim,
                    orth_tol=recycling_orth_tol,
                )
                record_recycled_counts(recycled.iterations, recycled.matvecs)
                forward_candidates = combine_recycle_bases(
                    forward_candidates,
                    recycled.candidate_basis,
                    max_dim=recycling_max_dim,
                    n=n,
                    orth_tol=recycling_orth_tol,
                )
                x = recycled.x
                iters = recycled.iterations
                gmres_residual = recycled.residual
            else:
                x, iters, gmres_residual = _right_preconditioned_gmres(
                    lambda z: _matvec(operator, z), precondition, rhs, rtol=rtol, maxiter=maxiter
                )
                record_unrecycled_counts(iters, actual=True)
            iter_log.append(iters)
            residual_log.append(max(gmres_residual, _relative_residual(operator, x, np.asarray(rhs, dtype=complex))))
            return x

        def apply_H_adj(rhs: Array) -> Array:
            nonlocal adjoint_candidates
            if recycling_enabled:
                baseline = _right_preconditioned_gmres(
                    lambda z: _adjoint_matvec(operator, z), factor.solve_adjoint, rhs, rtol=rtol, maxiter=maxiter
                )
                record_unrecycled_counts(baseline[1], actual=False)
                recycled = recycled_right_preconditioned_gmres(
                    lambda z: _adjoint_matvec(operator, z),
                    factor.solve_adjoint,
                    rhs,
                    rtol=rtol,
                    maxiter=maxiter,
                    recycle=adjoint_recycle,
                    candidate_dim=recycling_max_dim,
                    orth_tol=recycling_orth_tol,
                )
                record_recycled_counts(recycled.iterations, recycled.matvecs)
                adjoint_candidates = combine_recycle_bases(
                    adjoint_candidates,
                    recycled.candidate_basis,
                    max_dim=recycling_max_dim,
                    n=n,
                    orth_tol=recycling_orth_tol,
                )
                x = recycled.x
                iters = recycled.iterations
                gmres_residual = recycled.residual
            else:
                x, iters, gmres_residual = _right_preconditioned_gmres(
                    lambda z: _adjoint_matvec(operator, z), factor.solve_adjoint, rhs, rtol=rtol, maxiter=maxiter
                )
                record_unrecycled_counts(iters, actual=True)
            adj_iter_log.append(iters)
            adj_residual_log.append(max(gmres_residual, _relative_residual(operator, x, np.asarray(rhs, dtype=complex), adjoint=True)))
            return x

        probes_a, probes_b = _cfg_get(cfg, "probes_a"), _cfg_get(cfg, "probes_b")
        if probes_a is None or probes_b is None:
            probes_a, probes_b = _default_probes(n)
        probe_forward_residuals = []
        probe_adjoint_residuals = []
        used_probe_block = False
        if use_block_solve and not compressed_anchor and hasattr(factor, "solve_block") and omega == anchor and len(probes_a) > 1:
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

        cert_cfg = _cert_cfg(
            cfg,
            n,
            factor,
            probe_forward_residuals,
            probe_adjoint_residuals,
            compressed_anchor=compressed_anchor,
        )
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

        if not recycling_enabled:
            return forward_basis, adjoint_basis
        next_forward = combine_recycle_bases(
            forward_candidates,
            forward_basis,
            max_dim=recycling_max_dim,
            n=n,
            orth_tol=recycling_orth_tol,
        )
        next_adjoint = combine_recycle_bases(
            adjoint_candidates,
            adjoint_basis,
            max_dim=recycling_max_dim,
            n=n,
            orth_tol=recycling_orth_tol,
        )
        if next_forward.shape[1] > 0 or next_adjoint.shape[1] > 0:
            add_int_metric("recycle_basis_updates", 1)
        metrics["recycle_forward_rank_final"] = next_forward.shape[1]
        metrics["recycle_adjoint_rank_final"] = next_adjoint.shape[1]
        return next_forward, next_adjoint

    def solve_anchor_positions(anchor_i: int, factor: Any) -> None:
        forward_basis = empty_recycle_basis(n)
        adjoint_basis = empty_recycle_basis(n)
        for pos in np.flatnonzero(anchor_indices == anchor_i):
            forward_basis, adjoint_basis = solve_one(int(pos), factor, forward_basis, adjoint_basis)

    first_factor = factorize_anchor(anchor_values[0])
    refactorize_in_place = getattr(first_factor, "refactorize", None)
    if use_symbolic_reuse and anchor_values.size > 1 and callable(refactorize_in_place):
        for anchor_i, anchor in enumerate(anchor_values):
            if anchor_i > 0:
                try:
                    operator = shifted_operator(L0_beta, anchor)
                    with _temporary_petsc_options(strumpack_options):
                        first_factor.refactorize(operator)
                    record_compression_sample(operator, first_factor)
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
    if compressed_anchor:
        compressed_total = sum(value for value, _exact, _rank, _note in compression_samples if value is not None)
        exact_total = sum(value for _compressed, value, _rank, _note in compression_samples if value is not None)
        ranks = [rank for _compressed, _exact, rank, _note in compression_samples if rank is not None]
        notes = []
        for _compressed, _exact, _rank, note in compression_samples:
            if note not in notes:
                notes.append(note)
        if compressed_total > 0.0:
            metrics["compressed_factor_bytes"] = float(compressed_total)
        if exact_total > 0.0:
            metrics["exact_lu_fill_bytes"] = float(exact_total)
        if compressed_total > 0.0 and exact_total > 0.0:
            ratio = float(compressed_total / exact_total)
            metrics["compression_ratio"] = ratio
            notes.append("compression helped" if ratio < 1.0 else "compression ratio >= 1 on this operator")
        else:
            notes.append("ratio not extractable via current API")
        if ranks:
            metrics["effective_rank"] = float(np.mean(ranks))
        metrics["compression_note"] = "; ".join(notes)

    if recycling_enabled:
        metrics["matvecs_delta_recycled_minus_baseline"] = int(metrics["matvecs_total_recycled"]) - int(metrics["matvecs_total"])
        metrics["gmres_iterations_delta_recycled_minus_baseline"] = int(metrics["gmres_iterations_total_recycled"]) - int(
            metrics["gmres_iterations_total"]
        )

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
