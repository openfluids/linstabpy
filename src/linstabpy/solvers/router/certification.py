# SPDX-License-Identifier: GPL-3.0-only
"""Certification harness for randomized resolvent SVD actions.

The routines in this module are intentionally independent of the sweep router:
they certify the algebraic assumptions that an inexact resolvent action must
satisfy before it is safe to use in RSVD.  Tests can call each check directly,
while :func:`certified_rsvd` runs the checks and then computes a small dense
Rayleigh-Ritz RSVD result from the supplied ``apply_H``/``apply_H_adj`` seams.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

Array = np.ndarray
Apply = Callable[[Array], Array]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one certification check."""

    passed: bool
    measured: float
    tol: float
    message: str = ""

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "passed": self.passed,
            "measured": self.measured,
            "tol": self.tol,
            "message": self.message,
        }


@dataclass(frozen=True)
class CertificationReport:
    """Certification output returned by :func:`certified_rsvd`."""

    checks: dict[str, CheckResult]
    derived_solve_tol: float
    seeds: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "derived_solve_tol": self.derived_solve_tol,
            "seeds": self.seeds,
            "checks": {name: result.as_dict() for name, result in self.checks.items()},
        }


def _cfg_get(cfg: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_vector(v: Array, n: int | None = None) -> Array:
    arr = np.asarray(v, dtype=complex).reshape(-1)
    if n is not None and arr.size != n:
        raise ValueError(f"expected vector length {n}, got {arr.size}")
    return arr


def _norm(v: Array) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=complex).reshape(-1)))


def _relative_norm(v: Array, scale: Array | float) -> float:
    denom = _norm(scale) if not np.isscalar(scale) else float(abs(scale))
    return _norm(v) / max(denom, np.finfo(float).tiny)


def derive_solve_tolerance(
    sigma_tol: float,
    inv_norm_est: float,
    *,
    safety: float = 0.1,
    min_tol: float = 10.0 * np.finfo(float).eps,
) -> float:
    """Derive a linear-solve relative tolerance from singular-value accuracy.

    Larger ``||K^-1||`` estimates imply worse conditioning of the shifted
    operator and therefore a tighter residual tolerance.  ``safety`` keeps the
    inner linear-solve error below the requested singular-value error budget.
    """
    sigma_tol = float(sigma_tol)
    inv_norm_est = float(inv_norm_est)
    safety = float(safety)
    if sigma_tol <= 0.0:
        raise ValueError("sigma_tol must be positive")
    if inv_norm_est <= 0.0 or not np.isfinite(inv_norm_est):
        raise ValueError("inv_norm_est must be positive and finite")
    if safety <= 0.0 or safety > 1.0:
        raise ValueError("safety must lie in (0, 1]")
    return max(float(min_tol), safety * sigma_tol / inv_norm_est)


def check_tolerance_tied_to_sigma_accuracy(
    actual_tol: float,
    sigma_tol: float,
    inv_norm_est: float,
    *,
    safety: float = 0.1,
    consistency_tol: float = 1.0e-14,
) -> CheckResult:
    """Check that the configured solve tolerance is no looser than derived."""
    derived = derive_solve_tolerance(sigma_tol, inv_norm_est, safety=safety)
    actual = float(actual_tol)
    measured = actual / derived
    passed = actual <= derived * (1.0 + consistency_tol)
    return CheckResult(passed, measured, 1.0, "actual_tol / derived_tol")


def check_forward_adjoint_consistency(
    forward_residuals: Sequence[float],
    adjoint_residuals: Sequence[float],
    forward_tol: float,
    adjoint_tol: float,
    *,
    consistency_tol: float = 1.0e-12,
) -> CheckResult:
    """Check matching tolerances and comparable realized residuals."""
    f = np.asarray(forward_residuals, dtype=float)
    a = np.asarray(adjoint_residuals, dtype=float)
    if f.shape != a.shape or f.size == 0:
        raise ValueError("forward_residuals and adjoint_residuals must have the same nonzero shape")
    f_tol = float(forward_tol)
    a_tol = float(adjoint_tol)
    if f_tol <= 0.0 or a_tol <= 0.0:
        raise ValueError("forward_tol and adjoint_tol must be positive")
    tol_mismatch = abs(f_tol - a_tol) / max(abs(f_tol), abs(a_tol), np.finfo(float).tiny)
    residual_mismatch = float(
        np.max(np.abs(f - a) / np.maximum(np.maximum(f, a), np.finfo(float).tiny))
    )
    f_tol_violation = float(np.max(f / f_tol))
    a_tol_violation = float(np.max(a / a_tol))
    measured = max(
        tol_mismatch / consistency_tol,
        residual_mismatch / consistency_tol,
        f_tol_violation,
        a_tol_violation,
    )
    passed = (
        tol_mismatch <= consistency_tol
        and residual_mismatch <= consistency_tol
        and f_tol_violation <= 1.0
        and a_tol_violation <= 1.0
    )
    return CheckResult(
        passed,
        measured,
        1.0,
        "max normalized tol/residual mismatch or realized-residual tolerance violation",
    )


def check_adjoint_of_approx_factor(
    approx_solve: Apply,
    approx_solve_adjoint: Apply,
    probes_a: Sequence[Array],
    probes_b: Sequence[Array],
    *,
    tol: float = 1.0e-12,
) -> CheckResult:
    """Verify ``approx_solve_adjoint`` is the Hilbert adjoint of ``approx_solve``."""
    if len(probes_a) != len(probes_b) or len(probes_a) == 0:
        raise ValueError("probe lists must have the same nonzero length")
    errs = []
    for a, b in zip(probes_a, probes_b, strict=True):
        av = _as_vector(a)
        bv = _as_vector(b, av.size)
        left = np.vdot(approx_solve(av), bv)
        right = np.vdot(av, approx_solve_adjoint(bv))
        errs.append(abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny))
    measured = float(np.max(errs))
    return CheckResult(measured <= tol, measured, tol, "max relative inner-product defect")


def check_deterministic_linearity(
    apply: Apply,
    probes_a: Sequence[Array],
    probes_b: Sequence[Array],
    *,
    tol: float = 1.0e-12,
) -> CheckResult:
    """Check fixed-tolerance linearity: ``A(a+b) == A(a)+A(b)`` at probes."""
    if len(probes_a) != len(probes_b) or len(probes_a) == 0:
        raise ValueError("probe lists must have the same nonzero length")
    errs = []
    for a, b in zip(probes_a, probes_b, strict=True):
        av = _as_vector(a)
        bv = _as_vector(b, av.size)
        lhs = apply(av + bv)
        rhs = apply(av) + apply(bv)
        errs.append(_relative_norm(lhs - rhs, rhs))
    measured = float(np.max(errs))
    return CheckResult(measured <= tol, measured, tol, "max relative linearity defect")


def _apply_weighted_normal(
    apply_H: Apply, apply_H_adj: Apply, W: Array, invW: Array, v: Array
) -> Array:
    return invW * apply_H_adj(W * apply_H(v))


def randomized_resolvent_svd(
    apply_H: Apply,
    apply_H_adj: Apply,
    W: Array,
    invW: Array,
    *,
    n_modes: int = 1,
    oversample: int = 4,
    seed: int = 0,
) -> tuple[Array, Array, Array]:
    """Compute weighted leading gains from supplied resolvent actions.

    This intentionally small implementation is for certification probes.  The
    production sweep still uses the existing PETSc/SLEPc path.
    """
    Wv = _as_vector(W)
    invWv = _as_vector(invW, Wv.size)
    if np.any(Wv == 0.0) or np.any(invWv == 0.0):
        raise ValueError("W and invW must be nonzero diagonal vectors")
    n = Wv.size
    k = min(n, max(int(n_modes), 1) + max(int(oversample), 0))
    rng = np.random.default_rng(int(seed))
    omega = rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
    y = np.column_stack(
        [_apply_weighted_normal(apply_H, apply_H_adj, Wv, invWv, omega[:, j]) for j in range(k)]
    )
    q, _ = np.linalg.qr(y, mode="reduced")
    aq = np.column_stack(
        [
            _apply_weighted_normal(apply_H, apply_H_adj, Wv, invWv, q[:, j])
            for j in range(q.shape[1])
        ]
    )
    small = q.conj().T @ aq
    eigvals, eigvecs = np.linalg.eig(small)
    order = np.argsort(np.real(eigvals))[::-1]
    take = order[: int(n_modes)]
    gains = np.sqrt(np.maximum(np.real(eigvals[take]), 0.0))
    forcings = q @ eigvecs[:, take]
    responses = np.empty_like(forcings)
    for j in range(forcings.shape[1]):
        f = forcings[:, j]
        f_norm = np.sqrt(max(float(np.real(np.vdot(f, Wv * f))), np.finfo(float).tiny))
        f = f / f_norm
        forcings[:, j] = f
        response = apply_H(f)
        if gains[j] > 0.0:
            response = response / gains[j]
        responses[:, j] = response
    return gains.astype(float), responses, forcings


def check_multi_seed_rsvd(
    apply_H: Apply,
    apply_H_adj: Apply,
    W: Array,
    invW: Array,
    seeds: Sequence[int],
    *,
    n_modes: int = 1,
    gain_tol: float = 1.0e-8,
    mode_tol: float = 1.0e-6,
    oversample: int = 4,
) -> CheckResult:
    """Check leading gain and forcing-mode agreement across RSVD seeds."""
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    runs = [
        randomized_resolvent_svd(
            apply_H, apply_H_adj, W, invW, n_modes=n_modes, oversample=oversample, seed=s
        )
        for s in seeds
    ]
    ref_gain = runs[0][0][0]
    ref_mode = runs[0][2][:, 0]
    defects = []
    for gains, _responses, forcings in runs[1:]:
        gain_defect = abs(gains[0] - ref_gain) / max(abs(ref_gain), np.finfo(float).tiny)
        corr = abs(np.vdot(ref_mode, forcings[:, 0])) / max(
            _norm(ref_mode) * _norm(forcings[:, 0]), np.finfo(float).tiny
        )
        mode_defect = max(0.0, 1.0 - float(corr))
        defects.append(max(gain_defect / gain_tol, mode_defect / mode_tol))
    measured = float(np.max(defects)) if defects else 0.0
    return CheckResult(measured <= 1.0, measured, 1.0, "max normalized gain/mode seed defect")


def certified_rsvd(
    apply_H: Apply,
    apply_H_adj: Apply,
    W: Array,
    invW: Array,
    cfg: Mapping[str, Any] | object,
) -> tuple[Array, Array, Array, CertificationReport]:
    """Run certification checks, then return RSVD gains and modes.

    ``cfg`` may be a mapping or lightweight object.  Required certification
    fields are ``sigma_tol``, ``inv_norm_est``, ``actual_solve_tol``,
    ``forward_residuals``, ``adjoint_residuals``, ``approx_solve``,
    ``approx_solve_adjoint``, ``probes_a``, and ``probes_b``.
    """
    sigma_tol = float(_cfg_get(cfg, "sigma_tol"))
    inv_norm_est = float(_cfg_get(cfg, "inv_norm_est"))
    safety = float(_cfg_get(cfg, "safety", 0.1))
    derived_tol = derive_solve_tolerance(sigma_tol, inv_norm_est, safety=safety)
    n_modes = int(_cfg_get(cfg, "n_modes", 1))
    oversample = int(_cfg_get(cfg, "oversample", 4))
    seeds = tuple(int(s) for s in _cfg_get(cfg, "seeds", (0, 1, 2)))
    linearity_tol = float(_cfg_get(cfg, "linearity_tol", 1.0e-12))
    adjoint_factor_tol = float(_cfg_get(cfg, "adjoint_factor_tol", 1.0e-12))
    consistency_tol = float(_cfg_get(cfg, "consistency_tol", 1.0e-12))

    probes_a = _cfg_get(cfg, "probes_a")
    probes_b = _cfg_get(cfg, "probes_b")
    forward_linearity = check_deterministic_linearity(
        apply_H, probes_a, probes_b, tol=linearity_tol
    )
    adjoint_linearity = check_deterministic_linearity(
        apply_H_adj, probes_a, probes_b, tol=linearity_tol
    )
    linearity = CheckResult(
        forward_linearity.passed and adjoint_linearity.passed,
        max(forward_linearity.measured, adjoint_linearity.measured),
        linearity_tol,
        f"forward={forward_linearity.measured:.6e}, adjoint={adjoint_linearity.measured:.6e} max relative linearity defect",
    )
    checks = {
        "tolerance_tied_to_sigma_accuracy": check_tolerance_tied_to_sigma_accuracy(
            float(_cfg_get(cfg, "actual_solve_tol")), sigma_tol, inv_norm_est, safety=safety
        ),
        "forward_adjoint_consistency": check_forward_adjoint_consistency(
            _cfg_get(cfg, "forward_residuals"),
            _cfg_get(cfg, "adjoint_residuals"),
            float(_cfg_get(cfg, "forward_tol", _cfg_get(cfg, "actual_solve_tol"))),
            float(_cfg_get(cfg, "adjoint_tol", _cfg_get(cfg, "actual_solve_tol"))),
            consistency_tol=consistency_tol,
        ),
        "adjoint_of_approx_factor": check_adjoint_of_approx_factor(
            _cfg_get(cfg, "approx_solve"),
            _cfg_get(cfg, "approx_solve_adjoint"),
            probes_a,
            probes_b,
            tol=adjoint_factor_tol,
        ),
        "deterministic_tol_linearity": linearity,
        "multi_seed_rsvd": (
            check_multi_seed_rsvd(
                apply_H,
                apply_H_adj,
                W,
                invW,
                seeds,
                n_modes=n_modes,
                gain_tol=float(_cfg_get(cfg, "seed_gain_tol", sigma_tol)),
                mode_tol=float(_cfg_get(cfg, "seed_mode_tol", 1.0e-6)),
                oversample=oversample,
            )
            if len(seeds) >= 2
            else CheckResult(True, 0.0, 1.0, "one seed: multi-seed check skipped")
        ),
    }
    report = CertificationReport(checks=checks, derived_solve_tol=derived_tol, seeds=seeds)
    if not report.passed:
        failed = ", ".join(name for name, result in checks.items() if not result.passed)
        for name, result in checks.items():
            LOGGER.warning(
                "rsvd_check %s passed=%s measured=%.6e tol=%.6e %s",
                name,
                result.passed,
                result.measured,
                result.tol,
                result.message,
            )
        required = _cfg_get(cfg, "certification_required", True)
        if required in (True, 1, "1", "true", "True", "yes", "on"):
            raise RuntimeError(f"RSVD certification failed: {failed}")
        LOGGER.warning("RSVD certification failed (continuing): %s", failed)
    gains, responses, forcings = randomized_resolvent_svd(
        apply_H, apply_H_adj, W, invW, n_modes=n_modes, oversample=oversample, seed=seeds[0]
    )
    return gains, responses, forcings, report


__all__ = [
    "CertificationReport",
    "CheckResult",
    "certified_rsvd",
    "check_adjoint_of_approx_factor",
    "check_deterministic_linearity",
    "check_forward_adjoint_consistency",
    "check_multi_seed_rsvd",
    "check_tolerance_tied_to_sigma_accuracy",
    "derive_solve_tolerance",
    "randomized_resolvent_svd",
]
