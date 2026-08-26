# SPDX-License-Identifier: GPL-3.0-only
"""Walk a frequency line with measured MUMPS anchors, then a residual gate.

The first two frequencies on each beta line pay a real factor. Those
points measure factor wall and SLEPc apply count. Later frequencies
try that factor as a right preconditioner.
A probe on the last response must beat that cost in at most four GMRES
iterations. A dead
walk raises and factors this omega. The gain still comes from SLEPc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from linstabpy.linalg.factorize import factorize
from linstabpy.linalg.shifted import shifted_operator
from linstabpy.solvers.router.map_record import job_remaining_s, time_left_for_factor
from linstabpy.solvers.router.stable_sweep import _matvec, _right_preconditioned_gmres
from linstabpy.utils.petsc_core import _is_petsc_matrix

DEFAULT_GMRES_RESTART = 50
DEFAULT_GMRES_MAXITER = 200
DEFAULT_WALK_RTOL = 1.0e-10
DEFAULT_N_ANCHOR = 2
WALK_COST_SLACK = 0.25
WALK_PROBE_ITER_CAP = 4
WALK_SOLVE_ITER_CAP = 4
WALK_APPLY_FLOOR = 40


class WalkDeferred(Exception):
    """Raised when remaining wall is too short for a safe factor."""


class WalkAborted(Exception):
    """Raised when a walk solve misses the residual or the wall budget."""


def combine_probes(
    forward_s: float,
    forward_iters: int,
    forward_residual: float,
    adjoint_s: float,
    adjoint_iters: int,
    adjoint_residual: float,
) -> tuple[float, int, float]:
    """Return time sum and the worse residual and iteration count."""
    return (
        float(forward_s) + float(adjoint_s),
        max(int(forward_iters), int(adjoint_iters)),
        max(float(forward_residual), float(adjoint_residual)),
    )


def walk_is_cheaper(
    *,
    probe_s: float,
    probe_iters: int,
    probe_residual: float,
    last_factor_s: float,
    rtol: float,
    restart: int,
    n_modes: int,
    n_applies: int = 0,
) -> bool:
    """Return True only if one probe solve says the walk can beat a new LU."""
    if last_factor_s <= 0.0:
        return False
    if int(n_applies) <= 0:
        return False
    if int(probe_iters) <= 0:
        return False
    if float(probe_residual) > float(rtol):
        return False
    if int(probe_iters) > min(WALK_PROBE_ITER_CAP, int(restart)):
        return False
    applies = max(int(n_applies), WALK_APPLY_FLOOR)
    return float(probe_s) * float(applies) < WALK_COST_SLACK * float(last_factor_s)


def probe_right_hand_side(n: int, last_solve: np.ndarray | None) -> np.ndarray:
    """Return the last response, or a random vector. Never a ones vector."""
    if last_solve is not None and int(np.asarray(last_solve).size) == int(n):
        return np.asarray(last_solve, dtype=complex).reshape(-1).copy()
    rng = np.random.default_rng(0)
    size = int(n)
    return rng.standard_normal(size) + 1j * rng.standard_normal(size)


@dataclass
class WalkPointResult:
    """One frequency on a walked line."""

    gain: float
    residual: float
    source: str
    iterations: int
    factor_s: float
    solve_s: float
    # SVD residuals of the returned leading triplet, the same two numbers the
    # point farm certifies with (rho_fwd = ||H v - s u|| / (s ||v||), rho_adj
    # its adjoint twin, both in the forcing norm). NaN when not computed.
    rho_fwd: float = float("nan")
    rho_adj: float = float("nan")


@dataclass
class LineWalker:
    """One live factor per beta line. New beta means a new walker."""

    solver_cfg: dict[str, Any] | None
    cfg_obj: Any
    rtol: float = DEFAULT_WALK_RTOL
    maxiter: int = DEFAULT_GMRES_MAXITER
    restart: int = DEFAULT_GMRES_RESTART
    factor: Any = None
    n_factor: int = 0
    last_solve: np.ndarray | None = None
    last_direct_wall_s: float = 0.0
    last_factor_s: float = 0.0
    last_n_applies: int = 0
    n_direct: int = 0
    n_anchor: int = DEFAULT_N_ANCHOR

    def close(self) -> None:
        """Free the live factor this walker created."""
        if self.factor is not None and hasattr(self.factor, "destroy"):
            self.factor.destroy()
        self.factor = None
        self.last_solve = None
        self.last_factor_s = 0.0
        self.last_direct_wall_s = 0.0
        self.last_n_applies = 0
        self.n_direct = 0

    def compute(
        self,
        L0: Any,
        omega: float,
        W_diag: np.ndarray,
        n_modes: int,
        *,
        B_mat: Any = None,
        C_mat: Any = None,
        filter_fn: Any = None,
        filter_ct_fn: Any = None,
        gamma: float = 0.0,
    ) -> WalkPointResult:
        """Compute the leading gain at one frequency."""
        need_s = self.last_direct_wall_s if self.last_direct_wall_s > 0.0 else self.last_factor_s
        if not time_left_for_factor(job_remaining_s(), need_s):
            raise WalkDeferred(f"remaining wall is shorter than the last factor ({need_s:.3f} s)")
        t0 = time.perf_counter()
        operator = shifted_operator(L0, omega, gamma=gamma)
        try:
            return self._compute_with_operator(
                operator,
                W_diag,
                n_modes,
                t0,
                B_mat=B_mat,
                C_mat=C_mat,
                filter_fn=filter_fn,
                filter_ct_fn=filter_ct_fn,
            )
        finally:
            _destroy_mat(operator)

    def _compute_with_operator(
        self,
        operator: Any,
        W_diag: np.ndarray,
        n_modes: int,
        t0: float,
        *,
        B_mat: Any,
        C_mat: Any,
        filter_fn: Any,
        filter_ct_fn: Any,
    ) -> WalkPointResult:
        if self.factor is None or self.n_direct < int(self.n_anchor):
            return self._factor_point(
                operator,
                W_diag,
                n_modes,
                t0,
                B_mat=B_mat,
                C_mat=C_mat,
                filter_fn=filter_fn,
                filter_ct_fn=filter_ct_fn,
            )

        deadline = None
        apply_budget_s = 0.0
        if self.last_factor_s > 0.0:
            deadline = t0 + self.last_factor_s
            applies = max(int(self.last_n_applies), WALK_APPLY_FLOOR)
            apply_budget_s = self.last_factor_s / float(applies)
        solve_cap = min(int(self.restart), WALK_SOLVE_ITER_CAP)
        gated = _GatedPcSolve(
            operator,
            self.factor,
            rtol=self.rtol,
            maxiter=solve_cap,
            restart=solve_cap,
            x0=None,
            deadline=deadline,
            apply_budget_s=apply_budget_s,
            raise_on_fail=False,
        )
        n = int(self.last_solve.size) if self.last_solve is not None else int(W_diag.size)
        probe_rhs = probe_right_hand_side(n, self.last_solve)
        t_fwd = time.perf_counter()
        gated.solve(probe_rhs)
        forward_s = time.perf_counter() - t_fwd
        forward_iters = int(gated.iterations)
        forward_residual = float(gated.max_residual)
        gated.iterations = 0
        gated.max_residual = 0.0
        t_adj = time.perf_counter()
        gated.solve_adjoint(probe_rhs)
        adjoint_s = time.perf_counter() - t_adj
        probe_s, probe_iters, probe_residual = combine_probes(
            forward_s,
            forward_iters,
            forward_residual,
            adjoint_s,
            int(gated.iterations),
            float(gated.max_residual),
        )
        gated.max_residual = max(forward_residual, float(gated.max_residual))
        gated.iterations = forward_iters + int(gated.iterations)
        gated.x0 = self.last_solve
        past_deadline = deadline is not None and time.perf_counter() >= deadline
        cheap = (
            not past_deadline
            and not gated.failed
            and walk_is_cheaper(
                probe_s=probe_s,
                probe_iters=probe_iters,
                probe_residual=probe_residual,
                last_factor_s=self.last_factor_s,
                rtol=self.rtol,
                restart=self.restart,
                n_modes=n_modes,
                n_applies=self.last_n_applies,
            )
        )
        if cheap:
            gated.raise_on_fail = True
            try:
                result = self._analyse(
                    gated,
                    W_diag,
                    n_modes,
                    B_mat=B_mat,
                    C_mat=C_mat,
                    filter_fn=filter_fn,
                    filter_ct_fn=filter_ct_fn,
                )
            except WalkAborted:
                result = None
            except Exception:
                if not gated.failed:
                    raise
                result = None
            if not gated.failed and result is not None:
                self.last_solve = gated.last
                return WalkPointResult(
                    gain=float(np.asarray(result[0]).reshape(-1)[0]),
                    rho_fwd=float(np.asarray(result[5]).reshape(-1)[0]),
                    rho_adj=float(np.asarray(result[6]).reshape(-1)[0]),
                    residual=float(gated.max_residual),
                    source="walk",
                    iterations=int(gated.iterations),
                    factor_s=0.0,
                    solve_s=time.perf_counter() - t0,
                )
        # Both labels factor this omega. They record different reasons.
        # "budget" means cost: the probe ran past its wall slice, or it was
        # healthy but no cheaper than a new factor. "fallback" means the
        # walk answered and the answer missed the residual or the cap.
        # A deadline sets both over_budget and failed, so test it first.
        if gated.over_budget:
            source = "budget"
        elif gated.failed or (
            float(gated.max_residual) > self.rtol or int(gated.iterations) > self.restart
        ):
            source = "fallback"
        else:
            source = "budget"
        t_replace = time.perf_counter()
        lu, factor_s = self._replace_factor(operator)
        recorder = _RecordingSolve(lu)
        result = self._analyse(
            recorder,
            W_diag,
            n_modes,
            B_mat=B_mat,
            C_mat=C_mat,
            filter_fn=filter_fn,
            filter_ct_fn=filter_ct_fn,
        )
        self.last_solve = recorder.last
        solve_only = time.perf_counter() - t_replace - factor_s
        self.last_factor_s = factor_s
        self.last_direct_wall_s = factor_s + max(solve_only, 0.0)
        self.last_n_applies = int(recorder.n_apply)
        self.n_direct += 1
        return WalkPointResult(
            gain=float(np.asarray(result[0]).reshape(-1)[0]),
            rho_fwd=float(np.asarray(result[5]).reshape(-1)[0]),
            rho_adj=float(np.asarray(result[6]).reshape(-1)[0]),
            residual=float(gated.max_residual),
            source=source,
            iterations=int(gated.iterations),
            factor_s=factor_s,
            solve_s=time.perf_counter() - t0 - factor_s,
        )

    def _factor_point(
        self,
        operator: Any,
        W_diag: np.ndarray,
        n_modes: int,
        t0: float,
        *,
        B_mat: Any,
        C_mat: Any,
        filter_fn: Any,
        filter_ct_fn: Any,
    ) -> WalkPointResult:
        if self.factor is None:
            t_fac = time.perf_counter()
            self.factor = factorize(
                operator,
                env_prefix="LINSTABPY_RESOLVENT",
                solver_cfg=self.solver_cfg,
                cfg=self.cfg_obj,
            )
            factor_s = time.perf_counter() - t_fac
            self.n_factor += 1
        else:
            _, factor_s = self._replace_factor(operator)
        recorder = _RecordingSolve(self.factor)
        result = self._analyse(
            recorder,
            W_diag,
            n_modes,
            B_mat=B_mat,
            C_mat=C_mat,
            filter_fn=filter_fn,
            filter_ct_fn=filter_ct_fn,
        )
        self.last_solve = recorder.last
        solve_s = time.perf_counter() - t0 - factor_s
        self.last_factor_s = factor_s
        self.last_direct_wall_s = factor_s + solve_s
        self.last_n_applies = int(recorder.n_apply)
        self.n_direct += 1
        return WalkPointResult(
            gain=float(np.asarray(result[0]).reshape(-1)[0]),
            rho_fwd=float(np.asarray(result[5]).reshape(-1)[0]),
            rho_adj=float(np.asarray(result[6]).reshape(-1)[0]),
            residual=0.0,
            source="factor",
            iterations=0,
            factor_s=factor_s,
            solve_s=solve_s,
        )

    def _replace_factor(self, operator: Any) -> tuple[Any, float]:
        t_fac = time.perf_counter()
        if self.factor is not None and hasattr(self.factor, "refactorize"):
            try:
                self.factor.refactorize(operator)
                self.n_factor += 1
                return self.factor, time.perf_counter() - t_fac
            except Exception:
                if hasattr(self.factor, "destroy"):
                    self.factor.destroy()
                self.factor = None
        self.factor = factorize(
            operator,
            env_prefix="LINSTABPY_RESOLVENT",
            solver_cfg=self.solver_cfg,
            cfg=self.cfg_obj,
        )
        self.n_factor += 1
        return self.factor, time.perf_counter() - t_fac

    def _analyse(
        self,
        lu: Any,
        W_diag: np.ndarray,
        n_modes: int,
        *,
        B_mat: Any,
        C_mat: Any,
        filter_fn: Any,
        filter_ct_fn: Any,
    ) -> tuple:
        from linstabpy.resolvent.analysis import _compute_resolvent_svd_residuals_from_apply
        from linstabpy.resolvent.gain import compute_leading_gains
        from linstabpy.resolvent.operator import ResolventOperator

        op = ResolventOperator(lu, B_mat, C_mat, filter_fn, filter_ct_fn)
        cfg = self.cfg_obj
        if cfg is None:
            from linstabpy.config import Config

            cfg = Config.load(self.solver_cfg if self.solver_cfg is not None else {})
        gains, responses, forcings, W_vec, invW_vec = compute_leading_gains(
            op, W_diag, n_modes, int(W_diag.size), cfg
        )
        # Certify the triplet the same way the point farm does: one forward and
        # one adjoint apply of the resolvent, measured in the forcing norm.
        rho_fwd, rho_adj = _compute_resolvent_svd_residuals_from_apply(
            gains,
            forcings,
            responses,
            W_vec,
            invW_vec,
            op.apply,
            op.apply_adjoint,
            cfg.resolvent.eps.forcing_norm,
        )
        return gains, responses, forcings, W_vec, invW_vec, rho_fwd, rho_adj


class _RecordingSolve:
    """Keep the last forward solve so the next omega can start from it."""

    def __init__(self, lu: Any) -> None:
        self._lu = lu
        self.last: np.ndarray | None = None
        self.n_apply: int = 0
        self.backend = getattr(lu, "backend", None)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        x = self._lu.solve(rhs)
        self.n_apply += 1
        self.last = np.asarray(x, dtype=complex).reshape(-1).copy()
        return x

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        self.n_apply += 1
        return self._lu.solve_transpose(rhs)

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        self.n_apply += 1
        if hasattr(self._lu, "solve_adjoint"):
            return self._lu.solve_adjoint(rhs)
        return self._lu.solve_transpose(rhs)


class _GatedPcSolve:
    """Look like a factor: solve by GMRES with a live factor as PC."""

    def __init__(
        self,
        operator: Any,
        pc_factor: Any,
        *,
        rtol: float,
        maxiter: int,
        restart: int,
        x0: np.ndarray | None,
        deadline: float | None = None,
        apply_budget_s: float = 0.0,
        raise_on_fail: bool = False,
    ) -> None:
        self._operator = operator
        self._pc = pc_factor
        self.rtol = float(rtol)
        self.maxiter = int(maxiter)
        self.restart = int(restart)
        self.x0 = x0
        self.deadline = deadline
        self.apply_budget_s = float(apply_budget_s)
        self.raise_on_fail = bool(raise_on_fail)
        self.failed = False
        self.over_budget = False
        self.max_residual = 0.0
        self.iterations = 0
        self.last: np.ndarray | None = None
        self.backend = getattr(pc_factor, "backend", "walk")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return self._solve(rhs, adjoint=False)

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        return self._solve(rhs, adjoint=True)

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        return self.solve_transpose(rhs)

    def _solve(self, rhs: np.ndarray, *, adjoint: bool) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=complex).reshape(-1)
        if self.failed or (self.deadline is not None and time.perf_counter() >= self.deadline):
            if self.deadline is not None and time.perf_counter() >= self.deadline:
                self.over_budget = True
            self.failed = True
            if self.raise_on_fail:
                raise WalkAborted("walk solve missed the residual or the wall budget")
            if self.last is not None and self.last.size == rhs_arr.size:
                return self.last
            return np.zeros(rhs_arr.size, dtype=complex)
        prec = self._pc.solve_adjoint if adjoint else self._pc.solve
        start = None if adjoint else self.x0
        if start is not None and np.asarray(start).size != rhs_arr.size:
            start = None
        apply_deadline = self.deadline
        if self.apply_budget_s > 0.0:
            local = time.perf_counter() + self.apply_budget_s
            apply_deadline = local if apply_deadline is None else min(apply_deadline, local)
        x, iters, residual = _right_preconditioned_gmres(
            lambda z: _matvec(self._operator, z, adjoint=adjoint),
            prec,
            rhs_arr,
            rtol=self.rtol,
            maxiter=self.maxiter,
            restart=self.restart,
            x0=start,
            deadline=apply_deadline,
        )
        self.iterations += int(iters)
        self.max_residual = max(self.max_residual, float(residual))
        self.last = np.asarray(x, dtype=complex).reshape(-1).copy()
        if apply_deadline is not None and time.perf_counter() >= apply_deadline:
            self.over_budget = True
            self.failed = True
        if int(iters) > self.restart:
            self.failed = True
        if residual > self.rtol:
            self.failed = True
        if self.failed and self.raise_on_fail:
            raise WalkAborted("walk solve missed the residual or the wall budget")
        return x


def _destroy_mat(mat: Any) -> None:
    if _is_petsc_matrix(mat) and hasattr(mat, "destroy"):
        mat.destroy()


__all__ = [
    "LineWalker",
    "WalkAborted",
    "WalkDeferred",
    "WalkPointResult",
    "combine_probes",
    "probe_right_hand_side",
    "walk_is_cheaper",
]
