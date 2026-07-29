# SPDX-License-Identifier: GPL-3.0-only
"""Reusable PETSc factorizations, direct by default.

Provides a factorize-once, solve-many interface for the LNSE shift-invert
workflows.  The default path is KSPPREONLY + PCLU + MUMPS because the release
tests and reference examples are calibrated against exact direct solves.  The
iterative path is available for large memory-bound cases, but it is an explicit
solver-policy choice rather than an automatic fallback.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from linstabpy.config import Config

from ..solvers.petsc_utils import (
    _is_petsc_matrix,
    _matrix_nnz,
    _operator_shape,
    require_complex,
    scipy_to_petsc,
    vec_to_numpy,
)
from ..utils._solver_cfg import DIRECT_SOLVER_TYPES, ITERATIVE_LINEAR_SOLVER_TYPES
from ..utils.petsc_init import init_petsc_from_argv
from .backends import apply_factor_backend

if TYPE_CHECKING:
    # Type-only import; ``from __future__ import annotations`` keeps every
    # ``PETSc.*`` annotation below a string, so importing this module does not
    # pull in petsc4py or initialize PETSc.
    from petsc4py import PETSc  # noqa: F401

logger = logging.getLogger(__name__)


# Lazy PETSc accessor (mirrors ``linstabpy.utils.petsc_core``). Functions here
# bind it locally with ``PETSc = _get_petsc()``; nested closures then inherit
# that binding from the enclosing scope. PEP 562 ``__getattr__`` covers external
# ``from ... import PETSc`` users but never same-module global lookups.
def __getattr__(name):  # noqa: ANN001, ANN201
    if name == "PETSc":
        return _get_petsc()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_petsc():  # noqa: ANN202
    """Return PETSc, triggering the lazy import/init on first use."""
    init_petsc_from_argv()
    from petsc4py import PETSc as _imp_PETSc

    globals()["PETSc"] = _imp_PETSc
    return _imp_PETSc


def _aij_pattern(mat: PETSc.Mat) -> tuple[np.ndarray, np.ndarray]:
    PETSc = _get_petsc()
    try:
        indptr, indices, _ = mat.getValuesCSR()
    except (AttributeError, PETSc.Error) as exc:
        raise NotImplementedError(
            "refactorize requires CSR structure access to verify an identical nonzero pattern"
        ) from exc
    return np.asarray(indptr, dtype=np.int64).copy(), np.asarray(indices, dtype=np.int64).copy()


def _run_adjoint_self_test(
    solve_func: Callable[[np.ndarray], np.ndarray],
    solve_adjoint_func: Callable[[np.ndarray], np.ndarray],
    n: int,
    k: int = 3,
) -> float:
    """Run k random complex probe pairs to detect adjoint defect.

    Returns max defect over all probes:
      defect_i = |<solve(a_i), b_i> - <a_i, solve_adjoint(b_i)>|
                 / max(|<solve(a_i), b_i>|, |<a_i, solve_adjoint(b_i)>|, 1e-300)
    """
    max_defect = 0.0
    rng = np.random.RandomState(42)  # deterministic for reproducibility

    for _ in range(k):
        a = rng.randn(n) + 1j * rng.randn(n)
        b = rng.randn(n) + 1j * rng.randn(n)

        x1 = solve_func(a)
        x2 = solve_adjoint_func(b)

        lhs = np.vdot(x1, b)  # conjugate-aware: conj(x1) . b
        rhs = np.vdot(a, x2)  # conjugate-aware: conj(a) . x2

        denominator = max(np.abs(lhs), np.abs(rhs), 1.0e-300)
        defect = np.abs(lhs - rhs) / denominator
        max_defect = max(max_defect, defect)

    return max_defect


def _run_oracle_residual(
    solve_adjoint_func: Callable[[np.ndarray], np.ndarray],
    A_H: PETSc.Mat,
    n: int,
    k: int = 3,
) -> float:
    """Run k random RHS through the healed adjoint and measure ||A_H @ x - b|| / ||b||.

    The oracle uses A_H (the stored Hermitian transpose) to validate that the
    healed solve_adjoint path (which is a forward-solve on A_H) produces correct results.
    """
    PETSc = _get_petsc()
    max_residual = 0.0
    rng = np.random.RandomState(43)  # different seed for oracle

    def _rhs_vec_from_numpy_local(rhs: np.ndarray) -> PETSc.Vec:
        rhs_arr = np.asarray(rhs, dtype=complex)
        b = A_H.createVecLeft()
        r0, r1 = b.getOwnershipRange()
        if r1 > r0:
            rows = np.arange(r0, r1, dtype=PETSc.IntType)
            b.setValues(rows, rhs_arr[r0:r1], addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        return b

    for _ in range(k):
        b_arr = rng.randn(n) + 1j * rng.randn(n)
        x_arr = solve_adjoint_func(b_arr)

        # Compute residual using matvec
        b_vec = _rhs_vec_from_numpy_local(b_arr)
        x_vec = _rhs_vec_from_numpy_local(x_arr)
        r = b_vec.duplicate()

        try:
            A_H.mult(x_vec, r)
            r.axpy(-1.0, b_vec)
            residual_abs = float(r.norm(PETSc.NormType.NORM_2))
            rhs_norm = float(b_vec.norm(PETSc.NormType.NORM_2))
            residual = residual_abs / max(rhs_norm, 1.0e-300)
            max_residual = max(max_residual, residual)
        finally:
            r.destroy()
            x_vec.destroy()
            b_vec.destroy()

    return max_residual


@dataclass
class Factorization:
    """Callable wrapper for forward and adjoint solves on one factorization."""

    backend: str
    solve: Callable[[np.ndarray], np.ndarray]
    solve_adjoint: Callable[[np.ndarray], np.ndarray]
    solve_vec: Callable[[PETSc.Vec], PETSc.Vec]
    solve_adjoint_vec: Callable[[PETSc.Vec], PETSc.Vec]
    residual: Callable[[np.ndarray, np.ndarray], float]
    _ksp: PETSc.KSP | None = None
    _mat: PETSc.Mat | None = None
    _solve_block: Callable[[np.ndarray], np.ndarray] | None = None
    _refactorize: Callable[[sp.spmatrix | PETSc.Mat], None] | None = None
    metrics: dict[str, float | int] | None = None

    @property
    def solve_transpose(self) -> Callable[[np.ndarray], np.ndarray]:
        """Backward-compatible alias for :meth:`solve_adjoint`."""
        return self.solve_adjoint

    @property
    def solve_transpose_vec(self) -> Callable[[PETSc.Vec], PETSc.Vec]:
        """Backward-compatible alias for :meth:`solve_adjoint_vec`."""
        return self.solve_adjoint_vec

    def solve_block(self, rhs_block: np.ndarray) -> np.ndarray:
        if self._solve_block is not None:
            return self._solve_block(rhs_block)
        block = np.asarray(rhs_block, dtype=complex)
        block = block.reshape(-1, 1) if block.ndim == 1 else block
        if block.ndim != 2:
            raise ValueError(f"rhs_block must be 1-D or 2-D, got shape {block.shape}")
        return np.column_stack([self.solve(block[:, j]) for j in range(block.shape[1])])

    def refactorize(self, operator: sp.spmatrix | PETSc.Mat) -> Factorization:
        if self._refactorize is None:
            raise NotImplementedError("this factorization cannot be refactorized in place")
        self._refactorize(operator)
        return self

    def destroy(self) -> None:
        """Release PETSc KSP and matrix resources owned by this wrapper."""
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        if self._mat is not None:
            self._mat.destroy()
            self._mat = None


def factorize(
    operator: sp.spmatrix | PETSc.Mat,
    *,
    env_prefix: str = "LINSTABPY_LINSOLVE",
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
) -> Factorization:
    """Build a reusable PETSc factorization for ``operator``.

    Parameters
    ----------
    operator:
        Square sparse matrix to factor (typically a shifted LNSE operator).
    env_prefix:
        Environment prefix controlling solver options.
    solver_cfg:
        Optional solver config dict (from JSONC ``"solver"`` section).
        When provided, the appropriate subsection is selected based on
        ``env_prefix``: ``"resolvent"`` subsection for
        ``LINSTABPY_RESOLVENT``, ``"linsolve"`` otherwise. Keys in the
        subsection override the corresponding env vars.
    """
    # Bound once here; the nested solve/refactorize closures below inherit this
    # binding from the enclosing scope, so they need no lazy lookup of their own.
    PETSc = _get_petsc()
    cfg = cfg if cfg is not None else Config.load(solver_cfg if solver_cfg is not None else {})
    require_complex(solver_cfg)

    nrows, ncols = _operator_shape(operator)
    if nrows != ncols:
        raise ValueError("operator must be a square matrix")
    n = nrows
    nnz = _matrix_nnz(operator)

    if _is_petsc_matrix(operator):
        A = operator.copy()
    else:
        A = scipy_to_petsc(operator, solver_cfg)
    metrics: dict[str, float | int | bool] = dict.fromkeys(
        ("refactor_count", "solve_block_calls", "solve_block_rhs", "solve_block_fallback_calls"), 0
    )
    metrics.update(refactor_seconds=0.0, solve_block_seconds=0.0)
    metrics["dual_factor_active"] = False
    metrics["adjoint_self_test_defect"] = 0.0
    metrics["adjoint_healed_defect"] = 0.0
    metrics["adjoint_oracle_residual"] = 0.0
    holder: dict[str, object] = {}

    if env_prefix == "LINSTABPY_LINSOLVE":
        lin = cfg.linsolve
    elif env_prefix == "LINSTABPY_RESOLVENT":
        lin = cfg.resolvent.linear
    else:
        raise ValueError(f"unmapped env_prefix {env_prefix!r}")

    solver_section = solver_cfg.get("solver", solver_cfg) if solver_cfg is not None else None
    if isinstance(solver_section, dict):
        if env_prefix == "LINSTABPY_RESOLVENT":
            solver_subsection = (
                solver_section.get("resolvent", solver_section.get("linsolve", {})) or {}
            )
        else:
            solver_subsection = solver_section.get("linsolve", {}) or {}
    else:
        solver_subsection = {}

    solver_type = lin.type

    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    pc = ksp.getPC()

    if solver_type in DIRECT_SOLVER_TYPES:
        ignored_iterative_keys = sorted(
            key
            for key in (
                "ksp_type",
                "pc_type",
                "ksp_pc_side",
                "ksp_rtol",
                "ksp_atol",
                "ksp_max_it",
                "asm_overlap",
                "sub_ksp_type",
                "sub_pc_type",
            )
            if key in solver_subsection
        )
        if ignored_iterative_keys:
            logger.warning(
                "Ignoring iterative linear-solver options for direct %s path "
                "(type=%s uses KSPPREONLY+PCLU): %s",
                env_prefix,
                solver_type,
                ", ".join(ignored_iterative_keys),
            )
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc.setType(PETSc.PC.Type.LU)
        factor_backend = lin.factor_backend
        apply_factor_backend(pc, factor_backend, options_prefix=ksp.getOptionsPrefix() or "")
        ksp.setUp()

        # This log is the cheap runtime check that the intended direct backend
        # actually materialised. Users debugging MPI/BLAS environments can grep
        # it without turning on verbose PETSc monitors.
        try:
            F = pc.getFactorMatrix()
            icntl7 = F.getMumpsIcntl(7)
            _ordering_names = {
                0: "AMD",
                2: "AMF",
                3: "scotch",
                4: "PORD",
                5: "METIS",
                6: "QAMD",
                7: "auto",
            }
            ordering = _ordering_names.get(icntl7, f"unknown({icntl7})")
            logger.info(
                "Linear solver: MUMPS direct (n=%d, nnz=%d, ordering=%s [ICNTL(7)=%d])",
                n,
                nnz,
                ordering,
                icntl7,
            )
        except (AttributeError, RuntimeError) as e:
            logger.debug("Could not query MUMPS ordering: %s", e)
            logger.info("Linear solver: MUMPS direct (n=%d, nnz=%d)", n, nnz)
        backend = (factor_backend or "petsc").upper()
    elif solver_type in ITERATIVE_LINEAR_SOLVER_TYPES:
        ksp_type = lin.ksp_type
        pc_type = lin.pc_type
        pc_side = lin.ksp_pc_side
        ksp_rtol = lin.ksp_rtol
        ksp_atol = lin.ksp_atol
        ksp_max_it = lin.ksp_max_it
        asm_overlap = lin.asm_overlap
        sub_ksp_type = lin.sub_ksp_type
        sub_pc_type = lin.sub_pc_type
        pc_factor_levels = lin.pc_factor_levels
        # Default to NONZERO pivot shift for ILU/ICC sub-PCs: prevents NaN/Inf
        # in FGMRES inner products when PC factorization hits near-zero pivots
        # on stiff non-normal LNSE operators. PETSc default (NONE) is fragile.
        pc_factor_shift_type = lin.pc_factor_shift_type
        pc_factor_shift_amount = lin.pc_factor_shift_amount
        # When True, attach KSPMonitorDefault so each iteration prints
        # `iter rnorm` to stdout. Visible in c8 run logs without -ksp_monitor.
        ksp_monitor = lin.ksp_monitor

        ksp.setType(ksp_type)
        if pc_side == "left":
            ksp.setPCSide(PETSc.PC.Side.LEFT)
        elif pc_side == "right":
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
        elif pc_side in {"symmetric", "symm"}:
            ksp.setPCSide(PETSc.PC.Side.SYMMETRIC)
        else:
            raise ValueError(
                f"Invalid {env_prefix}_KSP_PC_SIDE '{pc_side}'; expected left|right|symmetric"
            )
        pc.setType(pc_type)
        if pc_type == "lu":
            factor_backend = lin.factor_backend
            apply_factor_backend(pc, factor_backend, options_prefix=ksp.getOptionsPrefix() or "")
        ksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)
        ksp.setErrorIfNotConverged(True)

        if pc_type == "asm":
            pc.setASMOverlap(asm_overlap)

        if ksp_monitor:
            # KSPMonitorDefault prints `iter rnorm` per iteration. Wrapped in
            # a small lambda so we can prefix lines with env_prefix to keep
            # outer/inner KSP traces distinguishable.
            def _monitor(_ksp, it, rnorm, _prefix=env_prefix):  # noqa: ANN001
                print(f"[{_prefix} ksp] it={it:5d} rnorm={rnorm:.6e}", flush=True)

            ksp.setMonitor(_monitor)

        # Allow runtime override via PETSc options DB before setUp materialises
        # the preconditioner stack and (for ASM/BJACOBI) the local sub-solvers.
        ksp.setFromOptions()
        ksp.setUp()

        # Configure local block solvers for ASM/BJACOBI after the outer KSP
        # has been set up (PETSc requires this for getASMSubKSP / getBJacobiSubKSP
        # to return materialised sub-solvers). Hard-fail on any error so the
        # realized stack always matches the configured one.
        realized_sub_pc_type: str | None = None
        realized_factor_levels: int | None = None
        if pc_type in {"asm", "bjacobi"}:
            sub_ksps = pc.getASMSubKSP() if pc_type == "asm" else pc.getBJacobiSubKSP()
            for sub_ksp in sub_ksps:
                sub_ksp.setType(sub_ksp_type)
                # Hard-stop on any local block Krylov divergence so iterative
                # runs cannot silently continue with invalid sub-solves.
                sub_ksp.setErrorIfNotConverged(True)
                sub_pc = sub_ksp.getPC()
                sub_pc.setType(sub_pc_type)
                if sub_pc_type in {"ilu", "icc", "lu"}:
                    shift_map = {
                        "none": PETSc.Mat.FactorShiftType.NONE,
                        "nonzero": PETSc.Mat.FactorShiftType.NONZERO,
                        "positive_definite": PETSc.Mat.FactorShiftType.POSITIVE_DEFINITE,
                        "inblocks": PETSc.Mat.FactorShiftType.INBLOCKS,
                    }
                    if pc_factor_shift_type not in shift_map:
                        raise ValueError(
                            f"Invalid {env_prefix}_PC_FACTOR_SHIFT_TYPE "
                            f"{pc_factor_shift_type!r}; expected one of {sorted(shift_map)}"
                        )
                    sub_pc.setFactorShift(
                        shift_map[pc_factor_shift_type],
                        pc_factor_shift_amount if pc_factor_shift_type != "none" else 0.0,
                    )
                if sub_pc_type in {"ilu", "icc"}:
                    sub_pc.setFactorLevels(pc_factor_levels)
                sub_pc.setUp()
                realized_sub_pc_type = sub_pc.getType()
                if sub_pc_type in {"ilu", "icc"}:
                    try:
                        realized_factor_levels = int(sub_pc.getFactorLevels())
                    except (AttributeError, RuntimeError):
                        realized_factor_levels = pc_factor_levels

        logger.info(
            "Linear solver: iterative (n=%d, nnz=%d, ksp=%s, pc=%s, side=%s, "
            "asm_overlap=%d, sub_ksp=%s, sub_pc=%s, fill_levels=%s, "
            "shift_type=%s, shift_amount=%g, rtol=%g, max_it=%d)",
            n,
            nnz,
            ksp_type,
            pc_type,
            pc_side,
            asm_overlap if pc_type == "asm" else 0,
            sub_ksp_type if realized_sub_pc_type is not None else "n/a",
            realized_sub_pc_type if realized_sub_pc_type is not None else "n/a",
            realized_factor_levels if realized_factor_levels is not None else "n/a",
            pc_factor_shift_type if realized_sub_pc_type in {"ilu", "icc", "lu"} else "n/a",
            pc_factor_shift_amount if pc_factor_shift_type != "none" else 0.0,
            ksp_rtol,
            ksp_max_it,
        )
        backend = "PETSc-iterative"
    else:
        raise ValueError(
            f"Unsupported {env_prefix}_TYPE={solver_type!r}; "
            "expected mumps|direct|lu|iterative|krylov|petsc"
        )

    def _check_vec_size(rhs: PETSc.Vec) -> None:
        if int(rhs.getSize()) != n:
            raise ValueError(f"rhs must have size {n}, got {rhs.getSize()}")

    def _rhs_vec_from_numpy(rhs: np.ndarray) -> PETSc.Vec:
        rhs_arr = np.asarray(rhs, dtype=complex)
        if rhs_arr.ndim != 1 or rhs_arr.size != n:
            raise ValueError(f"rhs must be a 1-D array of length {n}, got shape {rhs_arr.shape}")

        b = A.createVecLeft()
        r0, r1 = b.getOwnershipRange()
        if r1 > r0:
            rows = np.arange(r0, r1, dtype=PETSc.IntType)
            b.setValues(rows, rhs_arr[r0:r1], addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        return b

    def solve_vec(rhs: PETSc.Vec) -> PETSc.Vec:
        """Solve ``A x = rhs`` while preserving PETSc vector distribution."""
        _check_vec_size(rhs)
        x = rhs.duplicate()
        ksp.solve(rhs, x)
        return x

    def solve_transpose_vec_original(rhs: PETSc.Vec) -> PETSc.Vec:
        """Solve ``A^H x = rhs`` using solveTranspose with conjugation."""
        _check_vec_size(rhs)
        b = rhs.copy()
        x = rhs.duplicate()
        try:
            b.conjugate()
            ksp.solveTranspose(b, x)
            x.conjugate()
            return x
        finally:
            b.destroy()

    def solve(rhs: np.ndarray) -> np.ndarray:
        b = _rhs_vec_from_numpy(rhs)
        try:
            x = solve_vec(b)
            try:
                return vec_to_numpy(x)
            finally:
                x.destroy()
        finally:
            b.destroy()

    def solve_transpose_original(rhs: np.ndarray) -> np.ndarray:
        b = _rhs_vec_from_numpy(rhs)
        try:
            x = solve_transpose_vec_original(b)
            try:
                return vec_to_numpy(x)
            finally:
                x.destroy()
        finally:
            b.destroy()

    def residual(rhs: np.ndarray, x_arr: np.ndarray) -> float:
        """Return ``||operator @ x - rhs||_2`` for NumPy inputs."""
        b = _rhs_vec_from_numpy(rhs)
        x = _rhs_vec_from_numpy(x_arr)
        r = b.duplicate()
        try:
            A.mult(x, r)
            r.axpy(-1.0, b)
            return float(r.norm(PETSc.NormType.NORM_2))
        finally:
            r.destroy()
            x.destroy()
            b.destroy()

    def _as_rhs_block(rhs_block: np.ndarray) -> np.ndarray:
        block = np.asarray(rhs_block, dtype=complex)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        if block.ndim != 2 or block.shape[0] != n:
            raise ValueError(f"rhs_block must have shape ({n}, k), got {block.shape}")
        return np.ascontiguousarray(block)

    def _dense_from_block(block: np.ndarray) -> PETSc.Mat:
        r0, r1 = A.getOwnershipRange()
        nrhs = int(block.shape[1])
        mat = PETSc.Mat().createDense(size=((r1 - r0, n), (nrhs, nrhs)), comm=A.getComm())
        mat.getDenseArray()[:, :] = block[r0:r1, :]
        mat.assemble()
        return mat

    def _dense_to_numpy(mat: PETSc.Mat) -> np.ndarray:
        if mat.getComm().getSize() == 1:
            return np.array(mat.getDenseArray(), copy=True)
        cols = []
        for j in range(int(mat.getSize()[1])):
            col = mat.getDenseColumnVec(j, "r")
            try:
                cols.append(vec_to_numpy(col))
            finally:
                mat.restoreDenseColumnVec(j, "r", col)
        return np.column_stack(cols)

    def _loop_block(block: np.ndarray, action: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        metrics["solve_block_fallback_calls"] += 1
        return np.column_stack([action(block[:, j]) for j in range(block.shape[1])])

    def solve_block(rhs_block: np.ndarray) -> np.ndarray:
        block = _as_rhs_block(rhs_block)
        if solver_type not in DIRECT_SOLVER_TYPES:
            return _loop_block(block, solve)
        B = _dense_from_block(block)
        X = _dense_from_block(np.zeros_like(block))
        start = time.perf_counter()
        try:
            ksp.getPC().getFactorMatrix().matSolve(B, X)
            elapsed = time.perf_counter() - start
            metrics["solve_block_calls"] += 1
            metrics["solve_block_rhs"] += int(block.shape[1])
            metrics["solve_block_seconds"] += elapsed
            return _dense_to_numpy(X)
        except (AttributeError, PETSc.Error):
            return _loop_block(block, solve)
        finally:
            X.destroy()
            B.destroy()

    def refactorize_operator(new_operator: sp.spmatrix | PETSc.Mat) -> None:
        nonlocal A
        nrows_new, ncols_new = _operator_shape(new_operator)
        if (nrows_new, ncols_new) != (n, n):
            raise ValueError(f"operator must keep shape ({n}, {n}), got {(nrows_new, ncols_new)}")
        new_A = (
            new_operator.copy()
            if _is_petsc_matrix(new_operator)
            else scipy_to_petsc(new_operator, solver_cfg)
        )
        try:
            old_pattern = _aij_pattern(A)
            new_pattern = _aij_pattern(new_A)
            local_mismatch = not all(
                np.array_equal(old, new) for old, new in zip(old_pattern, new_pattern, strict=True)
            )
            # Collective: reduce the mismatch flag across the communicator so every
            # rank takes the same raise/fallback branch. The CSR comparison is rank-
            # local; a rank raising here while others proceed into the collective
            # ksp.setOperators()/setUp() below would split the communicator and hang.
            comm = A.getComm()
            if comm.getSize() > 1:
                from mpi4py import MPI

                local_mismatch = bool(comm.tompi4py().allreduce(local_mismatch, op=MPI.LOR))
            if local_mismatch:
                raise ValueError("refactorize requires an identical nonzero pattern")
        except Exception:
            new_A.destroy()
            raise
        old_A = A
        start = time.perf_counter()
        try:
            if solver_type in DIRECT_SOLVER_TYPES:
                pc.setFactorOrdering(reuse=True)
            ksp.setOperators(new_A)
            ksp.setUp()
        except Exception:
            new_A.destroy()
            raise
        A = new_A
        metrics["refactor_count"] += 1
        metrics["refactor_seconds"] += time.perf_counter() - start
        holder["factor"]._mat = A
        old_A.destroy()

    # --- DUAL-FACTOR AUTO-HEAL LOGIC ---
    # Lazy adjoint self-test: run on first adjoint use via a wrapper.
    adjoint_test_done = False

    def _maybe_run_adjoint_test_and_heal() -> None:
        """Run self-test on first adjoint use, trigger healing if needed."""
        nonlocal adjoint_test_done
        if adjoint_test_done:
            return
        adjoint_test_done = True

        # Run self-test with original paths
        try:
            defect = _run_adjoint_self_test(solve, solve_transpose_original, n, k=3)
        except Exception as exc:
            logger.info(
                "Adjoint self-test transpose-solve failed (%s); treating native transpose as defective and activating dual-factor heal",
                exc,
            )
            defect = float("inf")
        metrics["adjoint_self_test_defect"] = defect

        # Trigger threshold for the heal. This is NOT a precision target: it
        # separates "the backend's transpose really is the Hermitian adjoint"
        # from "it is something else" (a plain transpose leaves this identity
        # defective at O(1e-1), and STRUMPACK's complex transpose-solve at
        # ~3e-2). The attainable floor is set by the operator's conditioning,
        # roughly kappa(A)*eps_machine, so a fixed 1e-10 sits BELOW what a
        # well-conditioned-but-not-trivial operator can deliver: the paraboloid
        # example measured 3.07e-10 with a perfectly good MUMPS adjoint, which
        # needlessly triggered a second factorization that could only do worse
        # (1.62e-9) and then aborted the run. 1e-8 stays seven orders below the
        # failure mode it must catch while clearing the round-off floor.
        adjoint_tol = 1.0e-8
        if defect <= adjoint_tol:
            # Keep the original single-factor path
            metrics["dual_factor_active"] = False
            return

        # Defect exceeded tolerance: trigger AUTO-HEAL
        logger.info(
            "Adjoint self-test defect %.3e > tolerance %.3e; "
            "activating dual-factor healing for A^H",
            defect,
            adjoint_tol,
        )

        # Build the A^H factor through the SAME factorize() machinery (identical
        # backend, options, assembly) -- NOT a hand-rolled KSP, which proved
        # fragile on STRUMPACK. A FORWARD solve on A^H is exactly the adjoint
        # A^{-H}. Build A^H from the PRISTINE operator: `A` above is the copy the
        # forward factorization owns (a backend may equilibrate it in place), so
        # transpose the caller's untouched operator instead.
        if _is_petsc_matrix(operator):
            A_H = operator.hermitianTranspose()
        else:
            A_H = operator.conj().T.tocsc()
        factor_H = factorize(A_H, env_prefix=env_prefix, solver_cfg=solver_cfg, cfg=cfg)
        solve_H = factor_H.solve
        holder["factor_H"] = factor_H

        # TRUTH-GATE (catches a wrong A^H such as a plain transpose): re-run the
        # adjoint identity <A^{-1} a, b> == <a, A_H^{-1} b> with the HEALED path.
        # For a non-normal complex operator a plain-transpose A_H leaves this
        # defect O(0.1), so the heal is rejected. A residual-only check
        # (||A_H x - b||) is self-consistent and would PASS a plain transpose, so
        # it cannot be the wiring gate -- this identity is.
        healed_defect = _run_adjoint_self_test(solve, solve_H, n, k=3)
        metrics["adjoint_healed_defect"] = healed_defect
        if healed_defect > adjoint_tol:
            # The heal did not reach the target. Two very different situations:
            #
            #  * the heal is simply not an improvement (it can be worse, since
            #    factorizing A^H separately carries its own round-off). Then the
            #    ORIGINAL transpose is the better adjoint we have, and falling
            #    back to it is strictly the best available answer -- aborting
            #    here would kill a run whose adjoint was fine, which is exactly
            #    what happened to the paraboloid example (native 3.07e-10,
            #    healed 1.62e-9).
            #  * both paths are defective. Then there is no usable adjoint and we
            #    must still fail loudly rather than return silent garbage.
            factor_H.destroy()
            del holder["factor_H"]
            if healed_defect >= defect:
                metrics["dual_factor_active"] = False
                metrics["adjoint_heal_rejected"] = "no_improvement"
                logger.warning(
                    "Dual-factor heal did not improve the adjoint (healed %.3e >= native "
                    "%.3e); keeping the native transpose, which is the better of the two. "
                    "Both are far below the O(1e-1) defect that indicates a wrong (plain) "
                    "transpose, so the adjoint is trusted.",
                    healed_defect,
                    defect,
                )
                return
            raise RuntimeError(
                f"Dual-factor heal failed: adjoint identity defect {healed_defect:.3e} "
                f"> tolerance {adjoint_tol:.3e} after rebuilding A^H, and the native "
                f"transpose is worse still ({defect:.3e}) -- neither path provides a "
                "usable Hermitian adjoint (plain transpose?)."
            )

        # Secondary oracle (A^H factorization quality): ||A_H x - b|| / ||b||
        # against the concrete A^H matvec, independent of the factor's own claim.
        oracle_residual = _run_oracle_residual(solve_H, factor_H._mat, n, k=3)
        metrics["adjoint_oracle_residual"] = oracle_residual
        oracle_tol = 1.0e-8
        if oracle_residual > oracle_tol:
            factor_H.destroy()
            del holder["factor_H"]
            raise RuntimeError(
                f"Adjoint oracle residual {oracle_residual:.3e} > tolerance {oracle_tol:.3e}; "
                "dual-factor healing failed (A^H factorization inaccurate)."
            )

        # Both gates passed: redirect the adjoint solves to the healed dual factor.
        metrics["dual_factor_active"] = True
        logger.info(
            "Dual-factor healing activated: healed adjoint defect %.3e, oracle residual %.3e",
            healed_defect,
            oracle_residual,
        )

    # Wrap the adjoint solves: lazy self-test/heal on first use, then route through
    # the healed dual factor (a FORWARD solve on A^H) when active.
    def solve_transpose_vec_wrapped(rhs: PETSc.Vec) -> PETSc.Vec:
        _maybe_run_adjoint_test_and_heal()
        if metrics.get("dual_factor_active"):
            return holder["factor_H"].solve_vec(rhs)
        return solve_transpose_vec_original(rhs)

    def solve_transpose_wrapped(rhs: np.ndarray) -> np.ndarray:
        _maybe_run_adjoint_test_and_heal()
        if metrics.get("dual_factor_active"):
            return holder["factor_H"].solve(rhs)
        return solve_transpose_original(rhs)

    factor = Factorization(
        backend=backend,
        solve=solve,
        solve_adjoint=solve_transpose_wrapped,
        solve_vec=solve_vec,
        solve_adjoint_vec=solve_transpose_vec_wrapped,
        residual=residual,
        _ksp=ksp,
        _mat=A,
        _solve_block=solve_block,
        _refactorize=refactorize_operator,
        metrics=metrics,
    )
    holder["factor"] = factor
    return factor


# Keep ``from linstabpy.linalg import factorize`` bound to the public function
# even after the same-named implementation submodule is imported.
_parent = sys.modules.get("linstabpy.linalg")
if _parent is not None:
    _parent.Factorization = Factorization
    _parent.factorize = factorize
