# SPDX-License-Identifier: GPL-3.0-only
"""Reusable PETSc factorizations, direct by default.

Provides a factorize-once, solve-many interface for the LNSE shift-invert
workflows.
The default path is KSPPREONLY + PCLU + MUMPS.
The release tests and reference examples use exact direct solves.
The iterative path is available for large memory-bound cases.
It is an explicit solver-policy choice rather than an automatic fallback.
"""

from __future__ import annotations

import logging
import os
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

# Iteration cap for the adjoint self-test's own solves on an iterative KSP (see
# _maybe_run_adjoint_test_and_heal). Forward FGMRES around the STRUMPACK LU needs
# 2-3 iterations on the Savarino operator; 50 leaves a wide margin.
ADJOINT_SELF_TEST_MAX_IT = 50

if TYPE_CHECKING:
    # Type-only import; ``from __future__ import annotations`` keeps every
    # ``PETSc.*`` annotation below a string, so importing this module does not
    # pull in petsc4py or initialize PETSc.
    from petsc4py import PETSc  # noqa: F401

    from .cudss_factor import CudssSolver

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


def require_same_pattern(
    pattern: tuple[np.ndarray, np.ndarray],
    indptr: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Raise if the CSR structure differs from the factored pattern.

    A reused factorization keeps the symbolic phase of the first operator.
    A solve would then use the structure of that first operator.
    A second operator with another nonzero pattern must therefore raise.
    """
    same = np.array_equal(pattern[0], np.asarray(indptr, dtype=np.int64)) and np.array_equal(
        pattern[1], np.asarray(indices, dtype=np.int64)
    )
    if not same:
        raise ValueError("refactorize requires an identical nonzero pattern")


def hermitian_transpose_plan(
    indptr: np.ndarray,
    indices: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the permutation and CSR structure of the conjugate transpose.

    The structure of K^H depends only on the nonzero pattern of K.
    The values of K^H are ``conj(values[perm])`` in the returned structure.
    A refactorize with a fixed pattern can therefore skip the sparse transpose.
    """
    nnz = int(indptr[-1])
    order = sp.csr_matrix((np.arange(nnz, dtype=np.int64), indices, indptr), shape=(n, n))
    order_h = order.T.tocsr()
    order_h.sort_indices()
    return order_h.data.astype(np.int64), order_h.indptr, order_h.indices


def _run_adjoint_self_test(
    solve_func: Callable[[np.ndarray], np.ndarray],
    solve_adjoint_func: Callable[[np.ndarray], np.ndarray],
    n: int,
    k: int = 3,
    probe_family: Callable[[np.random.RandomState], np.ndarray] | None = None,
) -> float:
    """Run k random complex probe pairs to detect adjoint defect.

    Returns max defect over all probes:
      defect_i = |<solve(a_i), b_i> - <a_i, solve_adjoint(b_i)>|
                 / max(|<solve(a_i), b_i>|, |<a_i, solve_adjoint(b_i)>|, 1e-300)

    ``probe_family``, when given, draws each probe from the production
    right-hand-side family instead of dense random vectors. A
    single-precision cuDSS factor preconditions only that family; dense
    probes stall its FGMRES and would abort a healthy configuration.
    """
    max_defect = 0.0
    rng = np.random.RandomState(42)  # deterministic for reproducibility

    for _ in range(k):
        if probe_family is None:
            a = rng.randn(n) + 1j * rng.randn(n)
            b = rng.randn(n) + 1j * rng.randn(n)
        else:
            a = probe_family(rng)
            b = probe_family(rng)

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
    probe_family: Callable[[np.random.RandomState], np.ndarray] | None = None,
) -> float:
    """Run k random RHS through the healed adjoint and measure ||A_H @ x - b|| / ||b||.

    The check uses A_H (the stored Hermitian transpose).
    It validates that the healed solve_adjoint path (a forward-solve
    on A_H) produces correct results.
    """
    PETSc = _get_petsc()
    max_residual = 0.0
    rng = np.random.RandomState(43)  # different seed for the A_H residual check

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
        if probe_family is None:
            b_arr = rng.randn(n) + 1j * rng.randn(n)
        else:
            b_arr = probe_family(rng)
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
    # Production-family probe generator for the adjoint self-test; set by
    # ResolventOperator when a forcing mask exists. Used only where dense
    # probes sit outside the validated envelope (single-precision cuDSS).
    probe_family: Callable[[np.random.RandomState], np.ndarray] | None = None
    _cudss: CudssSolver | None = None

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
        if self._cudss is not None:
            self._cudss.free()
            self._cudss = None


class _CudssShellPC:
    """petsc4py Python PC context: one cuDSS solve per apply.

    No ``applyTranspose`` on purpose: cuDSS has no transpose or adjoint
    solve, so the adjoint must ride a dedicated K^H factor (the dual-factor
    path in ``factorize``). A missing method fails loudly instead of
    silently solving the wrong system.
    """

    def __init__(self, solver: CudssSolver) -> None:
        self.solver = solver

    def apply(self, pc, x, y) -> None:  # noqa: ANN001, ARG002 -- PETSc PCShell signature
        y.setArray(self.solver.solve(x.getArray(readonly=True)))


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
        Optional solver config dict (from YAML ``"solver"`` section).
        When provided, ``env_prefix`` picks the subsection:
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
    metrics["adjoint_refactor_count"] = 0
    holder: dict[str, object] = {}
    # The caller's untouched operator. The dual-factor heal transposes THIS (not
    # ``A``, the copy the forward factorization owns and a backend may equilibrate
    # in place). ``refactorize_operator`` rebinds it, so a heal that first runs
    # after a refactorize builds K^H of the operator currently factored.
    pristine_operator = operator
    # Trigger threshold for the dual-factor heal and the bound every healed
    # adjoint must meet, including after each refactorize. This is NOT a
    # precision target: it separates "the backend's transpose really is the
    # Hermitian adjoint" from "it is something else" (a plain transpose leaves
    # the adjoint identity defective at O(1e-1); the STRUMPACK path measured
    # ~3e-2 -- traced to PETSc's MatSolveTranspose, which finds no registered
    # solvetranspose for STRUMPACK and silently substitutes the FORWARD solve
    # because the factor's symmetry flag "unknown" is treated as true; the
    # defect is the interface fallback, not STRUMPACK's arithmetic). The attainable floor is set by the operator's
    # conditioning, roughly kappa(A)*eps_machine, so a fixed 1e-10 sits BELOW
    # what a well-conditioned-but-not-trivial operator can deliver: the
    # paraboloid example measured 3.07e-10 with a perfectly good MUMPS adjoint,
    # which needlessly triggered a second factorization that could only do worse
    # (1.62e-9) and then aborted the run. 1e-8 stays seven orders below the
    # failure mode it must catch while clearing the round-off floor.
    adjoint_tol = 1.0e-8

    def _hermitian_transpose_of(op: sp.spmatrix | PETSc.Mat) -> sp.spmatrix | PETSc.Mat:
        # Same recipe for the first heal and for every refactorize, so the K^H
        # factor sees one nonzero pattern for the life of the factorization.
        # petsc4py's Mat.hermitianTranspose(out=None) transposes IN PLACE and
        # returns the same object: the previous heal turned the CALLER's operator
        # into K^H (measured 2026-08-19, c9 job 2386: K x - b residual 1e10 after
        # the heal, 1e-6 before, ||A - K||_F = 0). An empty Mat as `out` asks for
        # a new matrix.
        if _is_petsc_matrix(op):
            return op.hermitianTranspose(out=PETSc.Mat())
        return op.conj().T.tocsc()

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
    # Set by the iterative cuDSS route below; ``refactorize_operator`` and the
    # adjoint self-test read it. None on every other path.
    cudss_solver: CudssSolver | None = None

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
        cudss_requested = (
            pc_type == "lu" and str(lin.factor_backend or "").strip().lower() == "cudss"
        )
        if cudss_requested:
            # cuDSS is not a PETSc factor type. The LU lives in a Python
            # preconditioner (one cuDSS solve per apply) around this Krylov
            # iteration; the outer iteration owns the tolerance. Measured on
            # the 450k-DOF SWBLI operator (RTX A6000): factor 5.7-9.0 s vs
            # 41-55 s STRUMPACK on the same GPU, FGMRES 1-2 iterations,
            # solution matches MUMPS to 1.2e-12.
            if A.getComm().getSize() > 1:
                raise ValueError(
                    "factor_backend='cudss' is a one-rank GPU backend; "
                    "run without MPI parallelism or use mumps/strumpack"
                )
            from .cudss_factor import CudssSolver as _CudssSolver

            indptr_csr, indices_csr, values_csr = A.getValuesCSR()
            cudss_solver = _CudssSolver(
                sp.csr_matrix((values_csr, indices_csr, indptr_csr), shape=(n, n)),
                precision=lin.factor_precision,
            )
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(_CudssShellPC(cudss_solver))
            metrics["cudss_precision"] = lin.factor_precision
            metrics["cudss_plan_seconds"] = cudss_solver.plan_seconds
            metrics["cudss_factor_seconds"] = cudss_solver.factor_seconds
            if cudss_solver.lu_nnz is not None:
                metrics["cudss_lu_nnz"] = cudss_solver.lu_nnz
            logger.info(
                "Linear solver: %s + cuDSS %s-precision LU preconditioner (n=%d, nnz=%d)",
                ksp_type,
                lin.factor_precision,
                n,
                nnz,
            )
        else:
            pc.setType(pc_type)
            if pc_type == "lu":
                factor_backend = lin.factor_backend
                apply_factor_backend(
                    pc, factor_backend, options_prefix=ksp.getOptionsPrefix() or ""
                )
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
        # setUp (PETSc requires this for getASMSubKSP / getBJacobiSubKSP
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
        backend = "cudss-fgmres" if cudss_solver is not None else "PETSc-iterative"
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

    def _raise_if_diverged(what: str) -> None:
        # PETSc returns from KSPSolve whatever the last iterate was, converged or not
        # (unless -ksp_error_if_not_converged is set). A diverged FGMRES around a
        # sparse LU must fail here, not appear as a silently inaccurate resolvent mode.
        reason = int(ksp.getConvergedReason())
        if reason < 0:
            raise RuntimeError(
                f"{what}: KSP diverged (reason {reason}, {ksp.getIterationNumber()} iterations, "
                f"residual norm {ksp.getResidualNorm():.3e})"
            )

    def solve_vec(rhs: PETSc.Vec) -> PETSc.Vec:
        """Solve ``A x = rhs`` while preserving PETSc vector distribution."""
        _check_vec_size(rhs)
        x = rhs.duplicate()
        ksp.solve(rhs, x)
        _raise_if_diverged("forward solve")
        return x

    def solve_transpose_vec_original(rhs: PETSc.Vec) -> PETSc.Vec:
        """Solve ``A^H x = rhs`` using solveTranspose with conjugation."""
        _check_vec_size(rhs)
        b = rhs.copy()
        x = rhs.duplicate()
        try:
            b.conjugate()
            ksp.solveTranspose(b, x)
            _raise_if_diverged("transpose solve")
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

    # The in-place path is the default for serial cuDSS. Set
    # LINSTABPY_FAST_REFACTOR=0 to use the general path instead.
    _fast_refactor = os.environ.get("LINSTABPY_FAST_REFACTOR", "1") != "0"

    def _fast_refactorize(
        new_operator: sp.spmatrix | PETSc.Mat,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refactorize from one CSR extraction, serial cuDSS only.

        The CSR triple comes out of the new operator once.
        The nonzero pattern is compared against the cached pattern of ``A``.
        The values then go into the same Mat in place, so PETSc keeps the
        symbolic factorization and cuDSS keeps its analysis.
        """
        t0 = time.perf_counter()
        if _is_petsc_matrix(new_operator):
            indptr, indices, values = new_operator.getValuesCSR()
        else:
            csr = sp.csr_matrix(new_operator)
            csr.sort_indices()
            indptr, indices, values = csr.indptr, csr.indices, csr.data
        pattern = holder.get("_pattern")
        if pattern is None:
            pattern = _aij_pattern(A)
            holder["_pattern"] = pattern
        require_same_pattern(pattern, indptr, indices)
        A.setValuesCSR(
            np.asarray(indptr, dtype=PETSc.IntType),
            np.asarray(indices, dtype=PETSc.IntType),
            np.asarray(values, dtype=PETSc.ScalarType),
        )
        A.assemblyBegin()
        A.assemblyEnd()
        cudss_solver.refactorize(sp.csr_matrix((values, indices, indptr), shape=(n, n)))
        ksp.setOperators(A)
        ksp.setUp()
        metrics["refactor_count"] += 1
        metrics["refactor_seconds"] += time.perf_counter() - t0
        return indptr, indices, values

    def refactorize_operator(new_operator: sp.spmatrix | PETSc.Mat) -> None:
        nonlocal pristine_operator
        nrows_new, ncols_new = _operator_shape(new_operator)
        if (nrows_new, ncols_new) != (n, n):
            raise ValueError(f"operator must keep shape ({n}, {n}), got {(nrows_new, ncols_new)}")
        if _fast_refactor and cudss_solver is not None and A.getComm().getSize() == 1:
            indptr, indices, values = _fast_refactorize(new_operator)
            pristine_operator = new_operator
            factor_H = holder.get("factor_H")
            if factor_H is not None:
                plan = holder.get("_transpose_plan")
                if plan is None:
                    plan = hermitian_transpose_plan(indptr, indices, n)
                    holder["_transpose_plan"] = plan
                    logger.info("refactorize: in-place path for serial cuDSS (K and K^H)")
                perm, indptr_h, indices_h = plan
                k_h = sp.csr_matrix(
                    (np.conj(np.asarray(values)[perm]), indices_h, indptr_h), shape=(n, n)
                )
                factor_H.refactorize(k_h)
                metrics["adjoint_refactor_count"] += 1
                healed_defect = _run_adjoint_self_test(
                    solve, factor_H.solve, n, k=1, probe_family=_self_test_probe_family()
                )
                metrics["adjoint_healed_defect"] = healed_defect
                if healed_defect > adjoint_tol:
                    raise RuntimeError(
                        f"Dual-factor adjoint after fast refactorize: identity defect {healed_defect:.3e} "
                        f"> tolerance {adjoint_tol:.3e}"
                    )
            return
        owns_new_A = not _is_petsc_matrix(new_operator)
        new_A = new_operator if not owns_new_A else scipy_to_petsc(new_operator, solver_cfg)
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
            if owns_new_A:
                new_A.destroy()
            raise
        start = time.perf_counter()
        try:
            # Copy the new values into the Mat the KSP already holds. Same object,
            # same nonzero pattern -> PETSc/MUMPS keeps the symbolic factorization
            # (ordering + analysis) and redoes only the numeric phase. Handing the
            # KSP a new Mat object redoes the analysis every call, even with
            # reuse-ordering: measured 2026-08-18 on the Savarino 450k-DOF operator
            # (c9, MKL) 90.0 s new Mat, 80.9 s new Mat + reuse-ordering, 74.9 s
            # values copied into the same Mat -- the 9 s analysis, per point.
            new_A.copy(result=A, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            if cudss_solver is not None:
                # The Python PC wraps its own device factor; the values copy
                # above only feeds the FGMRES matvec. Push the new values to
                # the device and redo the numeric factorization on the kept
                # analysis (measured 9.0-9.2 s at 450k DOF; a fresh plan
                # would add 3-8 s per point).
                indptr_new, indices_new, values_new = A.getValuesCSR()
                cudss_solver.refactorize(
                    sp.csr_matrix((values_new, indices_new, indptr_new), shape=(n, n))
                )
            ksp.setOperators(A)
            ksp.setUp()
        finally:
            if owns_new_A:
                new_A.destroy()
        metrics["refactor_count"] += 1
        metrics["refactor_seconds"] += time.perf_counter() - start
        # The adjoint must follow the operator. With the dual-factor heal active
        # (STRUMPACK: PETSc's MatSolveTranspose silently substitutes the forward
        # solve -- no registered solvetranspose -- so the true adjoint must be a
        # FORWARD solve on a separate K^H factor) a refactorize that
        # left ``factor_H`` alone would forward-solve K_new and adjoint-solve
        # K_old^H from the second operator on -- silently. Refactorize K^H from
        # the new operator with the heal's own recipe, then re-run the adjoint
        # identity: a stale or mismatched K^H fails it at O(1) and must fail loudly.
        pristine_operator = new_operator
        factor_H = holder.get("factor_H")
        if factor_H is not None:
            A_H_new = _hermitian_transpose_of(new_operator)
            try:
                factor_H.refactorize(A_H_new)
            finally:
                if _is_petsc_matrix(A_H_new):
                    A_H_new.destroy()
            metrics["adjoint_refactor_count"] += 1
            healed_defect = _run_adjoint_self_test(
                solve, factor_H.solve, n, k=3, probe_family=_self_test_probe_family()
            )
            metrics["adjoint_healed_defect"] = healed_defect
            if healed_defect > adjoint_tol:
                raise RuntimeError(
                    f"Dual-factor adjoint after refactorize: identity defect {healed_defect:.3e} "
                    f"> tolerance {adjoint_tol:.3e} -- the healed K^H factor does not match "
                    "the refactorized operator."
                )

    def _self_test_probe_family() -> Callable[[np.random.RandomState], np.ndarray] | None:
        """Return production-family probes where dense probes are invalid.

        A single-precision cuDSS factor preconditions the production
        (forcing-masked) right-hand-side family, not dense random vectors:
        dense probes excite the boundary-penalty rows and stall FGMRES on a
        configuration that is healthy on every RHS the resolvent produces
        (measured on the 450k-DOF SWBLI operator). Every other path keeps
        the stricter dense probes, unchanged.
        """
        if cudss_solver is None or lin.factor_precision != "single":
            return None
        factor_obj = holder.get("factor")
        return getattr(factor_obj, "probe_family", None)

    # --- DUAL-FACTOR AUTO-HEAL LOGIC ---
    # Lazy adjoint self-test: run on first adjoint use via a wrapper.
    adjoint_test_done = False

    def _maybe_run_adjoint_test_and_heal() -> None:
        """Run self-test on first adjoint use, trigger healing if needed."""
        nonlocal adjoint_test_done
        if adjoint_test_done:
            return
        adjoint_test_done = True

        if cudss_solver is not None:
            # cuDSS has no transpose or adjoint solve at all (cuDSS 0.8:
            # CUDSS_CONFIG_SOLVE_MODE accepts only its default value), so
            # there is no native path worth testing: go straight to the
            # dedicated K^H factor.
            defect = float("inf")
            logger.info(
                "cuDSS has no transpose solve; building a dedicated K^H factor for the adjoint"
            )
        else:
            # Run self-test with original paths. On an iterative KSP a defective native
            # transpose (STRUMPACK) sends the transposed FGMRES to its full iteration cap
            # before the identity can fail -- measured 400 iterations, 276 s of a 437 s
            # point (c9 job 2381). A correct adjoint of an LU preconditioner converges in a
            # handful of iterations, so the self-test's own solves get a short cap and the
            # call restores production tolerances right after.
            rtol0, atol0, dtol0, max_it0 = ksp.getTolerances()
            capped = (
                ksp.getType() != PETSc.KSP.Type.PREONLY
                and int(max_it0) > ADJOINT_SELF_TEST_MAX_IT
            )
            if capped:
                ksp.setTolerances(max_it=ADJOINT_SELF_TEST_MAX_IT)
            try:
                defect = _run_adjoint_self_test(solve, solve_transpose_original, n, k=3)
            except Exception as exc:
                logger.info(
                    "Adjoint self-test transpose-solve failed (%s); treating native transpose as defective and activating dual-factor heal",
                    exc,
                )
                defect = float("inf")
            finally:
                if capped:
                    ksp.setTolerances(rtol=rtol0, atol=atol0, divtol=dtol0, max_it=max_it0)
        metrics["adjoint_self_test_defect"] = defect

        # ``adjoint_tol`` (closure scope, above): trigger threshold for the heal.
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

        # Fail before paying for the K^H factor: a single-precision cuDSS
        # factor cannot pass dense-probe gates (measured), so without the
        # production probe family the heal would build the second factor and
        # then stall in its own self-test.
        probe_family = _self_test_probe_family()
        if cudss_solver is not None and lin.factor_precision == "single" and probe_family is None:
            raise RuntimeError(
                "factor_precision='single' with the cudss backend validates its "
                "adjoint against the production right-hand-side family "
                "(forcing-masked probes), and none is attached (probe_family is "
                "unset). Dense random probes stall a single-precision factor by "
                "design. Run through the resolvent driver, which attaches the "
                "family, or use factor_precision='double'."
            )

        # Build the A^H factor through the SAME factorize() machinery (identical
        # backend, options, assembly) -- NOT a hand-rolled KSP, which proved
        # fragile on STRUMPACK. A FORWARD solve on A^H is exactly the adjoint
        # A^{-H}. Build A^H from the PRISTINE operator: `A` above is the copy the
        # forward factorization owns (a backend may equilibrate it in place), so
        # transpose the caller's untouched operator instead -- the one currently
        # factored, which ``refactorize_operator`` keeps up to date.
        A_H = _hermitian_transpose_of(pristine_operator)
        factor_H = factorize(A_H, env_prefix=env_prefix, solver_cfg=solver_cfg, cfg=cfg)
        solve_H = factor_H.solve
        holder["factor_H"] = factor_H

        # TRUTH-GATE (catches a mismatched A^H such as a plain transpose): re-run the
        # adjoint identity <A^{-1} a, b> == <a, A_H^{-1} b> with the HEALED path.
        # For a non-normal complex operator a plain-transpose A_H leaves this
        # defect O(0.1), so the identity rejects the heal. A residual-only check
        # (||A_H x - b||) is self-consistent and would PASS a plain transpose, so
        # it cannot be the wiring gate -- this identity is.
        healed_defect = _run_adjoint_self_test(solve, solve_H, n, k=3, probe_family=probe_family)
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

        # Secondary A_H residual check (factorization quality): ||A_H x - b|| / ||b||
        # against the concrete A^H matvec, independent of the factor's own claim.
        oracle_residual = _run_oracle_residual(
            solve_H, factor_H._mat, n, k=3, probe_family=probe_family
        )
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
        _cudss=cudss_solver,
    )
    holder["factor"] = factor
    return factor


# Keep ``from linstabpy.linalg import factorize`` bound to the public function
# even after the same-named implementation submodule loads.
_parent = sys.modules.get("linstabpy.linalg")
if _parent is not None:
    _parent.Factorization = Factorization
    _parent.factorize = factorize
