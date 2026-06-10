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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

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
from .backends import apply_factor_backend

logger = logging.getLogger(__name__)


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

    @property
    def solve_transpose(self) -> Callable[[np.ndarray], np.ndarray]:
        """Backward-compatible alias for :meth:`solve_adjoint`."""
        return self.solve_adjoint

    @property
    def solve_transpose_vec(self) -> Callable[[PETSc.Vec], PETSc.Vec]:
        """Backward-compatible alias for :meth:`solve_adjoint_vec`."""
        return self.solve_adjoint_vec

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

    def solve_transpose_vec(rhs: PETSc.Vec) -> PETSc.Vec:
        """Solve ``A^H x = rhs`` while preserving PETSc vector distribution."""
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

    def solve_transpose(rhs: np.ndarray) -> np.ndarray:
        b = _rhs_vec_from_numpy(rhs)
        try:
            x = solve_transpose_vec(b)
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

    return Factorization(
        backend=backend,
        solve=solve,
        solve_adjoint=solve_transpose,
        solve_vec=solve_vec,
        solve_adjoint_vec=solve_transpose_vec,
        residual=residual,
        _ksp=ksp,
        _mat=A,
    )


# Keep ``from linstabpy.linalg import factorize`` bound to the public function
# even after the same-named implementation submodule is imported.
_parent = sys.modules.get("linstabpy.linalg")
if _parent is not None:
    _parent.Factorization = Factorization
    _parent.factorize = factorize
