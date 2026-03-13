"""Direct linear solver via PETSc KSP with MUMPS.

Provides a factorize-once, solve-many interface for the LNSE shift-invert
workflows.  Both forward and conjugate-transpose solves reuse the same
MUMPS factorisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from .petsc_utils import require_complex, scipy_to_petsc, petsc_vec, vec_to_numpy
from ..utils._solver_cfg import _scfg, _scfg_float, _scfg_int

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearSolver:
    """Callable wrapper for forward and adjoint solves."""

    backend: str
    solve: Callable[[np.ndarray], np.ndarray]
    solve_transpose: Callable[[np.ndarray], np.ndarray]


def make_linear_solver(
    operator: sp.spmatrix,
    *,
    env_prefix: str = "LINSTAB2D_LINSOLVE",
    solver_cfg: dict | None = None,
) -> LinearSolver:
    """Build a reusable PETSc linear solver for ``operator``.

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
        ``LINSTAB2D_RESOLVENT``, ``"linsolve"`` otherwise. Keys in the
        subsection override the corresponding env vars.
    """
    require_complex(solver_cfg)

    A = scipy_to_petsc(operator, solver_cfg)
    n = operator.shape[0]

    # Resolve the config subsection for this linear-solver instance.
    _sub: dict | None = None
    if solver_cfg is not None:
        if env_prefix == "LINSTAB2D_RESOLVENT":
            _sub = solver_cfg.get("resolvent", solver_cfg.get("linsolve", {}))
        else:
            _sub = solver_cfg.get("linsolve", {})

    solver_type = _scfg(_sub, "type", f"{env_prefix}_TYPE", "mumps")

    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(A)
    pc = ksp.getPC()

    if solver_type in {"mumps", "direct", "lu"}:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        ksp.setUp()

        # Log MUMPS ordering method (one-time verification that METIS is active)
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
                operator.nnz,
                ordering,
                icntl7,
            )
        except (AttributeError, RuntimeError) as e:
            logger.debug("Could not query MUMPS ordering: %s", e)
            logger.info("Linear solver: MUMPS direct (n=%d, nnz=%d)", n, operator.nnz)
        backend = "MUMPS"
    elif solver_type in {"iterative", "krylov", "petsc"}:
        ksp_type = _scfg(_sub, "ksp_type", f"{env_prefix}_KSP_TYPE", "fgmres")
        pc_type = _scfg(_sub, "pc_type", f"{env_prefix}_PC_TYPE", "asm")
        pc_side = _scfg(_sub, "ksp_pc_side", f"{env_prefix}_KSP_PC_SIDE", "right")
        ksp_rtol = _scfg_float(_sub, "ksp_rtol", f"{env_prefix}_KSP_RTOL", 1e-10)
        ksp_atol = _scfg_float(_sub, "ksp_atol", f"{env_prefix}_KSP_ATOL", 1e-14)
        ksp_max_it = _scfg_int(_sub, "ksp_max_it", f"{env_prefix}_KSP_MAX_IT", 20000)
        asm_overlap = _scfg_int(_sub, "asm_overlap", f"{env_prefix}_ASM_OVERLAP", 2)
        sub_ksp_type = _scfg(_sub, "sub_ksp_type", f"{env_prefix}_SUB_KSP_TYPE", "preonly")
        sub_pc_type = _scfg(_sub, "sub_pc_type", f"{env_prefix}_SUB_PC_TYPE", "lu")

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

        # Configure local block solvers for ASM/BJACOBI (parallel-safe defaults).
        if pc_type in {"asm", "bjacobi"}:
            try:
                sub_ksps = pc.getASMSubKSP() if pc_type == "asm" else pc.getBJacobiSubKSP()
                for sub_ksp in sub_ksps:
                    sub_ksp.setType(sub_ksp_type)
                    # Hard-stop on any local block Krylov divergence so iterative
                    # runs cannot silently continue with invalid sub-solves.
                    sub_ksp.setErrorIfNotConverged(True)
                    sub_pc = sub_ksp.getPC()
                    sub_pc.setType(sub_pc_type)
            except PETSc.Error:
                pass

        # Allow runtime override via PETSc options DB.
        ksp.setFromOptions()
        ksp.setUp()
        logger.info(
            "Linear solver: iterative (n=%d, nnz=%d, ksp=%s, pc=%s, side=%s, rtol=%g, max_it=%d)",
            n,
            operator.nnz,
            ksp_type,
            pc_type,
            pc_side,
            ksp_rtol,
            ksp_max_it,
        )
        backend = "PETSc-iterative"
    else:
        raise ValueError(
            f"Unsupported {env_prefix}_TYPE={solver_type!r}; "
            "expected mumps|direct|lu|iterative|krylov|petsc"
        )

    def solve(rhs: np.ndarray) -> np.ndarray:
        b = petsc_vec(rhs)
        x = b.duplicate()
        ksp.solve(b, x)
        return vec_to_numpy(x)

    def solve_transpose(rhs: np.ndarray) -> np.ndarray:
        """Solve A^H x = rhs via conjugation: A^T conj(x) = conj(rhs)."""
        b = petsc_vec(rhs.conj())
        x = b.duplicate()
        ksp.solveTranspose(b, x)
        return vec_to_numpy(x).conj()

    return LinearSolver(backend=backend, solve=solve, solve_transpose=solve_transpose)
