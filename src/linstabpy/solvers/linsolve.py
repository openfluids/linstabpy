# SPDX-License-Identifier: GPL-3.0-only
"""Backward-compatible linear solver entry point."""

from __future__ import annotations

import scipy.sparse as sp

from linstabpy.config import Config
from linstabpy.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from linstabpy.linalg.factorize import Factorization, factorize
from linstabpy.solvers.backends import apply_factor_backend

LinearSolver = Factorization


def make_linear_solver(
    operator: sp.spmatrix | PETSc.Mat,
    *,
    env_prefix: str = "LINSTABPY_LINSOLVE",
    solver_cfg: dict | None = None,
    cfg: Config | None = None,
) -> Factorization:
    """Build a reusable PETSc linear solver for ``operator``."""
    return factorize(operator, env_prefix=env_prefix, solver_cfg=solver_cfg, cfg=cfg)


__all__ = ["Factorization", "LinearSolver", "apply_factor_backend", "make_linear_solver"]
