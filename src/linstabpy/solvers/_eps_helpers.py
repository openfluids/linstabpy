# SPDX-License-Identifier: GPL-3.0-only
"""Shared SLEPc EPS helpers used by temporal, adjoint, and resolvent solvers."""

from __future__ import annotations

import numpy as np

from ..utils._solver_cfg import SolverConfig
from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
# mpi4py must bind MPI before PETSc does: mpi4py requests MPI_THREAD_MULTIPLE,
# PETSc only MPI_THREAD_FUNNELED, and init cannot raise the level later.
# STRUMPACK/SLATE refuses to build a factor below MULTIPLE (PETSc error 57).
# Sorts above petsc4py alphabetically, so `ruff --fix` preserves the order.
# Enforced by tests/test_mpi_thread_level.py.
import mpi4py.MPI  # noqa: F401
from petsc4py import PETSc
from slepc4py import SLEPc

_CONV_TEST_MAPPING = {
    "abs": SLEPc.EPS.Conv.ABS,
    "absolute": SLEPc.EPS.Conv.ABS,
    "rel": SLEPc.EPS.Conv.REL,
    "relative": SLEPc.EPS.Conv.REL,
    "norm": SLEPc.EPS.Conv.NORM,
}

_ERROR_TYPE_MAPPING = {
    "abs": SLEPc.EPS.ErrorType.ABSOLUTE,
    "absolute": SLEPc.EPS.ErrorType.ABSOLUTE,
    "rel": SLEPc.EPS.ErrorType.RELATIVE,
    "relative": SLEPc.EPS.ErrorType.RELATIVE,
    "bwd": SLEPc.EPS.ErrorType.BACKWARD,
    "backward": SLEPc.EPS.ErrorType.BACKWARD,
}


def _resolve_eps_convergence_test(name: str) -> SLEPc.EPS.Conv:
    """Map config strings to SLEPc EPS convergence tests."""
    key = str(name).strip().lower()
    try:
        return _CONV_TEST_MAPPING[key]
    except KeyError as exc:
        raise ValueError(f"Invalid EPS convergence test {name!r}; expected abs|rel|norm") from exc


def _resolve_eps_error_type(name: str) -> SLEPc.EPS.ErrorType:
    """Map config strings to SLEPc EPS residual/error types."""
    key = str(name).strip().lower()
    try:
        return _ERROR_TYPE_MAPPING[key]
    except KeyError as exc:
        raise ValueError(
            f"Invalid EPS error type {name!r}; expected absolute|relative|backward"
        ) from exc


def _install_eps_monitor(
    eps: SLEPc.EPS,
    every: int,
    *,
    label: str = "eigs",
    track_state: bool = True,
) -> dict[str, float | int] | None:
    """Log sparse EPS progress; return a mutable state dict iff ``track_state``."""
    interval = max(int(every), 1)
    state: dict[str, float | int] | None = (
        {
            "last_nconv": -1,
            "last_iteration": 0,
            "last_min_err": float("nan"),
            "last_max_err": float("nan"),
        }
        if track_state
        else None
    )

    def _monitor(
        eps_obj: SLEPc.EPS,
        iteration: int,
        nconv: int,
        eigs: list[complex],
        errs: list[float],
    ) -> None:
        if state is not None:
            should_log = iteration == 1 or iteration % interval == 0 or nconv != state["last_nconv"]
        else:
            should_log = iteration == 1 or iteration % interval == 0
        if not should_log:
            return
        finite_errs = [float(val) for val in errs if np.isfinite(val)]
        max_err = max(finite_errs) if finite_errs else float("nan")
        min_err = min(finite_errs) if finite_errs else float("nan")
        if state is not None:
            state["last_iteration"] = int(iteration)
            state["last_nconv"] = int(nconv)
            state["last_min_err"] = float(min_err)
            state["last_max_err"] = float(max_err)
        PETSc.Sys.Print(
            f"[{label}] EPS iter={int(iteration)} nconv={int(nconv)} "
            f"err_range=[{min_err:.3e}, {max_err:.3e}]"
        )

    eps.setTrackAll(True)
    eps.setMonitor(_monitor)
    return state


def _monitor_state_details(state: dict[str, float | int] | None) -> list[str]:
    """Format the latest EPS monitor state for logs and errors."""
    if not state:
        return []
    details: list[str] = []
    last_iteration = int(state.get("last_iteration", 0))
    if last_iteration > 0:
        details.append(f"monitor_iter={last_iteration}")
    last_nconv = int(state.get("last_nconv", -1))
    if last_nconv >= 0:
        details.append(f"monitor_nconv={last_nconv}")
    last_min_err = float(state.get("last_min_err", np.nan))
    if np.isfinite(last_min_err):
        details.append(f"monitor_min_err={last_min_err:.3e}")
    last_max_err = float(state.get("last_max_err", np.nan))
    if np.isfinite(last_max_err):
        details.append(f"monitor_max_err={last_max_err:.3e}")
    return details


def _resolve_solver_cfg_subsection(
    solver_cfg: dict | None,
    env_prefix: str,
) -> dict | None:
    """Return the YAML subsection matching ``env_prefix``.

    Compatibility bridge for older solver code.
    Callers still ask by environment-prefix family.
    SolverConfig owns the actual subsection policy and the
    resolvent -> linsolve fallback.
    """
    config = SolverConfig.from_config(solver_cfg)
    return None if config is None else config.subsection(env_prefix)
