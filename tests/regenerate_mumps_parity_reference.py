# SPDX-License-Identifier: GPL-3.0-only
"""Regenerate the CPU/MUMPS parity reference fixture.

This helper is intentionally explicit: it is never imported by production code
and never runs automatically from the parity test. To refresh the fixture after
a reviewed numerical-baseline decision, run from the repository root:

    PYTHONPATH=src .venv/bin/python tests/regenerate_mumps_parity_reference.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from examples.martini.matlab_baseflow import load_blasius
from linstabpy.baseflow import sutherland_air
from linstabpy.boundary_conditions import apply_dirichlet
from linstabpy.mesh import Mesh
from linstabpy.operators import build_lnse_operator
from linstabpy.stability import eigenvalues as temporal_eigenvalues

REFERENCE_PATH = Path(__file__).with_name("fixtures") / "mumps_parity_reference.npz"
# Regression guard, not a bit-parity gate. The tolerance must sit ABOVE the
# build-to-build rounding floor and far BELOW any physically meaningful drift:
#
#   * within one build the solve is deterministic (repeat runs agree to 0.0);
#   * across independently configured PETSc/MUMPS/BLAS builds the converged
#     eigenvalues move by ~1e-8 relative (measured: CPU vs GPU-stack venv on the
#     same host, identical nconv/ncv/eps_tol/backend). That is expected: the
#     eigensolver converges to eps_tol=1e-10 on the residual, and for a strongly
#     non-normal operator such as the LNSE the eigenvalue error is that residual
#     scaled by the eigenvalue condition number, so it exceeds eps_tol.
#   * a real regression (e.g. a dropped operator term) moves eigenvalues by
#     ~1e-3 or more -- three orders of magnitude above this bound.
#
# The former 1.0e-12 sat *below* the cross-build noise floor, so it could never
# be stably green: it failed on every fresh build even when the physics was
# unchanged. Do NOT relax this further to make a run pass -- if a drift above
# 1e-6 appears, that is a finding to investigate, not a number to raise.
RELATIVE_DRIFT_TOL = 1.0e-6

CASE_PARAMS: dict[str, Any] = {
    "case_id": "mumps_parity_blasius_lnse_v1",
    "description": "Small deterministic Blasius LNSE EVP exercising the default SLEPc/MUMPS shift-invert path.",
    "mesh": {"xrange": [200.0, 300.0], "yrange": [0.0, 30.0], "nx": 15, "ny": 10, "fd_order": 4},
    "y_deformation": {"kind": "sinh", "s": 3.0},
    "baseflow": {"Re": 1.0e4, "Ma": 0.3, "Pr": 0.7, "kappa": 1.4, "T_0": 288.15},
    "operator": {"kx": 0.0, "dirichlet_borders": "bt", "dirichlet_variables": "uvwT"},
    "target_real": 0.0,
    "target_imag": -0.1,
    "n_modes": 5,
}

SOLVER_OPTIONS: dict[str, Any] = {
    "solver_cfg": {},
    "ncv": None,
}


def build_reference_operator():
    """Return the small Blasius LNSE operator used by the parity baseline."""
    mesh_params = CASE_PARAMS["mesh"]
    mesh = Mesh.create(
        xrange=tuple(mesh_params["xrange"]),
        yrange=tuple(mesh_params["yrange"]),
        nx=mesh_params["nx"],
        ny=mesh_params["ny"],
        fd_order=mesh_params["fd_order"],
    )
    deformation = CASE_PARAMS["y_deformation"]
    y_top = mesh_params["yrange"][1]
    s = deformation["s"]
    Y_new = y_top * np.sinh(s * mesh.Y / y_top) / np.sinh(s)
    mesh = mesh.deform(Y_new=Y_new)

    bf_params = CASE_PARAMS["baseflow"]
    bf = load_blasius(mesh.X, mesh.Y, **bf_params)
    sutherland_air(bf)

    L0, idx = build_lnse_operator(mesh, bf, kx=CASE_PARAMS["operator"]["kx"])
    return apply_dirichlet(
        L0,
        idx,
        CASE_PARAMS["operator"]["dirichlet_borders"],
        CASE_PARAMS["operator"]["dirichlet_variables"],
        mesh.ngp,
    )[0]


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip()


def _deterministic_storage_order(values: np.ndarray, target: complex) -> np.ndarray:
    values = np.asarray(values, dtype=complex).reshape(-1)
    order = np.lexsort((values.imag, values.real, np.abs(values - target)))
    return values[order]


def _polluted_linstabpy_env() -> list[str]:
    return sorted(name for name in os.environ if name.startswith("LINSTABPY_"))


def refuse_polluted_linstabpy_env() -> None:
    """Abort fixture regeneration when ambient LINSTABPY_* settings are present."""
    polluted = _polluted_linstabpy_env()
    if polluted:
        joined = ", ".join(polluted)
        raise RuntimeError(
            "refusing to regenerate MUMPS parity fixture with LINSTABPY_* "
            f"environment variables set: {joined}"
        )


def _solve_reference(operator, *, n_modes: int, target: complex):
    eigenvalues, _modes, info = temporal_eigenvalues(
        operator,
        n_modes=n_modes,
        target=target,
        return_info=True,
    )
    return np.asarray(eigenvalues, dtype=complex).reshape(-1), info


def compute_reference_quantities() -> dict[str, Any]:
    """Compute the stored default-path quantities for the parity fixture."""
    operator = build_reference_operator()
    target = complex(CASE_PARAMS["target_real"], CASE_PARAMS["target_imag"])
    requested_n_modes = int(CASE_PARAMS["n_modes"])
    eigenvalues, info = _solve_reference(operator, n_modes=requested_n_modes, target=target)
    nconv = int(info.get("nconv", eigenvalues.size))
    stored_n_modes = requested_n_modes
    if nconv > eigenvalues.size:
        # The production eigenvalue API intentionally returns only the requested
        # nearest modes.  For this parity fixture we freeze every converged mode
        # reported by the reviewed baseline solve, so a second solve extracts the
        # complete converged count instead of truncating a near-degenerate pair.
        eigenvalues, storage_info = _solve_reference(operator, n_modes=nconv, target=target)
        stored_n_modes = nconv
        if eigenvalues.size != nconv:
            raise RuntimeError(
                "could not extract all converged modes for MUMPS parity fixture: "
                f"baseline nconv={nconv}, extracted={eigenvalues.size}"
            )
        if int(storage_info.get("nconv", eigenvalues.size)) != nconv:
            raise RuntimeError(
                "MUMPS parity fixture regeneration is not stable across extraction solves: "
                f"baseline nconv={nconv}, storage nconv={storage_info.get('nconv')}"
            )
        info = {**info, "storage_solve_nconv": storage_info.get("nconv")}
    sorted_eigenvalues = _deterministic_storage_order(eigenvalues, target)
    metadata = {
        "case_params": CASE_PARAMS,
        "solver_options": SOLVER_OPTIONS,
        "ndof": int(operator.shape[0]),
        "git_sha": _git_sha(),
        "st_solver": info.get("st_solver"),
        "factor_backend": info.get("backend_probe_factor_backend"),
        "ncv_used": info.get("ncv_used"),
        "n_modes": requested_n_modes,
        "stored_n_modes": stored_n_modes,
        "nconv": nconv,
        "eps_tol": info.get("eps_tol"),
    }
    return {
        "leading_eigenvalues": sorted_eigenvalues,
        "metadata": metadata,
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def write_reference(path: Path = REFERENCE_PATH) -> None:
    """Regenerate the fixture file at *path*."""
    quantities = compute_reference_quantities()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        leading_eigenvalues=quantities["leading_eigenvalues"],
        metadata_json=np.asarray(quantities["metadata_json"]),
        relative_drift_tol=np.asarray(RELATIVE_DRIFT_TOL, dtype=float),
    )


if __name__ == "__main__":
    refuse_polluted_linstabpy_env()
    write_reference()
    print(f"wrote {REFERENCE_PATH}")
