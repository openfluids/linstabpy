#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-only
# Benchmark tool for the linstabpy backend measurements.
"""Dump K = L0 - i*omega*I at one Savarino map point.

The dump feeds the FP32 and cuDSS benchmarks.
It writes one ``.npz`` file with: the CSR arrays of K in complex double;
the forcing matrix B in CSR; two right-hand sides; W_diag; point metadata.
The right-hand sides are a random complex vector (seed 0) and ``B @ f``
with a random ``f``. No factorization happens here.

Run on one rank from the linstabpy repo root, on c9 through sbatch::

    PYTHONPATH=src:. python dump_operator.py --config CFG --beta 40 --f 1 --out DIR

The forcing region is the farm's (P_x_min, P_x_max, P_y_max) for f <= 1.8.
Pass ``--x-max`` for other frequencies; the 200-point CSV lists it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml

TWO_PI = 6.283185307179586
X_MIN, Y_MAX = 0.20291667, 0.29627645
X_MAX_DEFAULT = 1.7954167  # farm P_x_max for f <= 1.8


def git_sha() -> str:
    """Return the short git sha of the current directory, or ``unknown``."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        return subprocess.check_output([git, "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S603
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def to_csr(mat: Any) -> sp.csr_matrix:  # noqa: ANN401  (scipy matrix or PETSc.Mat; PETSc has no stubs)
    """Return a scipy CSR copy of a scipy matrix, a 1-D diagonal, or a one-rank PETSc matrix."""
    if sp.issparse(mat):
        return sp.csr_matrix(mat)
    if isinstance(mat, np.ndarray):
        return sp.csr_matrix(sp.diags(mat.ravel()) if mat.ndim == 1 else mat)
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=tuple(mat.getSize()))


def load_point_cfg(config: Path, beta: float) -> dict[str, Any]:
    """Load the point config and apply the campaign settings (no EVP, no sponge)."""
    from linstabpy.utils.yaml_io import load_config  # noqa: PLC0415  (after MPI init, see main)

    cfg: dict[str, Any] = load_config(config)
    case_dir = Path(str(cfg["paths"]["case_dir"]))
    if not case_dir.is_absolute():
        cfg["paths"]["case_dir"] = str((config.parent / case_dir).resolve())
    cfg["analysis"]["run_stability"] = False
    cfg["analysis"]["run_resolvent"] = True
    cfg["analysis"]["skip_band_marking"] = True
    cfg["physics"]["sponge"] = {"enabled": False}
    cfg["solver"]["mask_bc_forcing"] = True
    eig = cfg["solver"].setdefault("resolvent", {}).setdefault("eigen", {})
    eig["forcing_norm"] = "l2"
    cfg["numerics"]["beta"] = float(beta)
    return cfg


def assemble_point(
    cfg: dict[str, Any], beta: float, f: float, x_max: float, log_path: Path, rank: int
) -> dict[str, Any]:
    """Assemble K, B and the two right-hand sides at (beta, f)."""
    # The Savarino example exposes its assembly and forcing builders only as
    # private helpers; the map walker calls the same two functions.
    # The router package re-exports a FUNCTION named ``resolvent_map`` that shadows
    # the submodule attribute, so ``import ... as`` (getattr-based) returns the
    # function; ``import_module`` goes through ``sys.modules`` and returns the module.
    rmap = importlib.import_module("linstabpy.solvers.router.resolvent_map")
    from examples.savarino.savarino_mapwalk import _assemble  # noqa: PLC0415
    from linstabpy.linalg.shifted import shifted_operator  # noqa: PLC0415

    t0 = time.perf_counter()
    L0, idx, bc_indices, mesh, baseflow, W_diag, ndof = _assemble(cfg, float(beta), log_path, rank)  # noqa: N806  (paper names)
    t_assemble = time.perf_counter() - t0
    point_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    point_cfg["solver"]["input_region"] = {
        "space": "conservative_momentum",
        "vars": ["u", "v", "w"],
        "x": [X_MIN, float(x_max)],
        "y": [0.0, Y_MAX],
    }
    B_mat, _c_mat = rmap._forcing_response_mats(  # noqa: N806, SLF001  (paper name; private helper)
        L0, ndof, bc_indices, point_cfg, mesh=mesh, idx=idx, base_flow=baseflow
    )
    omega = TWO_PI * float(f)
    K = to_csr(shifted_operator(L0, omega))  # noqa: N806
    K.sort_indices()
    B = to_csr(B_mat)  # noqa: N806
    rng = np.random.default_rng(0)
    rhs_random = rng.standard_normal(ndof) + 1j * rng.standard_normal(ndof)
    f_forcing = rng.standard_normal(B.shape[1]) + 1j * rng.standard_normal(B.shape[1])
    rhs_forcing = np.asarray(B @ f_forcing).reshape(-1)
    return {
        "K": K,
        "B": B,
        "rhs_random": rhs_random,
        "rhs_forcing": rhs_forcing,
        "f_forcing": f_forcing,
        "W_diag": np.asarray(W_diag),
        "ndof": int(ndof),
        "omega": omega,
        "assemble_s": t_assemble,
        "L0_backend": "scipy" if sp.issparse(L0) else "petsc",
    }


def main() -> int:
    """Parse arguments, assemble the point, write the ``.npz`` file."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--f", type=float, required=True, help="f X_sh/U_inf; omega = 2 pi f")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--x-max", type=float, default=X_MAX_DEFAULT)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # Import mpi4py before petsc4py: the SLATE build of the GPU venv needs
    # MPI_THREAD_MULTIPLE, and only mpi4py's init requests it.
    from mpi4py import MPI  # noqa: PLC0415

    rank = MPI.COMM_WORLD.Get_rank()
    if MPI.COMM_WORLD.Get_size() != 1:
        msg = "dump_operator.py runs on one rank"
        raise SystemExit(msg)

    config = args.config.resolve()
    cfg = load_point_cfg(config, args.beta)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"b{args.beta:g}_f{args.f:g}".replace(".", "p")
    pt = assemble_point(cfg, args.beta, args.f, args.x_max, args.out / f"dump_{tag}.log", rank)

    K, B = pt["K"], pt["B"]  # noqa: N806
    row_inf = np.asarray(abs(K).max(axis=1).todense()).ravel()
    pos = row_inf[row_inf > 0]
    meta = {
        "beta": float(args.beta),
        "f": float(args.f),
        "omega": pt["omega"],
        "ndof": pt["ndof"],
        "nnz": int(K.nnz),
        "n_forcing": int(B.shape[1]),
        "x_max": float(args.x_max),
        "config": str(config),
        "git_sha": git_sha(),
        "assemble_s": pt["assemble_s"],
        "row_inf_norm_max": float(pos.max()),
        "row_inf_norm_min": float(pos.min()),
        "row_inf_norm_spread_decades": float(np.log10(pos.max() / pos.min())),
        "L0_backend": pt["L0_backend"],
    }
    out = args.out / f"K_{tag}.npz"
    np.savez(
        out,
        K_indptr=K.indptr,
        K_indices=K.indices,
        K_data=K.data.astype(np.complex128),
        B_indptr=B.indptr,
        B_indices=B.indices,
        B_data=B.data.astype(np.complex128),
        B_shape=np.array(B.shape),
        rhs_random=pt["rhs_random"],
        rhs_forcing=pt["rhs_forcing"],
        f_forcing=pt["f_forcing"],
        W_diag=pt["W_diag"],
        meta=json.dumps(meta),
    )
    sys.stdout.write(json.dumps({"out": str(out), **meta}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
