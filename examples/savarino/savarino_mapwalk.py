# SPDX-License-Identifier: GPL-3.0-only
"""Time LineWalker on the Savarino farm grid against committed MUMPS walls.

Assemble L0 once per beta. Rebuild B and C when farm x_max changes.
Keep the live K factor. Gain comes from SLEPc. Append each row.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from examples.savarino.gen_campaign import TWO_PI, X_MIN, Y_MAX, x_max  # noqa: E402
from linstabpy.solvers.router.map_record import (  # noqa: E402
    append_row,
    finished_pairs,
    last_factor_seconds,
    remaining,
)
from linstabpy.solvers.router.map_walk import WalkDeferred  # noqa: E402
from linstabpy.solvers.router.resolvent_map import _peak_rss_bytes  # noqa: E402

STEM = "savarino_mapwalk"
FARM_CSV = HERE / "savarino_map2_200pt.csv"
FIELDS = (
    "beta_Xsh",
    "f_Xsh_Uinf",
    "gain",
    "farm_gain",
    "farm_wall_s",
    "rel_err",
    "walk_source",
    "residual",
    "rho_fwd",
    "rho_adj",
    "iterations",
    "factor_s",
    "solve_s",
    "wall_s",
    "window_s",
    "P_x_max",
    "n_factor",
    "last_n_applies",
    "ndof",
    "peak_rss_bytes",
    "epsilon",
    "stride",
    "ranks",
    "backend",
    "blas",
    "ncv",
    "n_anchor",
    "pin_domain",
    "git_sha",
    "slurm_job",
    "machine",
)


def load_farm(csv_path: Path | None = None) -> list[dict[str, float]]:
    """Load farm rows from the committed campaign CSV."""
    path = FARM_CSV if csv_path is None else Path(csv_path)
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for rec in csv.DictReader(handle):
            rows.append(
                {
                    "beta": float(rec["beta_Xsh"]),
                    "f": float(rec["f_Xsh_Uinf"]),
                    "gain": float(rec["gain"]),
                    "wall_s": float(rec["wall_s"]),
                }
            )
    rows.sort(key=lambda row: (row["beta"], row["f"]))
    return rows


def farm_betas(rows: list[dict[str, float]]) -> list[float]:
    """Return unique beta values in farm order."""
    seen: list[float] = []
    for row in rows:
        if not seen or abs(row["beta"] - seen[-1]) > 1.0e-12:
            seen.append(row["beta"])
    return seen


def relative_error(gain: float, farm_gain: float) -> float:
    """Return |gain - farm_gain| / |farm_gain|."""
    denom = abs(float(farm_gain))
    if denom == 0.0:
        raise ValueError("farm_gain must be nonzero")
    return abs(float(gain) - float(farm_gain)) / denom


def _backend_label(cfg: Mapping[str, Any]) -> str:
    """Name the resolvent linear solver a row ran on, from the resolved config."""
    from linstabpy.config.typed import Config

    lin = Config.load(cfg).resolvent.linear
    solver_type = str(lin.type or "").lower()
    factor = str(lin.factor_backend or "").lower()
    if solver_type in ("", "mumps", "direct", "lu"):
        return factor or solver_type or "unknown"
    return f"{solver_type}/{factor}" if factor else solver_type


def _pin_domain(env: Mapping[str, str]) -> str:
    """Name the MPI rank placement a job asked for, as it lands in the row.

    Intel MPI: ``I_MPI_PIN_DOMAIN`` when set, ``off`` when pinning was switched off,
    ``default`` otherwise. Placement changed the factor time by 1.6x on c9
    (job 1861), so a cost row without it is not comparable.
    """
    domain = env.get("I_MPI_PIN_DOMAIN", "").strip()
    if domain:
        return domain
    if env.get("I_MPI_PIN", "").strip().lower() in {"0", "off", "disable", "no", "false"}:
        return "off"
    return "default"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _log(path: Path, line: str) -> None:
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _assemble(
    cfg: dict[str, Any], beta: float, log_path: Path, rank: int
) -> tuple[Any, Any, Any, Any, Any, Any, int]:
    from linstabpy.baseflow import load_case_fields, sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system
    from linstabpy.operators.energy_norm import compute_chu_norm

    case_dir = Path(str(cfg["paths"]["case_dir"]))
    if not case_dir.is_absolute():
        case_dir = (HERE / case_dir).resolve()
    phys = cfg["physics"]
    num = cfg["numerics"]
    t_load = time.perf_counter()
    _X, _Y, baseflow, sponge, _info = load_case_fields(
        case_dir,
        Re=float(phys["Re"]),
        Ma=float(phys["Ma"]),
        Pr=float(phys["Pr"]),
        kappa=float(phys["kappa"]),
        T_0=float(phys["T_0"]),
        stride_ksi=int(num.get("stride_ksi", 1)),
        stride_eta=int(num.get("stride_eta", 1)),
        transpose_input=True,
        zero_spanwise=bool(phys.get("zero_spanwise", False)),
    )
    sutherland_air(baseflow)
    ny, nx = _X.shape
    od = num.get("operator_dissipation") or {}
    mesh = Mesh.create(
        (0.0, 1.0),
        (0.0, 0.5),
        nx=nx,
        ny=ny,
        fd_order=int(num.get("fd_order", 4)),
        alpha_filter=num.get("alpha_filter", 0.0),
        numerics_operator_dissipation=od if bool(od.get("enabled", False)) else None,
    )
    mesh = mesh.deform(X_new=_X, Y_new=_Y)
    mesh.sponge = sponge
    load_s = time.perf_counter() - t_load
    t_as = time.perf_counter()
    L0, idx, bc_indices = build_lnse_system(
        mesh,
        baseflow,
        kx=float(beta),
        model="2D",
        dissipation_epsilon=0.0,
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=float(cfg["bc"]["penalty"]),
        solver_cfg=cfg.get("solver") or {},
    )
    assemble_s = time.perf_counter() - t_as
    p = mesh.used_ind
    W_diag, _inv = compute_chu_norm(
        mesh.W.ravel(order="F")[p],
        baseflow.RHO.ravel(order="F")[p],
        baseflow.T.ravel(order="F")[p],
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
    )
    ndof = int(mesh.ngp * 5)
    if rank == 0:
        _log(
            log_path,
            f"load_mesh {load_s:.3f} s assemble {assemble_s:.3f} s nx={nx} ny={ny} ndof={ndof} beta={beta}",
        )
    return L0, idx, bc_indices, mesh, baseflow, W_diag, ndof


def main() -> int:
    from mpi4py import MPI
    from petsc4py import PETSc  # noqa: F401

    from linstabpy.config import Config
    from linstabpy.solvers.router.map_walk import LineWalker
    from linstabpy.solvers.router.resolvent_map import _forcing_response_mats
    from linstabpy.utils.provenance import collect_provenance
    from linstabpy.utils.yaml_io import load_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "savarino_point_f2_b45.yaml")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--farm-csv", type=Path, default=FARM_CSV)
    ap.add_argument("--beta", type=float, nargs="+", default=None)
    ap.add_argument(
        "--n-anchor",
        type=int,
        default=None,
        help="direct (MUMPS) points per beta line before the walk starts; "
        "a value >= the line length makes this a pure direct driver (the farm arm)",
    )
    ap.add_argument(
        "--ncv",
        type=int,
        default=None,
        help="SLEPc Krylov subspace size for the H^H H gain (nev=1); "
        "overrides solver.resolvent.eigen.ncv in the config",
    )
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    farm_rows = load_farm(args.farm_csv)
    betas = args.beta if args.beta is not None else farm_betas(farm_rows)
    farm_gain = {(row["beta"], row["f"]): row["gain"] for row in farm_rows}
    farm_wall = {(row["beta"], row["f"]): row["wall_s"] for row in farm_rows}
    by_beta: dict[float, list[float]] = {}
    for row in farm_rows:
        by_beta.setdefault(row["beta"], []).append(row["f"])

    out_dir = args.out
    csv_path = out_dir / f"{STEM}.csv"
    log_path = out_dir / f"{STEM}.log"
    sha = _git_sha()
    machine = socket.gethostname().split(".")[0]
    slurm_job = os.environ.get("SLURM_JOB_ID", "")
    blas = collect_provenance(None)["blas"]
    pin_domain = _pin_domain(os.environ)

    cfg = load_config(args.config)
    cfg["analysis"]["run_stability"] = False
    cfg["analysis"]["run_resolvent"] = True
    cfg["analysis"]["skip_band_marking"] = True
    cfg["physics"]["sponge"] = {"enabled": False}
    cfg["solver"]["mask_bc_forcing"] = True
    eigen_cfg = cfg["solver"].setdefault("resolvent", {}).setdefault("eigen", {})
    eigen_cfg["forcing_norm"] = "l2"
    # What factored the operator, as the row's own provenance: "mumps" (direct)
    # or "iterative/strumpack" (FGMRES around a STRUMPACK LU, the GPU arm).
    backend = _backend_label(cfg)
    if args.ncv is not None:
        eigen_cfg["ncv"] = int(args.ncv)
    ncv = eigen_cfg.get("ncv", "")
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(
            log_path,
            f"{STEM} sha={sha} machine={machine} slurm_job={slurm_job} "
            f"ranks={ranks} betas={betas} blas={blas} ncv={ncv or 'default'} "
            f"n_anchor={args.n_anchor if args.n_anchor is not None else 'default'} "
            f"pin_domain={pin_domain} "
            "resolution=full-res 300x300 stride=1 epsilon=0 skip_band_marking=true",
        )

    done: set[tuple[float, float]] = set()
    if rank == 0:
        done = finished_pairs(csv_path, beta_field="beta_Xsh", f_field="f_Xsh_Uinf")
    done = comm.bcast(done, root=0)

    for beta in betas:
        freqs = remaining(by_beta[float(beta)], done, beta=float(beta))
        if rank == 0:
            _log(
                log_path,
                f"beta={beta} n_left={len(freqs)} already={len(by_beta[float(beta)]) - len(freqs)}",
            )
        if not freqs:
            continue
        cfg["numerics"]["beta"] = float(beta)
        L0, idx, bc_indices, mesh, baseflow, W_diag, ndof = _assemble(
            cfg, float(beta), log_path, rank
        )
        cfg_obj = Config.load(cfg)
        solver_cfg = cfg.get("solver") or {}
        filter_fn = mesh.filters.filter if getattr(mesh, "filters", None) is not None else None
        filter_ct = mesh.filters.filter_ct if getattr(mesh, "filters", None) is not None else None
        walker = LineWalker(solver_cfg=solver_cfg, cfg_obj=cfg_obj)
        if args.n_anchor is not None:
            walker.n_anchor = int(args.n_anchor)
        if rank == 0:
            seeded = last_factor_seconds(csv_path)
        else:
            seeded = 0.0
        seeded = comm.bcast(seeded, root=0)
        if seeded > 0.0:
            walker.last_factor_s = float(seeded)
            walker.last_direct_wall_s = float(seeded)
        last_xmax: float | None = None
        B_mat = None
        C_mat = None
        try:
            for freq in freqs:
                xmax = float(x_max(freq))
                omega = TWO_PI * float(freq)
                t_pt = time.perf_counter()
                window_s = 0.0
                if last_xmax is None or xmax != last_xmax:
                    point_cfg = yaml.safe_load(yaml.safe_dump(cfg))
                    point_cfg["solver"]["input_region"] = {
                        "space": "conservative_momentum",
                        "vars": ["u", "v", "w"],
                        "x": [X_MIN, xmax],
                        "y": [0.0, Y_MAX],
                    }
                    t_win = time.perf_counter()
                    B_mat, C_mat = _forcing_response_mats(
                        L0,
                        ndof,
                        bc_indices,
                        point_cfg,
                        mesh=mesh,
                        idx=idx,
                        base_flow=baseflow,
                    )
                    window_s = time.perf_counter() - t_win
                    last_xmax = xmax
                    if rank == 0:
                        _log(
                            log_path,
                            f"rebuild_BC beta={beta} f={freq} P_x_max={xmax} window={window_s:.3f} s",
                        )
                try:
                    walked = walker.compute(
                        L0,
                        omega,
                        W_diag,
                        1,
                        B_mat=B_mat,
                        C_mat=C_mat,
                        filter_fn=filter_fn,
                        filter_ct_fn=filter_ct,
                    )
                except WalkDeferred:
                    if rank == 0:
                        _log(
                            log_path,
                            f"stop beta={beta} f={freq}: remaining wall shorter than a factor",
                        )
                    return 0
                wall_s = time.perf_counter() - t_pt
                farm = float(farm_gain[(float(beta), float(freq))])
                rel = relative_error(walked.gain, farm)
                # Same meaning as the farm rows (driver/run.py): peak RSS summed
                # over ranks. Collective, so it stays outside the rank-0 block.
                peak_rss_total = comm.allreduce(int(_peak_rss_bytes()), op=MPI.SUM)
                if rank == 0:
                    _log(
                        log_path,
                        f"point beta={beta} f={freq} omega={omega:.6g} "
                        f"gain={walked.gain:.8g} farm={farm:.8g} rel_err={rel:.3e} "
                        f"source={walked.source} residual={walked.residual:.3e} "
                        f"rho_fwd={walked.rho_fwd:.3e} rho_adj={walked.rho_adj:.3e} "
                        f"iters={walked.iterations} factor={walked.factor_s:.3f} "
                        f"solve={walked.solve_s:.3f} wall={wall_s:.3f} "
                        f"farm_wall={farm_wall[(float(beta), float(freq))]:.3f} "
                        f"n_apply={walker.last_n_applies}",
                    )
                    append_row(
                        csv_path,
                        FIELDS,
                        (
                            beta,
                            freq,
                            f"{walked.gain:.10g}",
                            farm,
                            farm_wall[(float(beta), float(freq))],
                            f"{rel:.6e}",
                            walked.source,
                            walked.residual,
                            f"{walked.rho_fwd:.4e}",
                            f"{walked.rho_adj:.4e}",
                            walked.iterations,
                            f"{walked.factor_s:.4f}",
                            f"{walked.solve_s:.4f}",
                            f"{wall_s:.4f}",
                            f"{window_s:.4f}",
                            xmax,
                            walker.n_factor,
                            walker.last_n_applies,
                            ndof,
                            peak_rss_total,
                            0.0,
                            1,
                            ranks,
                            backend,
                            blas,
                            ncv,
                            walker.n_anchor,
                            pin_domain,
                            sha,
                            slurm_job,
                            machine,
                        ),
                    )
        finally:
            walker.close()
    if rank == 0:
        _log(log_path, f"done sha={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
