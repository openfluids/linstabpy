#!/usr/bin/env python3
"""Assemble one Savarino beta line once, then factor every frequency.

This is the line campaign: same physics as gen_campaign.py (conservative
momentum P, L2 forcing, Chu response, epsilon 0). It does not recycle
Krylov bases. Each frequency still uses a MUMPS factor of L0 - i omega.

Writes savarino_line_b<beta>.csv and savarino_line_b<beta>.log next to
--out. Rank 0 writes the files. Time every stage with time.perf_counter.

This run measures wall time.
It sets certification_required to False, so the recycled sweep logs a
failed RSVD check and continues.
A gain from this run is not a published gain.
Published gains come from the farm campaign (gen_campaign.py) and from
savarino_mapwalk.py.
Those two keep the certification and the residual checks on.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from examples.savarino.gen_campaign import GRID, TWO_PI, tag, x_max  # noqa: E402
from linstabpy.baseflow import load_case_fields, sutherland_air  # noqa: E402
from linstabpy.config import Config  # noqa: E402
from linstabpy.config.defaults import stage_dissipation_epsilon  # noqa: E402
from linstabpy.mesh import Mesh  # noqa: E402
from linstabpy.operators.assembly import build_lnse_system  # noqa: E402
from linstabpy.operators.energy_norm import compute_chu_norm  # noqa: E402
from linstabpy.resolvent.analysis import resolvent_analysis  # noqa: E402
from linstabpy.solvers.router.resolvent_map import _forcing_response_mats  # noqa: E402
from linstabpy.utils.yaml_io import load_config  # noqa: E402

FIELDS = (
    "beta_Xsh",
    "f_Xsh_Uinf",
    "gain",
    "rho_fwd",
    "rho_adj",
    "status",
    "wall_s",
    "load_s",
    "assemble_s",
    "window_s",
    "factor_s",
    "gmres_s",
    "rsvd_s",
    "recycle_setup_s",
    "line_wall_s",
    "peak_rss_bytes",
    "ndof",
    "nnz",
    "epsilon",
    "stride",
    "fd_order",
    "sponge",
    "forcing_norm",
    "response_norm",
    "P_vars",
    "P_x_min",
    "P_x_max",
    "P_y_max",
    "mask_bc_forcing",
    "ranks",
    "backend",
    "git_sha",
    "slurm_job",
    "machine",
    "method",
    "n_anchors",
    "recycle_rank",
)
HEADER = ",".join(FIELDS)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _rss_bytes() -> int:
    import resource

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _log(path: Path, line: str) -> None:
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    from mpi4py import MPI
    from petsc4py import PETSc  # noqa: F401

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--config", type=Path, default=HERE / "configs" / "mack_f2_b45.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--machine", default="")
    ap.add_argument(
        "--method",
        choices=("line_mumps", "recycle"),
        default="line_mumps",
        help="line_mumps: factor every f. recycle: anchor GMRES + Krylov bases.",
    )
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    beta = float(args.beta)
    if beta not in GRID:
        raise SystemExit(f"beta={beta} is not in gen_campaign.GRID")
    freqs = list(GRID[beta])
    stem = f"savarino_line_b{tag(beta)}"
    out_dir = args.out
    csv_path = out_dir / f"{stem}.csv"
    log_path = out_dir / f"{stem}.log"
    line_t0 = time.perf_counter()

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(log_path, f"savarino_line beta={beta} n_f={len(freqs)} ranks={ranks}")
        _log(
            log_path,
            f"method={args.method} assemble_once=1 recycle={int(args.method == 'recycle')}",
        )
    method = args.method

    cfg = load_config(args.config)
    cfg["numerics"]["beta"] = beta
    cfg["analysis"]["run_stability"] = False
    cfg["analysis"]["run_resolvent"] = True
    cfg["analysis"]["skip_band_marking"] = True
    cfg["analysis"]["amortized_stable_sweep"] = method == "recycle"
    cfg["physics"]["sponge"] = {"enabled": False}
    cfg["solver"]["mask_bc_forcing"] = True
    cfg["solver"].setdefault("resolvent", {})["eigen"] = {"forcing_norm": "l2"}

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
    if bool(phys.get("zero_mu_t_derivs", False)):
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0
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
    if rank == 0:
        _log(log_path, f"load_mesh {load_s:.3f} s")

    t_as = time.perf_counter()
    L0, idx, bc_indices = build_lnse_system(
        mesh,
        baseflow,
        kx=beta,
        model="2D",
        dissipation_epsilon=stage_dissipation_epsilon(od, "resolvent"),
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=float(cfg["bc"]["penalty"]),
        solver_cfg=cfg.get("solver") or {},
    )
    assemble_s = time.perf_counter() - t_as
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    W_diag, _inv = compute_chu_norm(
        W_int,
        baseflow.RHO.ravel(order="F")[p],
        baseflow.T.ravel(order="F")[p],
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
    )
    ndof = int(mesh.ngp * 5)
    nnz = int(getattr(L0, "getInfo", lambda: {})().get("nz_used", 0) or 0)
    if nnz == 0 and hasattr(L0, "nnz"):
        nnz = int(L0.nnz)
    if rank == 0:
        _log(log_path, f"assemble {assemble_s:.3f} s ndof={ndof} nnz={nnz}")

    cfg_obj = Config.load(cfg)
    solver_cfg = cfg.get("solver") or {}
    sha = _git_sha()
    job = os.environ.get("SLURM_JOB_ID", "")
    rows: list[dict[str, object]] = []
    filter_fn = mesh.filters.filter if getattr(mesh, "filters", None) is not None else None
    filter_ct = mesh.filters.filter_ct if getattr(mesh, "filters", None) is not None else None

    if method == "recycle":
        from linstabpy.solvers.router.stable_sweep import sweep_stable_regime

        groups: dict[float, list[float]] = {}
        for freq in freqs:
            groups.setdefault(x_max(freq), []).append(freq)
        for xmax, group in groups.items():
            point_cfg = yaml.safe_load(yaml.safe_dump(cfg))
            point_cfg["solver"]["input_region"] = {
                "space": "conservative_momentum",
                "vars": ["u", "v", "w"],
                "x": [0.20291667, xmax],
                "y": [0.0, 0.29627645],
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
            omegas = np.asarray([TWO_PI * float(f) for f in group], dtype=complex)
            if omegas.size == 1:
                anchors = omegas.copy()
            else:
                anchors = np.unique(np.array([omegas[0], omegas[omegas.size // 2], omegas[-1]]))
            sweep_cfg = {
                "W": np.asarray(W_diag, dtype=complex).reshape(-1),
                "invW": 1.0 / np.asarray(W_diag, dtype=complex).reshape(-1),
                "solver_cfg": solver_cfg,
                "factorize_cfg": cfg_obj,
                "recycling": {
                    "enabled": True,
                    "max_dim": 8,
                    "measure_baseline": False,
                },
                "fgmres_rtol": 1.0e-8,
                "fgmres_maxiter": 200,
                "fgmres_restart": 50,
                "actual_solve_tol": 1.0e-8,
                "adjoint_factor_tol": 1.0e-8,
                "linearity_tol": 1.0e-8,
                "sigma_tol": 1.0e-6,
                "seeds": (0,),
                # Timing run: log a failed check and continue. See the module
                # docstring. Do not copy this line into a campaign that
                # publishes gains.
                "certification_required": False,
            }
            t_sw = time.perf_counter()
            sweep = sweep_stable_regime(
                L0,
                omegas,
                anchors,
                sweep_cfg,
                B_mat=B_mat,
                C_mat=C_mat,
                filter_fn=filter_fn,
                filter_ct_fn=filter_ct,
            )
            sweep_s = time.perf_counter() - t_sw
            if rank == 0:
                _log(
                    log_path,
                    f"recycle_group xmax={xmax} n={len(group)} sweep_s={sweep_s:.3f} "
                    f"load_s={load_s:.3f} assemble_s={assemble_s:.3f} "
                    f"window_s={window_s:.3f} "
                    f"factor_s={float(sweep.metrics.get('factor_s') or 0.0):.3f} "
                    f"gmres_s={float(sweep.metrics.get('gmres_s') or 0.0):.3f} "
                    f"rsvd_s={float(sweep.metrics.get('rsvd_s') or 0.0):.3f} "
                    f"setup_s={float(sweep.metrics.get('recycle_setup_s') or 0.0):.3f} "
                    f"restart={sweep.metrics.get('fgmres_restart')} "
                    f"maxiter={sweep.metrics.get('fgmres_maxiter')}",
                )
            recycle_rank = int(sweep.metrics.get("recycle_forward_rank_final") or 0)
            n_anchors = int(np.unique(sweep.anchor_indices).size)
            if rank == 0:
                for i, report in enumerate(sweep.certification_reports):
                    failed = [name for name, check in report.checks.items() if not check.passed]
                    _log(
                        log_path,
                        f"cert f={group[i]} passed={report.passed} "
                        f"failed={','.join(failed) or '-'}",
                    )
            for i, freq in enumerate(group):
                factor_i = float(sweep.stage_factor_s[i])
                gmres_i = float(sweep.stage_gmres_s[i])
                rsvd_i = float(sweep.stage_rsvd_s[i])
                setup_i = float(sweep.stage_recycle_setup_s[i])
                wall_i = factor_i + gmres_i + rsvd_i + setup_i
                if rank == 0:
                    _log(
                        log_path,
                        f"point f={freq} gain={sweep.gain[i]} factor={factor_i:.3f} "
                        f"gmres={gmres_i:.3f} rsvd={rsvd_i:.3f} setup={setup_i:.3f} "
                        f"wall={wall_i:.3f}",
                    )
                rows.append(
                    {
                        "beta_Xsh": beta,
                        "f_Xsh_Uinf": freq,
                        "gain": float(sweep.gain[i]),
                        "rho_fwd": float(sweep.residuals[i]),
                        "rho_adj": float(sweep.residuals[i]),
                        "status": "ok",
                        "wall_s": wall_i,
                        "window_s": window_s,
                        "factor_s": factor_i,
                        "gmres_s": gmres_i,
                        "rsvd_s": rsvd_i,
                        "recycle_setup_s": setup_i,
                        "P_x_max": xmax,
                        "n_anchors": n_anchors,
                        "recycle_rank": recycle_rank,
                    }
                )
    else:
        for freq in freqs:
            point_cfg = yaml.safe_load(yaml.safe_dump(cfg))
            point_cfg["solver"]["input_region"] = {
                "space": "conservative_momentum",
                "vars": ["u", "v", "w"],
                "x": [0.20291667, x_max(freq)],
                "y": [0.0, 0.29627645],
            }
            omega = TWO_PI * float(freq)
            t_pt = time.perf_counter()
            status = "ok"
            gain = float("nan")
            rho_fwd = float("nan")
            rho_adj = float("nan")
            window_s = 0.0
            factor_s = float("nan")
            rsvd_s = float("nan")
            try:
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
                gains, _resp, _forc, rho_f, rho_a, info = resolvent_analysis(
                    L0,
                    omega,
                    int(cfg["analysis"].get("n_modes_resolvent", 1)),
                    W_diag,
                    B_mat=B_mat,
                    C_mat=C_mat,
                    filter_fn=filter_fn,
                    filter_ct_fn=filter_ct,
                    solver_cfg=solver_cfg,
                    cfg=cfg_obj,
                    residuals=True,
                    return_info=True,
                )
                gain = float(np.asarray(gains).reshape(-1)[0])
                rho_fwd = float(np.max(np.asarray(rho_f)))
                rho_adj = float(np.max(np.asarray(rho_a)))
                factor_s = float(info.get("factor_s", float("nan")))
                rsvd_s = float(info.get("svd_s", float("nan"))) + float(info.get("residual_s", 0.0))
            except Exception as exc:  # noqa: BLE001 - a failed frequency must stay in the CSV
                status = f"failed:{exc.__class__.__name__}"
                if rank == 0:
                    _log(log_path, f"FAIL f={freq} {exc}")
            wall_s = time.perf_counter() - t_pt
            if rank == 0:
                _log(
                    log_path,
                    f"point f={freq} omega={omega:.6g} gain={gain} window={window_s:.3f} "
                    f"factor={factor_s:.3f} rsvd={rsvd_s:.3f} {status} {wall_s:.3f} s",
                )
            rows.append(
                {
                    "beta_Xsh": beta,
                    "f_Xsh_Uinf": freq,
                    "gain": gain,
                    "rho_fwd": rho_fwd,
                    "rho_adj": rho_adj,
                    "status": status,
                    "wall_s": wall_s,
                    "window_s": window_s,
                    "factor_s": factor_s,
                    "gmres_s": 0.0,
                    "rsvd_s": rsvd_s,
                    "recycle_setup_s": 0.0,
                    "P_x_max": x_max(freq),
                    "n_anchors": 0,
                    "recycle_rank": 0,
                }
            )

    line_wall_s = time.perf_counter() - line_t0
    if rank == 0:
        _log(log_path, f"line_wall {line_wall_s:.3f} s")
        out_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            for row in rows:
                writer.writerow(
                    [
                        row["beta_Xsh"],
                        row["f_Xsh_Uinf"],
                        row["gain"],
                        row["rho_fwd"],
                        row["rho_adj"],
                        row["status"],
                        f"{row['wall_s']:.4f}",
                        f"{load_s:.4f}",
                        f"{assemble_s:.4f}",
                        f"{float(row.get('window_s', 0.0)):.4f}",
                        f"{float(row.get('factor_s', float('nan'))):.4f}",
                        f"{float(row.get('gmres_s', 0.0)):.4f}",
                        f"{float(row.get('rsvd_s', float('nan'))):.4f}",
                        f"{float(row.get('recycle_setup_s', 0.0)):.4f}",
                        f"{line_wall_s:.4f}",
                        _rss_bytes(),
                        ndof,
                        nnz,
                        0.0,
                        int(num.get("stride_ksi", 1)),
                        int(num.get("fd_order", 4)),
                        "false",
                        "l2",
                        "chu",
                        "u|v|w",
                        0.20291667,
                        row["P_x_max"],
                        0.29627645,
                        "true",
                        ranks,
                        "mumps",
                        sha,
                        job,
                        args.machine,
                        method,
                        int(row.get("n_anchors", 0)),
                        int(row.get("recycle_rank", 0)),
                    ]
                )
        np.savez(
            out_dir / f"{stem}.npz",
            beta=np.array([beta]),
            f=np.asarray([r["f_Xsh_Uinf"] for r in rows], dtype=float),
            gain=np.asarray([r["gain"] for r in rows], dtype=float),
            wall_s=np.asarray([r["wall_s"] for r in rows], dtype=float),
            load_s=np.array([load_s]),
            assemble_s=np.array([assemble_s]),
            window_s=np.asarray([r.get("window_s", 0.0) for r in rows], dtype=float),
            factor_s=np.asarray([r.get("factor_s", np.nan) for r in rows], dtype=float),
            gmres_s=np.asarray([r.get("gmres_s", 0.0) for r in rows], dtype=float),
            rsvd_s=np.asarray([r.get("rsvd_s", np.nan) for r in rows], dtype=float),
            recycle_setup_s=np.asarray([r.get("recycle_setup_s", 0.0) for r in rows], dtype=float),
            line_wall_s=np.array([line_wall_s]),
        )
        _log(log_path, f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
