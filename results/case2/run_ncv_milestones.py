#!/usr/bin/env python3
"""Case 2 staged Krylov-space saturation study for the temporal EVP."""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
from linstab2d.results_runtime import (
    _effective_alpha_filter,
    _log,
    _log_stage,
    _postcheck_stability_modes,
    _resolve_omega_target,
)
from linstab2d.baseflow import load_hamada_case, sutherland_air
from linstab2d.boundary_conditions import analyze_bc_recipe
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_system
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.eigenvalue import temporal_eigenvalues
from linstab2d.solvers.petsc_utils import _is_petsc_matrix
from linstab2d.utils.jsonc import load_jsonc
from linstab2d.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cfg_get(config: dict, path: str, default):
    node = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _count_thresholds(values: np.ndarray) -> dict[str, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "count_le_1e6": int(np.sum(finite <= 1e-6)),
        "count_le_1e5": int(np.sum(finite <= 1e-5)),
        "count_le_1e4": int(np.sum(finite <= 1e-4)),
    }


def _prepare_case(config: dict) -> tuple[PETSc.Mat | np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex, dict]:
    case_dir = Path(str(_cfg_get(config, "paths.case_dir", "baseflow")))
    if not case_dir.is_absolute():
        case_dir = (SCRIPT.parent / case_dir).resolve()
    sponge_mat_path = Path(str(_cfg_get(config, "paths.case_inputs_mat", "MATLAB_exports_resolvent/case_inputs_used.mat")))
    if not sponge_mat_path.is_absolute():
        sponge_mat_path = (SCRIPT.parent / sponge_mat_path).resolve()
    Re = float(_cfg_get(config, "physics.Re", 395000.0))
    Ma = float(_cfg_get(config, "physics.Ma", 2.0))
    Pr = float(_cfg_get(config, "physics.Pr", 0.747))
    kappa = float(_cfg_get(config, "physics.kappa", 1.4))
    T_0 = float(_cfg_get(config, "physics.T_0", 1540.0))
    stride_ksi = int(_cfg_get(config, "numerics.stride_ksi", 1))
    stride_eta = int(_cfg_get(config, "numerics.stride_eta", 1))
    fd_order = int(_cfg_get(config, "numerics.fd_order", 4))
    beta = float(_cfg_get(config, "numerics.beta", 0.0))
    penalty = float(_cfg_get(config, "bc.penalty", 1e6))
    zero_spanwise = bool(_cfg_get(config, "physics.zero_spanwise", False))
    use_matlab_sponge_aux = bool(_cfg_get(config, "physics.use_matlab_sponge_aux", False))
    run_resolvent = bool(_cfg_get(config, "analysis.run_resolvent", True))
    alpha_filter_raw = _cfg_get(config, "numerics.alpha_filter", 0.0)
    alpha_filter = _effective_alpha_filter(
        alpha_filter_raw,
        run_resolvent=run_resolvent,
        context="case2-ncv",
    )
    zero_mu_t_derivs = bool(_cfg_get(config, "physics.zero_mu_t_derivs", True))
    ratio_u = float(_cfg_get(config, "analysis.ratio_uinf", 1.0))
    ratio_l = float(_cfg_get(config, "analysis.ratio_l", 0.1))
    St = float(_cfg_get(config, "analysis.St", 0.1))
    omega_target = _resolve_omega_target(
        _cfg_get(config, "analysis.omega_target", None),
        2.0 * np.pi * St * ratio_u / ratio_l,
    )
    solver_cfg = copy.deepcopy(config.get("solver", {}))
    matrix_mode = solver_cfg.get("petsc_matrix_mode", "distributed")
    bc_recipe = _cfg_get(
        config,
        "bc.recipe",
        [
            {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
            {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
            {"type": "neumann", "borders": "b", "variables": "T", "exclude_corners": True},
            {"type": "neumann", "borders": "r", "variables": "ruvwT", "exclude_corners": True},
        ],
    )

    stage_start = time.perf_counter()
    X, Y, baseflow, sponge = load_hamada_case(
        case_dir,
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        stride_ksi=stride_ksi,
        stride_eta=stride_eta,
        transpose_input=True,
        zero_spanwise=zero_spanwise,
        sponge_mat_path=sponge_mat_path,
        prefer_matlab_sponge_aux=use_matlab_sponge_aux and sponge_mat_path.exists(),
        native_sponge_cfg=None,
    )
    sutherland_air(baseflow)
    if zero_mu_t_derivs:
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0
    _log(
        f"loaded case arrays: shape={X.shape}, points={X.size}, "
        f"sponge_max={float(np.max(sponge)):.3g}, sponge_nnz={int(np.count_nonzero(sponge))}"
    )
    _log_stage("load_case", stage_start)

    stage_start = time.perf_counter()
    ny, nx = X.shape
    cmesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=nx, ny=ny, fd_order=fd_order, alpha_filter=alpha_filter)
    mesh = cmesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge
    _log_stage("build_mesh", stage_start)

    stage_start = time.perf_counter()
    _log("assembling LNSE operator")
    L0, idx, bc_indices = build_lnse_system(
        mesh,
        baseflow,
        kx=beta,
        model="2D",
        bc_recipe=bc_recipe,
        bc_penalty=penalty,
        backend=None,
        solver_cfg=solver_cfg | {"petsc_matrix_mode": matrix_mode},
    )
    nnz = int(L0.getInfo()["nz_used"]) if hasattr(L0, "getInfo") else int(L0.nnz)
    _log(f"operator assembled with BCs: shape={(mesh.ngp * 5, mesh.ngp * 5)}, nnz={nnz}")
    _log_stage("assemble_operator_bc", stage_start)

    bc_analysis = analyze_bc_recipe(mesh, idx, bc_recipe)
    effective_dirichlet_indices = np.asarray(
        bc_analysis.get("effective_dirichlet_indices", np.array([], dtype=np.int64)),
        dtype=np.int64,
    )
    total_bc_indices = np.asarray(
        bc_analysis.get("total_bc_indices", np.array([], dtype=np.int64)),
        dtype=np.int64,
    )
    overlap_indices = np.asarray(
        bc_analysis.get("overlap_indices", np.array([], dtype=np.int64)),
        dtype=np.int64,
    )
    effective_neumann_indices = np.asarray(
        bc_analysis.get("effective_neumann_indices", np.array([], dtype=np.int64)),
        dtype=np.int64,
    )
    _log(
        "bc analysis: "
        f"dirichlet_total={effective_dirichlet_indices.size} "
        f"effective_dirichlet={effective_dirichlet_indices.size} "
        f"effective_neumann={effective_neumann_indices.size} "
        f"cross_type_overlap={overlap_indices.size}"
    )

    stage_start = time.perf_counter()
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = baseflow.RHO.ravel(order="F")[p]
    T_p = baseflow.T.ravel(order="F")[p]
    W_diag, _ = compute_chu_norm(W_int, RHO_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp)
    _log_stage("compute_chu_norm", stage_start)

    meta = {
        "mesh_ngp": int(mesh.ngp),
        "fd_order": int(fd_order),
        "omega_target": complex(omega_target),
        "bc_indices": total_bc_indices,
        "dirichlet_indices": effective_dirichlet_indices,
        "solver_cfg": solver_cfg,
    }
    return L0, W_diag, total_bc_indices, effective_dirichlet_indices, complex(omega_target), meta


def _write_summary(summary_rows: list[dict], summary_json: Path, summary_csv: Path) -> None:
    payload = {"generated_at": time.time(), "stages": summary_rows}
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _build_stage_specs(study_cfg: dict) -> list[dict]:
    raw_stages = study_cfg.get("stages")
    if raw_stages:
        stages: list[dict] = []
        for idx, raw in enumerate(raw_stages):
            stage = dict(raw)
            if "nev" not in stage:
                stage["nev"] = int(study_cfg.get("n_modes_stability", 100))
            if "ncv" not in stage:
                raise ValueError(f"stage {idx} is missing required key 'ncv'")
            stage["nev"] = int(stage["nev"])
            stage["ncv"] = int(stage["ncv"])
            stage["name"] = str(stage.get("name", f"stage{idx + 1:02d}"))
            stages.append(stage)
        return stages

    ncv_milestones = [int(val) for val in study_cfg.get("ncv_milestones", [120, 200, 300, 400, 500, 600])]
    nev = int(study_cfg.get("n_modes_stability", 100))
    return [
        {
            "name": f"ncv{ncv}",
            "nev": nev,
            "ncv": int(ncv),
        }
        for ncv in ncv_milestones
    ]


def main() -> int:
    os.environ.setdefault("LINSTAB2D_CASE_LOG_PREFIX", "case2-ncv")
    cfg_path = Path(os.getenv("LINSTAB2D_CASE2_NCV_CONFIG", str(SCRIPT.parent / "ncv_milestones.jsonc"))).resolve()
    study_cfg = load_jsonc(cfg_path)
    base_cfg = load_jsonc((cfg_path.parent / str(study_cfg.get("base_config", "parity_config.jsonc"))).resolve())
    output_dir = (cfg_path.parent / str(study_cfg.get("output_dir", "output/ncv_milestones"))).resolve()
    fd_orders = [int(val) for val in study_cfg.get("fd_orders", [4])]
    stages = _build_stage_specs(study_cfg)
    solver_overrides = dict(study_cfg.get("solver", {}))
    explicit_target = study_cfg.get("omega_target", None)

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    summary_rows: list[dict] = []

    for fd_order in fd_orders:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("numerics", {})["fd_order"] = int(fd_order)
        cfg.setdefault("analysis", {})["n_modes_stability"] = int(stages[0]["nev"])
        cfg["analysis"]["run_stability"] = True
        cfg["analysis"]["run_resolvent"] = False
        if explicit_target is not None:
            cfg["analysis"]["omega_target"] = explicit_target
        eig_cfg = cfg.setdefault("solver", {}).setdefault("eigen", {})
        eig_cfg.update(copy.deepcopy(solver_overrides))

        if rank == 0:
            stage_desc = [
                f"{stage['name']}(nev={stage['nev']},ncv={stage['ncv']},tol={stage.get('eps_tol', eig_cfg.get('eps_tol', 'na'))})"
                for stage in stages
            ]
            _log(f"preparing fd{fd_order} milestone study with stages={stage_desc}")
        L0, W_diag, bc_indices, dirichlet_indices, omega_target, meta = _prepare_case(cfg)

        seed_modes: np.ndarray | None = None
        for stage_index, stage in enumerate(stages):
            stage_nev = int(stage["nev"])
            ncv = int(stage["ncv"])
            stage_solver_cfg = copy.deepcopy(cfg["solver"])
            stage_solver_cfg.setdefault("eigen", {})["ncv"] = int(ncv)
            if "eps_tol" in stage:
                stage_solver_cfg["eigen"]["eps_tol"] = float(stage["eps_tol"])
            if "eps_max_it" in stage:
                stage_solver_cfg["eigen"]["eps_max_it"] = int(stage["eps_max_it"])
            if "solver" in stage:
                stage_solver_cfg["eigen"].update(copy.deepcopy(stage["solver"]))
            if seed_modes is not None and seed_modes.size:
                n_seed = min(seed_modes.shape[1], stage_nev, max(1, int(ncv) - 1))
                initial_space = np.asarray(seed_modes[:, :n_seed], dtype=complex)
            else:
                initial_space = None

            stage_name = str(stage.get("name", f"stage{stage_index + 1:02d}"))
            stage_tol = stage_solver_cfg["eigen"].get("eps_tol", "na")
            stage_id = f"fd{fd_order}_{stage_name}_nev{stage_nev}_ncv{ncv}_tol{stage_tol}"
            stage_start = time.perf_counter()
            _log(
                f"running staged temporal eigensolve: fd={fd_order} stage={stage_name} "
                f"ncv={ncv} nev={stage_nev} target={omega_target.real:+.8e}{omega_target.imag:+.8e}j "
                f"seed_cols={0 if initial_space is None else initial_space.shape[1]}"
            )
            try:
                omega, direct_modes, info = temporal_eigenvalues(
                    L0,
                    n_modes=stage_nev,
                    target=omega_target,
                    W_diag=W_diag,
                    ncv=int(ncv),
                    initial_space=initial_space,
                    solver_cfg=stage_solver_cfg,
                    return_info=True,
                )
            except Exception as exc:
                stage_seconds = float(time.perf_counter() - stage_start)
                if rank == 0:
                    row = {
                        "stage_index": int(stage_index),
                        "stage_name": stage_name,
                        "fd_order": int(fd_order),
                        "requested_nev": int(stage_nev),
                        "ncv": int(ncv),
                        "eps_tol": float(stage_solver_cfg["eigen"].get("eps_tol", np.nan)),
                        "stage_seconds": stage_seconds,
                        "nconv": 0,
                        "eps_iterations": int(stage_solver_cfg["eigen"].get("eps_max_it", -1)),
                        "eps_rel_count_le_1e6": 0,
                        "eps_rel_count_le_1e5": 0,
                        "eps_rel_count_le_1e4": 0,
                        "post_rel_count_le_1e6": 0,
                        "post_rel_count_le_1e5": 0,
                        "post_rel_count_le_1e4": 0,
                        "post_rel_median": np.nan,
                        "post_rel_max": np.nan,
                        "artifact_npz": "",
                        "status": "failed",
                        "error": str(exc),
                    }
                    summary_rows.append(row)
                    _write_summary(
                        summary_rows,
                        output_dir / "case2_ncv_milestones_summary.json",
                        output_dir / "case2_ncv_milestones_summary.csv",
                    )
                    _log(
                        f"stage failed: fd={fd_order} stage={stage_name} ncv={ncv} "
                        f"nev={stage_nev} time={stage_seconds:.2f}s error={exc}"
                    )
                raise
            info.update(
                _postcheck_stability_modes(
                    L0,
                    omega,
                    direct_modes,
                    target=omega_target,
                    bc_indices=bc_indices,
                    constrained_dirichlet_indices=dirichlet_indices,
                )
            )
            stage_seconds = float(time.perf_counter() - stage_start)

            eps_rel = np.asarray(info.get("residual_relative", np.array([], dtype=float)), dtype=float)
            post_rel = np.asarray(info.get("postcheck_residual_rel", np.array([], dtype=float)), dtype=float)
            eps_counts = _count_thresholds(eps_rel)
            post_counts = _count_thresholds(post_rel)
            if rank == 0:
                stage_npz = output_dir / f"{stage_id}.npz"
                stage_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    stage_npz,
                    omega=np.asarray(omega, dtype=complex),
                    omega_target=np.asarray([omega_target], dtype=complex),
                    fd_order=np.int32(fd_order),
                    requested_nev=np.int32(stage_nev),
                    ncv=np.int32(ncv),
                    stage_index=np.int32(stage_index),
                    stage_name=np.asarray([stage_name]),
                    stage_elapsed_seconds=np.float64(stage_seconds),
                    eps_tol=np.float64(stage_solver_cfg["eigen"].get("eps_tol", np.nan)),
                    stability_eps_nconv=np.int32(info.get("nconv", -1)),
                    stability_eps_iterations=np.int32(info.get("eps_iterations", -1)),
                    stability_residual_relative=np.asarray(info.get("residual_relative", np.array([], dtype=float))),
                    stability_residual_backward=np.asarray(info.get("residual_backward", np.array([], dtype=float))),
                    stability_postcheck_residual_rel=post_rel,
                    stability_postcheck_target_distance=np.asarray(
                        info.get("postcheck_target_distance", np.array([], dtype=float))
                    ),
                )

                row = {
                    "stage_index": int(stage_index),
                    "stage_name": stage_name,
                    "fd_order": int(fd_order),
                    "requested_nev": int(stage_nev),
                    "ncv": int(ncv),
                    "eps_tol": float(stage_solver_cfg["eigen"].get("eps_tol", np.nan)),
                    "stage_seconds": stage_seconds,
                    "nconv": int(info.get("nconv", -1)),
                    "eps_iterations": int(info.get("eps_iterations", -1)),
                    "eps_rel_count_le_1e6": eps_counts["count_le_1e6"],
                    "eps_rel_count_le_1e5": eps_counts["count_le_1e5"],
                    "eps_rel_count_le_1e4": eps_counts["count_le_1e4"],
                    "post_rel_count_le_1e6": post_counts["count_le_1e6"],
                    "post_rel_count_le_1e5": post_counts["count_le_1e5"],
                    "post_rel_count_le_1e4": post_counts["count_le_1e4"],
                    "post_rel_median": float(np.median(post_rel[np.isfinite(post_rel)])) if np.isfinite(post_rel).any() else np.nan,
                    "post_rel_max": float(np.max(post_rel[np.isfinite(post_rel)])) if np.isfinite(post_rel).any() else np.nan,
                    "artifact_npz": str(stage_npz),
                    "status": "ok",
                    "error": "",
                }
                summary_rows.append(row)
                _write_summary(
                    summary_rows,
                    output_dir / "case2_ncv_milestones_summary.json",
                    output_dir / "case2_ncv_milestones_summary.csv",
                )
                _log(
                    f"stage complete: fd={fd_order} stage={stage_name} ncv={ncv} "
                    f"nconv={row['nconv']} post<=1e-6={row['post_rel_count_le_1e6']} "
                    f"post<=1e-5={row['post_rel_count_le_1e5']} time={stage_seconds:.2f}s"
                )

            seed_cols = min(direct_modes.shape[1], stage_nev, max(1, int(ncv) - 1))
            seed_modes = np.asarray(direct_modes[:, :seed_cols], dtype=complex).copy()
            comm.Barrier()

        if _is_petsc_matrix(L0):
            L0.destroy()

    if rank == 0:
        _log(f"ncv milestone study finished; summary in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
