#!/usr/bin/env python3
"""Opt-in Case 2 shock-cleanup experiment runner.

This script leaves the default Case 2 workflow untouched. It loads the raw
baseflow, optionally applies the experimental mean-flow cleanup, and compares
baseline vs cleaned stability/resolvent outputs plus mean-flow plots.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

SCRIPT = Path(__file__).resolve()
from linstab2d.results_runtime import (
    _effective_alpha_filter,
    _expand_reduced_modes,
    _log,
    _log_stage,
    _postcheck_stability_modes,
    _resolve_omega_target,
    _restrict_operator_for_stability,
)
from linstab2d.baseflow import BaseFlow, load_hamada_case, sutherland_air
from linstab2d.baseflow.shock_cleanup import apply_optional_shock_cleanup
from linstab2d.boundary_conditions import analyze_bc_recipe
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_system
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.eigenvalue import _eigs_near_target, temporal_eigenvalues
from linstab2d.solvers.petsc_utils import _is_petsc_matrix, scipy_to_petsc
from linstab2d.solvers.resolvent import resolvent_analysis
from linstab2d.utils.jsonc import load_jsonc
from linstab2d.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cfg_get(config: dict[str, Any], path: str, default):
    node: Any = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _load_experiment_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    experiment_cfg = load_jsonc(path)
    base_config_rel = experiment_cfg.get("base_config", "parity_config.jsonc")
    base_config_path = (path.parent / str(base_config_rel)).resolve()
    base_cfg = load_jsonc(base_config_path)
    overrides = experiment_cfg.get("config_overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise TypeError("config_overrides must be a JSON object when provided")
    merged = _deep_merge(base_cfg, overrides)
    return experiment_cfg, merged, base_config_path


def _rebuild_baseflow(
    template: BaseFlow,
    *,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    zero_mu_t_derivs: bool,
) -> BaseFlow:
    baseflow = BaseFlow(
        Re=float(template.Re),
        Ma=float(template.Ma),
        Pr=float(template.Pr),
        kappa=float(template.kappa),
        T_0=float(template.T_0),
        RHO=np.array(rho, copy=True),
        U=np.array(u, copy=True),
        V=np.array(v, copy=True),
        W=np.array(w, copy=True),
        T=np.array(T, copy=True),
    )
    sutherland_air(baseflow)
    if zero_mu_t_derivs:
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0
    return baseflow


def _load_case_state(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, BaseFlow, np.ndarray, dict[str, Any]]:
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
    zero_spanwise = bool(_cfg_get(config, "physics.zero_spanwise", False))
    use_matlab_sponge_aux = bool(_cfg_get(config, "physics.use_matlab_sponge_aux", False))
    zero_mu_t_derivs = bool(_cfg_get(config, "physics.zero_mu_t_derivs", True))

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
        f"loaded raw case arrays: shape={X.shape}, points={X.size}, "
        f"sponge_max={float(np.max(sponge)):.3g}, sponge_nnz={int(np.count_nonzero(sponge))}"
    )
    _log_stage("load_case", stage_start)

    meta = {
        "case_dir": str(case_dir),
        "zero_mu_t_derivs": zero_mu_t_derivs,
    }
    return X, Y, baseflow, sponge, meta


def _sanitize_cleanup_info(info: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, dict):
            sanitized[key] = _sanitize_cleanup_info(value)
        elif isinstance(value, (np.floating, np.integer)):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value
    return sanitized


def _run_branch(
    name: str,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    baseflow: BaseFlow,
    sponge: np.ndarray,
    config: dict[str, Any],
    run_stability: bool,
    run_resolvent: bool,
    save_mode_count: int,
    output_dir: Path,
    cleanup_summary: str,
) -> dict[str, Any]:
    fd_order = int(_cfg_get(config, "numerics.fd_order", 4))
    beta = float(_cfg_get(config, "numerics.beta", 0.0))
    penalty = float(_cfg_get(config, "bc.penalty", 1e6))
    alpha_filter_raw = _cfg_get(config, "numerics.alpha_filter", 0.0)
    alpha_filter = _effective_alpha_filter(
        alpha_filter_raw,
        run_resolvent=run_resolvent,
        context=name,
    )

    ratio_u = float(_cfg_get(config, "analysis.ratio_uinf", 1.0))
    ratio_l = float(_cfg_get(config, "analysis.ratio_l", 0.1))
    St = float(_cfg_get(config, "analysis.St", 0.1))
    omega_target = _resolve_omega_target(
        _cfg_get(config, "analysis.omega_target", None),
        2.0 * np.pi * St * ratio_u / ratio_l,
    )
    n_modes_stability = int(_cfg_get(config, "analysis.n_modes_stability", 50))
    n_modes_resolvent = int(_cfg_get(config, "analysis.n_modes_resolvent", 10))
    use_raw_eigs = bool(_cfg_get(config, "analysis.use_raw_eigs", False))
    assembly_backend = _cfg_get(config, "assembly_backend", None)

    solver_cfg = copy.deepcopy(config.get("solver", {}))
    eigen_cfg = solver_cfg.get("eigen", {})
    reduce_dirichlet = bool(eigen_cfg.get("reduce_dirichlet", False))
    reduce_dirichlet_set = str(eigen_cfg.get("reduce_dirichlet_set", "column_zeroed")).strip().lower()

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
    ny, nx = X.shape
    cmesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=nx, ny=ny, fd_order=fd_order, alpha_filter=alpha_filter)
    mesh = cmesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = np.array(sponge, copy=True)
    _log_stage(f"{name}_build_mesh", stage_start)

    stage_start = time.perf_counter()
    _log(f"{name}: assembling LNSE operator")
    L0, idx, bc_indices = build_lnse_system(
        mesh,
        baseflow,
        kx=beta,
        model="2D",
        bc_recipe=bc_recipe,
        bc_penalty=penalty,
        backend=assembly_backend,
        solver_cfg=solver_cfg,
    )
    nnz = int(L0.getInfo()["nz_used"]) if hasattr(L0, "getInfo") else int(L0.nnz)
    _log(f"{name}: operator assembled with nnz={nnz}")
    _log_stage(f"{name}_assemble_operator_bc", stage_start)

    bc_analysis = analyze_bc_recipe(mesh, idx, bc_recipe)
    effective_dirichlet_indices = np.asarray(
        bc_analysis.get("effective_dirichlet_indices", np.array([], dtype=np.int64)),
        dtype=np.int64,
    )
    column_zeroed_dirichlet_indices = np.asarray(
        bc_analysis.get("column_zeroed_dirichlet_indices", np.array([], dtype=np.int64)),
        dtype=np.int64,
    )
    total_bc_indices = np.asarray(
        bc_analysis.get("total_bc_indices", np.array([], dtype=np.int64)),
        dtype=np.int64,
    )
    constrained_dirichlet_indices = (
        column_zeroed_dirichlet_indices
        if reduce_dirichlet_set == "column_zeroed"
        else effective_dirichlet_indices
    )

    stage_start = time.perf_counter()
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = baseflow.RHO.ravel(order="F")[p]
    T_p = baseflow.T.ravel(order="F")[p]
    W_diag, _ = compute_chu_norm(W_int, RHO_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp)
    _log_stage(f"{name}_compute_chu_norm", stage_start)

    omega = np.array([], dtype=complex)
    direct_modes = np.empty((mesh.ngp * 5, 0), dtype=complex)
    stability_info: dict[str, Any] = {}

    if run_stability:
        stage_start = time.perf_counter()
        if use_raw_eigs:
            if _is_petsc_matrix(L0):
                raise RuntimeError(
                    "analysis.use_raw_eigs=true is only supported with a SciPy matrix in the shock-cleanup experiment"
                )
            omega, direct_modes, stability_info = _eigs_near_target(
                L0.tocsc(),
                n_modes=n_modes_stability,
                target=omega_target,
                solver_cfg=solver_cfg,
                return_info=True,
            )
        elif reduce_dirichlet and constrained_dirichlet_indices.size:
            L0_stab, W_stab, free_dofs = _restrict_operator_for_stability(
                L0,
                W_diag,
                constrained_dirichlet_indices,
            )
            try:
                omega, reduced_modes, stability_info = temporal_eigenvalues(
                    L0_stab,
                    n_modes=n_modes_stability,
                    target=omega_target,
                    W_diag=W_stab,
                    solver_cfg=solver_cfg,
                    return_info=True,
                )
            finally:
                if _is_petsc_matrix(L0_stab):
                    L0_stab.destroy()
            direct_modes = _expand_reduced_modes(reduced_modes, mesh.ngp * 5, free_dofs)
        else:
            omega, direct_modes, stability_info = temporal_eigenvalues(
                L0,
                n_modes=n_modes_stability,
                target=omega_target,
                W_diag=W_diag,
                solver_cfg=solver_cfg,
                return_info=True,
            )

        order = np.argsort(np.abs(np.asarray(omega, dtype=complex) - complex(omega_target)))
        omega = np.asarray(omega, dtype=complex)[order]
        if direct_modes.size:
            direct_modes = np.asarray(direct_modes, dtype=complex)[:, order]

        for key in (
            "residual_estimates",
            "residual_absolute",
            "residual_relative",
            "residual_backward",
        ):
            values = np.asarray(stability_info.get(key, np.array([], dtype=float)), dtype=float)
            if values.size == omega.size:
                stability_info[key] = values[order]

        stability_info.update(
            _postcheck_stability_modes(
                L0,
                omega,
                direct_modes,
                target=omega_target,
                bc_indices=total_bc_indices,
                constrained_dirichlet_indices=constrained_dirichlet_indices,
            )
        )
        _log(f"{name}: stability finished with {omega.size} modes")
        _log_stage(f"{name}_stability", stage_start)

    gains = np.array([], dtype=float)
    responses = np.empty((mesh.ngp * 5, 0), dtype=complex)
    forcings = np.empty((mesh.ngp * 5, 0), dtype=complex)
    if run_resolvent:
        stage_start = time.perf_counter()
        B_diag = np.ones(mesh.ngp * 5, dtype=float)
        B_diag[total_bc_indices] = 0.0
        B_mat_sp = sp.diags(B_diag, format="csc")
        B_mat = scipy_to_petsc(B_mat_sp, solver_cfg) if hasattr(L0, "getInfo") else B_mat_sp
        C_mat = B_mat.copy()
        gains, responses, forcings = resolvent_analysis(
            L0,
            omega_target,
            n_modes_resolvent,
            W_diag,
            B_mat=B_mat,
            C_mat=C_mat,
            filter_fn=mesh.filters.filter if mesh.filters is not None else None,
            filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
            solver_cfg=solver_cfg,
        )
        _log(f"{name}: resolvent finished with {gains.size} gains")
        _log_stage(f"{name}_resolvent", stage_start)

    branch = {
        "name": name,
        "X": np.array(X, copy=True),
        "Y": np.array(Y, copy=True),
        "rho": np.array(baseflow.RHO, copy=True),
        "u": np.array(baseflow.U, copy=True),
        "v": np.array(baseflow.V, copy=True),
        "w": np.array(baseflow.W, copy=True),
        "T": np.array(baseflow.T, copy=True),
        "sponge": np.array(sponge, copy=True),
        "omega_target": complex(omega_target),
        "omega": omega,
        "direct_modes": direct_modes[:, :save_mode_count] if direct_modes.size else direct_modes,
        "gains": np.asarray(gains, dtype=float),
        "responses": responses[:, :save_mode_count] if responses.size else responses,
        "forcings": forcings[:, :save_mode_count] if forcings.size else forcings,
        "idx_u_j": np.asarray(idx["u_j"], dtype=np.int64),
        "cleanup_summary": cleanup_summary,
        "stability_info": stability_info,
    }

    if PETSc.COMM_WORLD.getRank() == 0:
        branch_path = output_dir / f"{name}.npz"
        np.savez_compressed(
            branch_path,
            X=branch["X"],
            Y=branch["Y"],
            rho=branch["rho"],
            u=branch["u"],
            v=branch["v"],
            w=branch["w"],
            T=branch["T"],
            sponge=branch["sponge"],
            omega=branch["omega"],
            direct_modes=branch["direct_modes"],
            gains=branch["gains"],
            responses=branch["responses"],
            forcings=branch["forcings"],
            omega_target=branch["omega_target"],
            idx_u_j=branch["idx_u_j"],
            cleanup_summary=np.array(str(cleanup_summary)),
        )
        _log(f"{name}: saved branch results to {branch_path}")
    return branch


def _draw_field(ax, X: np.ndarray, Y: np.ndarray, field: np.ndarray, title: str, *, vmin=None, vmax=None, cmap="coolwarm"):
    pcm = ax.pcolormesh(X, Y, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    return pcm


def _plot_baseflow_comparison(
    baseline: dict[str, Any],
    cleaned: dict[str, Any],
    *,
    fields: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(fields), 3, figsize=(12.5, 3.6 * len(fields)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, field_name in enumerate(fields):
        raw = np.asarray(baseline[field_name], dtype=float)
        clean = np.asarray(cleaned[field_name], dtype=float)
        delta = clean - raw
        vmin = float(min(np.min(raw), np.min(clean)))
        vmax = float(max(np.max(raw), np.max(clean)))
        dmax = float(np.max(np.abs(delta))) if delta.size else 0.0
        pcm0 = _draw_field(axes[row, 0], baseline["X"], baseline["Y"], raw, f"{field_name} raw", vmin=vmin, vmax=vmax)
        pcm1 = _draw_field(axes[row, 1], cleaned["X"], cleaned["Y"], clean, f"{field_name} cleaned", vmin=vmin, vmax=vmax)
        pcm2 = _draw_field(
            axes[row, 2],
            cleaned["X"],
            cleaned["Y"],
            delta,
            f"{field_name} delta",
            vmin=-dmax,
            vmax=dmax,
        )
        fig.colorbar(pcm0, ax=axes[row, 0], shrink=0.85)
        fig.colorbar(pcm1, ax=axes[row, 1], shrink=0.85)
        fig.colorbar(pcm2, ax=axes[row, 2], shrink=0.85)
    fig.suptitle("Case 2 mean-flow comparison", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_mask_summary(
    X: np.ndarray,
    Y: np.ndarray,
    cleanup_info: dict[str, Any],
    *,
    output_path: Path,
) -> None:
    panels = [
        ("sensor mask", np.asarray(cleanup_info.get("mask", np.zeros_like(X, dtype=bool)), dtype=float), "viridis"),
        (
            "buffered mask",
            np.asarray(cleanup_info.get("buffered_mask", np.zeros_like(X, dtype=bool)), dtype=float),
            "viridis",
        ),
        (
            "preserve weight",
            np.asarray(cleanup_info.get("preserve_weight", np.zeros_like(X, dtype=float)), dtype=float),
            "magma",
        ),
    ]
    shock_sponge = np.asarray(cleanup_info.get("shock_sponge", np.zeros_like(X, dtype=float)), dtype=float)
    if np.any(shock_sponge):
        panels.append(("shock sponge", shock_sponge, "coolwarm"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (title, field, cmap) in zip(axes, panels, strict=True):
        vmin = 0.0
        vmax = float(np.max(field)) if np.max(field) > 0.0 else 1.0
        pcm = _draw_field(ax, X, Y, field, title, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(pcm, ax=ax, shrink=0.85)
    fig.suptitle(str(cleanup_info.get("summary", "shock cleanup diagnostics")), fontsize=12)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _extract_u_mode(branch: dict[str, Any], key: str) -> np.ndarray | None:
    modes = np.asarray(branch.get(key, np.empty((0, 0), dtype=complex)), dtype=complex)
    if modes.ndim != 2 or modes.shape[1] == 0:
        return None
    idx_u = np.asarray(branch["idx_u_j"], dtype=int)
    return np.real(modes[idx_u, 0].reshape(branch["X"].shape, order="F"))


def _plot_stability_compare(
    baseline: dict[str, Any],
    cleaned: dict[str, Any],
    *,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    omega_raw = np.asarray(baseline.get("omega", np.array([], dtype=complex)), dtype=complex)
    omega_clean = np.asarray(cleaned.get("omega", np.array([], dtype=complex)), dtype=complex)
    axes[0].scatter(omega_raw.real, omega_raw.imag, label="raw", s=20, facecolors="none", edgecolors="tab:blue")
    axes[0].scatter(omega_clean.real, omega_clean.imag, label="cleaned", s=20, color="tab:orange")
    target = complex(baseline.get("omega_target", 0.0 + 0.0j))
    axes[0].scatter([target.real], [target.imag], marker="x", s=60, color="black", label="target")
    axes[0].set_title("Temporal spectrum")
    axes[0].set_xlabel("Re(omega)")
    axes[0].set_ylabel("Im(omega)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    raw_mode = _extract_u_mode(baseline, "direct_modes")
    clean_mode = _extract_u_mode(cleaned, "direct_modes")
    if raw_mode is not None and clean_mode is not None:
        vlim = float(max(np.max(np.abs(raw_mode)), np.max(np.abs(clean_mode))))
        pcm1 = _draw_field(axes[1], baseline["X"], baseline["Y"], raw_mode, "Raw leading direct u", vmin=-vlim, vmax=vlim)
        pcm2 = _draw_field(axes[2], cleaned["X"], cleaned["Y"], clean_mode, "Cleaned leading direct u", vmin=-vlim, vmax=vlim)
        fig.colorbar(pcm1, ax=axes[1], shrink=0.85)
        fig.colorbar(pcm2, ax=axes[2], shrink=0.85)
    else:
        axes[1].set_title("Raw leading direct u")
        axes[1].text(0.5, 0.5, "no stability modes", ha="center", va="center", transform=axes[1].transAxes)
        axes[2].set_title("Cleaned leading direct u")
        axes[2].text(0.5, 0.5, "no stability modes", ha="center", va="center", transform=axes[2].transAxes)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_resolvent_compare(
    baseline: dict[str, Any],
    cleaned: dict[str, Any],
    *,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    gains_raw = np.asarray(baseline.get("gains", np.array([], dtype=float)), dtype=float)
    gains_clean = np.asarray(cleaned.get("gains", np.array([], dtype=float)), dtype=float)
    n = max(gains_raw.size, gains_clean.size)
    if n > 0:
        idx = np.arange(1, n + 1, dtype=int)
        if gains_raw.size:
            axes[0].plot(np.arange(1, gains_raw.size + 1), gains_raw, "o-", label="raw")
        if gains_clean.size:
            axes[0].plot(np.arange(1, gains_clean.size + 1), gains_clean, "s-", label="cleaned")
        axes[0].set_xticks(idx)
    axes[0].set_title("Resolvent gains")
    axes[0].set_xlabel("mode index")
    axes[0].set_ylabel("gain")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    raw_response = _extract_u_mode(baseline, "responses")
    clean_response = _extract_u_mode(cleaned, "responses")
    if raw_response is not None and clean_response is not None:
        vlim = float(max(np.max(np.abs(raw_response)), np.max(np.abs(clean_response))))
        pcm1 = _draw_field(
            axes[1],
            baseline["X"],
            baseline["Y"],
            raw_response,
            "Raw leading response u",
            vmin=-vlim,
            vmax=vlim,
        )
        pcm2 = _draw_field(
            axes[2],
            cleaned["X"],
            cleaned["Y"],
            clean_response,
            "Cleaned leading response u",
            vmin=-vlim,
            vmax=vlim,
        )
        fig.colorbar(pcm1, ax=axes[1], shrink=0.85)
        fig.colorbar(pcm2, ax=axes[2], shrink=0.85)
    else:
        axes[1].set_title("Raw leading response u")
        axes[1].text(0.5, 0.5, "no resolvent modes", ha="center", va="center", transform=axes[1].transAxes)
        axes[2].set_title("Cleaned leading response u")
        axes[2].text(0.5, 0.5, "no resolvent modes", ha="center", va="center", transform=axes[2].transAxes)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_summary(
    baseline: dict[str, Any],
    cleaned: dict[str, Any],
    *,
    cleanup_info: dict[str, Any],
    config_path: Path,
    base_config_path: Path,
) -> dict[str, Any]:
    omega_raw = np.asarray(baseline.get("omega", np.array([], dtype=complex)), dtype=complex)
    omega_clean = np.asarray(cleaned.get("omega", np.array([], dtype=complex)), dtype=complex)
    gains_raw = np.asarray(baseline.get("gains", np.array([], dtype=float)), dtype=float)
    gains_clean = np.asarray(cleaned.get("gains", np.array([], dtype=float)), dtype=float)

    comparison: dict[str, Any] = {}
    common_omega = min(omega_raw.size, omega_clean.size)
    if common_omega > 0:
        delta = np.abs(omega_clean[:common_omega] - omega_raw[:common_omega])
        comparison["stability_common_modes"] = int(common_omega)
        comparison["stability_mode0_abs_shift"] = float(delta[0])
        comparison["stability_common_abs_shift_max"] = float(np.max(delta))
        comparison["stability_common_abs_shift_median"] = float(np.median(delta))
    common_gains = min(gains_raw.size, gains_clean.size)
    if common_gains > 0:
        gain_ratio = gains_clean[:common_gains] / np.maximum(gains_raw[:common_gains], np.finfo(float).eps)
        comparison["resolvent_common_modes"] = int(common_gains)
        comparison["resolvent_gain0_ratio"] = float(gain_ratio[0])
        comparison["resolvent_gain_ratio_max"] = float(np.max(gain_ratio))
        comparison["resolvent_gain_ratio_min"] = float(np.min(gain_ratio))

    return {
        "generated_at": time.time(),
        "config_path": str(config_path),
        "base_config_path": str(base_config_path),
        "cleanup": _sanitize_cleanup_info(cleanup_info),
        "baseline": {
            "omega_count": int(omega_raw.size),
            "gain_count": int(gains_raw.size),
            "cleanup_summary": str(baseline.get("cleanup_summary", "")),
        },
        "cleaned": {
            "omega_count": int(omega_clean.size),
            "gain_count": int(gains_clean.size),
            "cleanup_summary": str(cleaned.get("cleanup_summary", "")),
        },
        "comparison": comparison,
    }


def main() -> int:
    os.environ.setdefault("LINSTAB2D_CASE_LOG_PREFIX", "case2-shock")
    cfg_path = Path(
        os.getenv(
            "LINSTAB2D_CASE2_SHOCK_EXPERIMENT_CONFIG",
            str(SCRIPT.parent / "shock_cleanup_experiment.jsonc"),
        )
    ).resolve()
    experiment_cfg, base_cfg, base_config_path = _load_experiment_config(cfg_path)

    experiment = experiment_cfg.get("experiment", {})
    if experiment is None:
        experiment = {}
    if not isinstance(experiment, dict):
        raise TypeError("experiment must be a JSON object when provided")

    output_dir = Path(str(experiment_cfg.get("output_dir", "output/shock_cleanup_experiment")))
    if not output_dir.is_absolute():
        output_dir = (SCRIPT.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_baseline = bool(experiment.get("run_baseline", True))
    run_cleaned = bool(experiment.get("run_cleaned", True))
    run_stability = bool(experiment.get("run_stability", bool(_cfg_get(base_cfg, "analysis.run_stability", True))))
    run_resolvent = bool(experiment.get("run_resolvent", bool(_cfg_get(base_cfg, "analysis.run_resolvent", True))))
    plot_fields = [str(name) for name in experiment.get("plot_fields", ["rho", "u", "v", "T"])]
    save_mode_count = int(experiment.get("save_mode_count", 3))

    X, Y, raw_baseflow, raw_sponge, load_meta = _load_case_state(base_cfg)
    cleanup_cfg = experiment_cfg.get("shock_cleanup")
    rho_c, u_c, v_c, w_c, T_c, sponge_c, cleanup_info = apply_optional_shock_cleanup(
        X,
        Y,
        raw_baseflow.RHO,
        raw_baseflow.U,
        raw_baseflow.V,
        raw_baseflow.W,
        raw_baseflow.T,
        raw_sponge,
        cleanup_cfg,
    )
    cleaned_baseflow = _rebuild_baseflow(
        raw_baseflow,
        rho=rho_c,
        u=u_c,
        v=v_c,
        w=w_c,
        T=T_c,
        zero_mu_t_derivs=bool(load_meta["zero_mu_t_derivs"]),
    )

    if PETSc.COMM_WORLD.getRank() == 0:
        _plot_baseflow_comparison(
            {
                "X": X,
                "Y": Y,
                "rho": raw_baseflow.RHO,
                "u": raw_baseflow.U,
                "v": raw_baseflow.V,
                "w": raw_baseflow.W,
                "T": raw_baseflow.T,
                "sponge": raw_sponge,
                "cleanup_summary": "disabled",
            },
            {
                "X": X,
                "Y": Y,
                "rho": cleaned_baseflow.RHO,
                "u": cleaned_baseflow.U,
                "v": cleaned_baseflow.V,
                "w": cleaned_baseflow.W,
                "T": cleaned_baseflow.T,
                "sponge": sponge_c,
                "cleanup_summary": str(cleanup_info.get("summary", "disabled")),
            },
            fields=plot_fields,
            output_path=output_dir / "baseflow_compare.png",
        )
        _plot_mask_summary(
            X,
            Y,
            cleanup_info,
            output_path=output_dir / "cleanup_mask.png",
        )

    baseline_branch: dict[str, Any] | None = None
    cleaned_branch: dict[str, Any] | None = None
    if run_baseline:
        baseline_branch = _run_branch(
            "raw",
            X=X,
            Y=Y,
            baseflow=raw_baseflow,
            sponge=raw_sponge,
            config=base_cfg,
            run_stability=run_stability,
            run_resolvent=run_resolvent,
            save_mode_count=save_mode_count,
            output_dir=output_dir,
            cleanup_summary="disabled",
        )
    if run_cleaned:
        cleaned_branch = _run_branch(
            "cleaned",
            X=X,
            Y=Y,
            baseflow=cleaned_baseflow,
            sponge=sponge_c,
            config=base_cfg,
            run_stability=run_stability,
            run_resolvent=run_resolvent,
            save_mode_count=save_mode_count,
            output_dir=output_dir,
            cleanup_summary=str(cleanup_info.get("summary", "disabled")),
        )

    if PETSc.COMM_WORLD.getRank() == 0:
        reference_baseline = baseline_branch or {
            "X": X,
            "Y": Y,
            "rho": raw_baseflow.RHO,
            "u": raw_baseflow.U,
            "v": raw_baseflow.V,
            "w": raw_baseflow.W,
            "T": raw_baseflow.T,
            "sponge": raw_sponge,
            "omega": np.array([], dtype=complex),
            "gains": np.array([], dtype=float),
            "direct_modes": np.empty((0, 0), dtype=complex),
            "responses": np.empty((0, 0), dtype=complex),
            "idx_u_j": np.array([], dtype=int),
            "omega_target": 0.0 + 0.0j,
            "cleanup_summary": "disabled",
        }
        reference_cleaned = cleaned_branch or {
            "X": X,
            "Y": Y,
            "rho": cleaned_baseflow.RHO,
            "u": cleaned_baseflow.U,
            "v": cleaned_baseflow.V,
            "w": cleaned_baseflow.W,
            "T": cleaned_baseflow.T,
            "sponge": sponge_c,
            "omega": np.array([], dtype=complex),
            "gains": np.array([], dtype=float),
            "direct_modes": np.empty((0, 0), dtype=complex),
            "responses": np.empty((0, 0), dtype=complex),
            "idx_u_j": np.array([], dtype=int),
            "omega_target": 0.0 + 0.0j,
            "cleanup_summary": str(cleanup_info.get("summary", "disabled")),
        }

        if run_stability and baseline_branch is not None and cleaned_branch is not None:
            _plot_stability_compare(
                baseline_branch,
                cleaned_branch,
                output_path=output_dir / "stability_compare.png",
            )
        if run_resolvent and baseline_branch is not None and cleaned_branch is not None:
            _plot_resolvent_compare(
                baseline_branch,
                cleaned_branch,
                output_path=output_dir / "resolvent_compare.png",
            )

        summary = _build_summary(
            reference_baseline,
            reference_cleaned,
            cleanup_info=cleanup_info,
            config_path=cfg_path,
            base_config_path=base_config_path,
        )
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _log(f"saved experiment summary to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
