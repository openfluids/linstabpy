#!/usr/bin/env python3
"""Case 1: SWBLI (2D compressible LNSE, resolvent and stability).

This script is intentionally written as a single, readable pipeline:
1. Load one prepared Case~1 baseflow from ``.npy`` files.
2. Build/deform the mesh and assemble the compressible LNSE operator.
3. Apply boundary conditions exactly as in the MATLAB workflow.
4. Run optional resolvent and/or stability analysis (PETSc/SLEPc path).
5. Save everything needed for parity checks and plotting into one ``.npz``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from linstab2d.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc
from mpi4py import MPI

from linstab2d.baseflow import load_hamada_case, sutherland_air
from linstab2d.boundary_conditions.recipe import apply_bc_recipe
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.eigenvalue import _eigs_near_target, temporal_eigenvalues
from linstab2d.solvers.resolvent import resolvent_analysis
from linstab2d.utils.jsonc import load_jsonc

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")


def _log(message: str) -> None:
    """Small helper for consistent prefixed progress messages."""
    if PETSc.COMM_WORLD.getRank() == 0:
        print(f"[case1] {message}", flush=True)


def _log_stage(stage: str, stage_start: float) -> None:
    """Emit a machine-parseable stage timing line."""
    _log(f"stage complete: stage={stage} elapsed={time.perf_counter() - stage_start:.2f}s")


def compute() -> None:
    """Run the full Case 1 pipeline and write ``case1.npz``."""
    t0 = time.perf_counter()
    _log("starting compute")
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    config_path = Path(
        os.getenv(
            "LINSTAB2D_CASE1_CONFIG",
            str(SCRIPT.parent / "parity_config.jsonc"),
        )
    )
    config: dict = {}
    if config_path.exists():
        config = load_jsonc(config_path)
        _log(f"loaded config: {config_path}")
    elif "LINSTAB2D_CASE1_CONFIG" in os.environ:
        raise FileNotFoundError(f"LINSTAB2D_CASE1_CONFIG file not found: {config_path}")

    def cfg(path: str, default):
        node = config
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # ------------------------------------------------------------------
    # 1) Read runtime configuration from environment variables.
    #    Defaults here mirror the reference MATLAB setup.
    # ------------------------------------------------------------------
    case_dir = Path(str(cfg("paths.case_dir", str(SCRIPT.parent))))
    sponge_mat_path = Path(str(cfg("paths.case_inputs_mat", "MATLAB_exports_resolvent/case_inputs_used.mat")))
    Re = float(cfg("physics.Re", 395000.0))
    Ma = float(cfg("physics.Ma", 2.0))
    Pr = float(cfg("physics.Pr", 0.747))
    # MATLAB export note:
    # - `case_inputs_used.mat/baseFlow.kappa` is 1.4
    # - `run_metadata.mat/kappa` is 1.31
    # The assembled reference operators/spectrum align with baseFlow.kappa.
    kappa = float(cfg("physics.kappa", 1.4))
    T_0 = float(cfg("physics.T_0", 1540.0))

    stride_ksi = int(cfg("numerics.stride_ksi", 1))
    stride_eta = int(cfg("numerics.stride_eta", 1))
    fd_order = int(cfg("numerics.fd_order", 4))
    beta = float(cfg("numerics.beta", 0.0))
    penalty = float(cfg("bc.penalty", 1e8))

    ratio_u = float(cfg("analysis.ratio_uinf", 1.0))
    ratio_l = float(cfg("analysis.ratio_l", 0.1))
    St = float(cfg("analysis.St", 0.1))

    # Target angular frequency used by resolvent and shift-invert EVP.
    omega_target = 2.0 * np.pi * St * ratio_u / ratio_l

    n_modes = int(cfg("analysis.n_modes_stability", 100))
    n_modes_resolvent = int(cfg("analysis.n_modes_resolvent", 10))
    run_resolvent = bool(cfg("analysis.run_resolvent", True))
    run_stability = bool(cfg("analysis.run_stability", True))
    zero_spanwise = bool(cfg("physics.zero_spanwise", False))
    use_matlab_sponge_aux = bool(cfg("physics.use_matlab_sponge_aux", True))
    alpha_filter_raw = cfg("numerics.alpha_filter", "none")
    try:
        alpha_filter: float | str = float(alpha_filter_raw)
    except (ValueError, TypeError):
        alpha_filter = str(alpha_filter_raw).strip()
    # MATLAB export note:
    # case_inputs_used.mat stores uppercase dMUdT/d2MUdT2 as zeros; GetLinProblem.m
    # consumes those names in getCoeffsLaminar_Cartesian.m. Keep this default True
    # for strict parity with the exported reference matrices/spectrum.
    zero_mu_t_derivs = bool(cfg("physics.zero_mu_t_derivs", True))
    # MATLAB stability convention:
    #   L0_matlab = L0 / (-1i)  => omega = eig(L0_matlab) = eig(1j * L0)
    # Keep this as default for parity with exported MATLAB spectrum.
    use_raw_eigs = bool(cfg("analysis.use_raw_eigs", False))
    solver_cfg = config.get("solver")
    if solver_cfg is None:
        solver_cfg = {}
        config["solver"] = solver_cfg

    # Read solver knobs from JSONC (env-var fallback for backward compat).
    _ecfg = solver_cfg.get("eigen", {})
    st_solver = _ecfg.get("st_solver",
                          os.getenv("LINSTAB2D_EIGEN_ST_SOLVER", "mumps")).strip().lower()
    iterative_st_solver = st_solver in {"iterative", "krylov", "slepc"}
    _root_raw = os.getenv("LINSTAB2D_CASE1_ROOT_ASSEMBLY")
    requested_root_operator = size > 1 and (
        (_root_raw.strip().lower() in {"1", "true", "yes", "on"}) if _root_raw is not None else iterative_st_solver
    )
    use_root_operator = requested_root_operator and iterative_st_solver
    if requested_root_operator and not iterative_st_solver:
        _log(
            "root-assembly mode requested but disabled because "
            f"eigen.st_solver={st_solver!r} is not iterative"
        )
    if use_root_operator and run_resolvent:
        raise RuntimeError(
            "Root-only operator assembly mode is currently intended for stability solves only. "
            "Set LINSTAB2D_CASE1_ROOT_ASSEMBLY=0 when running resolvent in MPI."
        )
    if use_root_operator:
        solver_cfg.setdefault("root_scatter", True)

    _linsolve = solver_cfg.get("linsolve", {})
    _resolvent_cfg = solver_cfg.get("resolvent", _linsolve)
    resolvent_type = _resolvent_cfg.get("type", _linsolve.get("type", "mumps"))
    matrix_mode = solver_cfg.get("petsc_matrix_mode", "auto")

    _log(
        f"config: case_dir={case_dir} fd_order={fd_order} n_modes={n_modes} "
        f"run_stability={run_stability} run_resolvent={run_resolvent} "
        f"raw_eigs={use_raw_eigs} zero_mu_t_derivs={zero_mu_t_derivs} "
        f"alpha_filter={alpha_filter} matlab_sponge_aux={use_matlab_sponge_aux}"
    )
    _log(
        "solver setup: "
        f"mpi_ranks={size} "
        f"st_solver={st_solver} "
        f"resolvent_solver={resolvent_type} "
        f"root_assembly={use_root_operator} "
        f"matrix_mode={matrix_mode}"
    )

    # ------------------------------------------------------------------
    # 2) Load one case snapshot (x, y, rho, u, v, w, T, sponge) and
    #    compute viscosity/transport via Sutherland law.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3) Build a computational mesh, then deform it to physical X/Y.
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()
    ny, nx = X.shape
    cmesh = Mesh.create(
        (0.0, 1.0),
        (0.0, 0.5),
        nx=nx,
        ny=ny,
        fd_order=fd_order,
        alpha_filter=alpha_filter,
    )
    mesh = cmesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge
    _log_stage("build_mesh", stage_start)

    # ------------------------------------------------------------------
    # 4) Assemble LNSE operator and impose boundary conditions.
    # ------------------------------------------------------------------
    if use_root_operator and rank != 0:
        # Root-assembly mode for MPI scalability:
        # non-root ranks keep only a shape-compatible placeholder; the real
        # SciPy->PETSc operator transfer happens from root inside eigenvalue.py.
        ndof = mesh.ngp * 5
        L0 = sp.csc_matrix((ndof, ndof), dtype=complex)
        idx = {
            "rho_j": np.arange(0, mesh.ngp, dtype=np.int64),
            "u_j": np.arange(mesh.ngp, 2 * mesh.ngp, dtype=np.int64),
            "v_j": np.arange(2 * mesh.ngp, 3 * mesh.ngp, dtype=np.int64),
            "w_j": np.arange(3 * mesh.ngp, 4 * mesh.ngp, dtype=np.int64),
            "T_j": np.arange(4 * mesh.ngp, 5 * mesh.ngp, dtype=np.int64),
        }
        bc_indices = np.array([], dtype=np.int64)
    else:
        stage_start = time.perf_counter()
        _log("assembling LNSE operator")
        L0, idx = build_lnse_operator(mesh, baseflow, kx=beta, model="2D")
        _log(f"operator assembled: shape={L0.shape}, nnz={L0.nnz}")
        _log_stage("assemble_operator", stage_start)
        stage_start = time.perf_counter()
        bc_recipe = cfg("bc.recipe", [
            {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
            {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
            {"type": "neumann", "borders": "b", "variables": "T"},
            {"type": "neumann", "borders": "r", "variables": "ruvwT"},
        ])
        L0, bc_indices = apply_bc_recipe(L0, mesh, idx, bc_recipe, penalty=penalty)
        _log(f"boundary conditions applied: bc_dofs={bc_indices.size}, nnz={L0.nnz}")
        _log_stage("apply_bc", stage_start)

    # ------------------------------------------------------------------
    # 5) Build Chu energy norm weights. These weights are required for
    #    weighted analyses (resolvent and temporal formulation).
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = baseflow.RHO.ravel(order="F")[p]
    T_p = baseflow.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp)
    _log_stage("compute_chu_norm", stage_start)

    gains = np.array([], dtype=float)
    responses = np.empty((mesh.ngp * 5, 0), dtype=complex)
    forcings = np.empty((mesh.ngp * 5, 0), dtype=complex)

    # ------------------------------------------------------------------
    # 6) Optional resolvent analysis at omega_target.
    # ------------------------------------------------------------------
    if run_resolvent:
        stage_start = time.perf_counter()
        _log(f"running resolvent: omega_target={omega_target:.8g}, modes={n_modes_resolvent}")
        B_diag = np.ones(mesh.ngp * 5, dtype=float)
        B_diag[bc_indices] = 0.0
        B_mat = sp.diags(B_diag, format="csc")
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
        _log("resolvent analysis finished")
        _log_stage("resolvent", stage_start)

    omega = np.array([], dtype=complex)
    direct_modes = np.empty((mesh.ngp * 5, 0), dtype=complex)

    # ------------------------------------------------------------------
    # 7) Optional stability analysis.
    #    - raw_eigs: eig(L0) directly (diagnostic-only path, not MATLAB parity)
    #    - temporal: eig(i*L0), which matches MATLAB's L0 = L0 / -1i convention
    # ------------------------------------------------------------------
    if run_stability:
        stage_start = time.perf_counter()
        if use_raw_eigs:
            _log(
                "running raw eigensolve on L0 "
                f"(diagnostic path): target={omega_target:.8g}, modes={n_modes}"
            )
            omega, direct_modes = _eigs_near_target(
                L0.tocsc(),
                n_modes=n_modes,
                target=omega_target,
                solver_cfg=solver_cfg,
            )
            order = np.argsort(np.abs(omega - omega_target))
            omega = omega[order]
            direct_modes = direct_modes[:, order]
        else:
            _log(
                "running temporal eigensolve on i*L0 "
                f"(MATLAB convention): target={omega_target:.8g}, modes={n_modes}"
            )
            omega, direct_modes = temporal_eigenvalues(
                L0,
                n_modes=n_modes,
                target=omega_target,
                W_diag=W_diag,
                solver_cfg=solver_cfg,
            )
        _log(f"stability analysis finished: converged_modes={omega.size}")
        _log_stage("stability", stage_start)

    # ------------------------------------------------------------------
    # 8) Save full run state to one NPZ for downstream parity/plots.
    # ------------------------------------------------------------------
    if rank == 0:
        stage_start = time.perf_counter()
        np.savez_compressed(
            OUTPUT_NPZ,
            omega=omega,
            direct_modes=direct_modes,
            gains=gains,
            responses=responses,
            forcings=forcings,
            omega_target=omega_target,
            use_raw_eigs=use_raw_eigs,
            St=St,
            beta=beta,
            Re=Re,
            Ma=Ma,
            Pr=Pr,
            kappa=kappa,
            T_0=T_0,
            X=mesh.X,
            Y=mesh.Y,
            U=baseflow.U,
            sponge=mesh.sponge,
            idx_rho_j=idx["rho_j"],
            idx_u_j=idx["u_j"],
            idx_v_j=idx["v_j"],
            idx_w_j=idx["w_j"],
            idx_T_j=idx["T_j"],
            bc_indices=bc_indices,
            W_diag=W_diag,
            invW_diag=invW_diag,
        )
        _log_stage("save_results", stage_start)
    elapsed = time.perf_counter() - t0
    _log(f"results saved to {OUTPUT_NPZ}")
    _log(f"compute finished in {elapsed:.2f}s")
    comm.barrier()


def main() -> None:
    """Cache-aware entrypoint used when launched as ``python ...``."""
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    need_recompute = False
    if rank == 0:
        try:
            np.load(OUTPUT_NPZ, allow_pickle = False)
        except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
            print(f"Cached results unavailable or invalid ({exc}), recomputing.")
            need_recompute = True
    need_recompute = MPI.COMM_WORLD.bcast(need_recompute, root=0)
    if need_recompute:
        compute()


if __name__ == "__main__":
    main()
