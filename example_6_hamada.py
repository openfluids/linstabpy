#!/usr/bin/env python3
"""Example 6: Hamada case (2D compressible LNSE, resolvent and stability).

This script is intentionally written as a single, readable pipeline:
1. Load one prepared Hamada baseflow case from ``.npy`` files.
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

from linstab2d.baseflow import load_hamada_case, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.boundary_conditions.neumann import apply_neumann
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.eigenvalue import _eigs_near_target, temporal_eigenvalues
from linstab2d.solvers.resolvent import resolvent_analysis
from linstab2d.examples._helpers import env_bool, env_float, env_int

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")


def _log(message: str) -> None:
    """Small helper for consistent prefixed progress messages."""
    print(f"[example_6_hamada] {message}", flush=True)


def _bc_setup(mesh: Mesh, idx: dict[str, np.ndarray], L0: sp.csc_matrix, penalty: float) -> tuple[sp.csc_matrix, np.ndarray]:
    """Apply the Hamada boundary-condition recipe used in the MATLAB case.

    Layout:
    - Dirichlet on ``lt`` for ``ruvwT``
    - Dirichlet on ``b`` for ``ruvw``
    - Neumann on ``b`` for ``T``
    - Neumann on ``r`` for ``ruvwT``
    """
    L0, idx_dirichlet_lt = apply_dirichlet(L0, idx, "lt", "ruvwT", mesh.ngp, penalty=penalty)
    L0, idx_dirichlet_b = apply_dirichlet(L0, idx, "b", "ruvw", mesh.ngp, penalty=penalty)
    L0, idx_neumann_b = apply_neumann(L0, mesh, idx, "b", "T", penalty=penalty)
    L0, idx_neumann_r = apply_neumann(L0, mesh, idx, "r", "ruvwT", penalty=penalty)
    bc_all = np.unique(np.concatenate((idx_dirichlet_lt, idx_dirichlet_b, idx_neumann_b, idx_neumann_r)))
    return L0, bc_all


def compute() -> None:
    """Run the full Example-6 pipeline and write ``example_6_hamada.npz``."""
    t0 = time.perf_counter()
    _log("starting compute")

    # ------------------------------------------------------------------
    # 1) Read runtime configuration from environment variables.
    #    Defaults here mirror the reference MATLAB setup.
    # ------------------------------------------------------------------
    case_dir = Path(os.getenv("LINSTAB2D_HAMADA_CASE_DIR", "example_6_hamada"))
    Re = env_float("LINSTAB2D_HAMADA_RE", 395000.0)
    Ma = env_float("LINSTAB2D_HAMADA_MA", 2.0)
    Pr = env_float("LINSTAB2D_HAMADA_PR", 0.747)
    # MATLAB export note:
    # - `case_inputs_used.mat/baseFlow.kappa` is 1.4
    # - `run_metadata.mat/kappa` is 1.31
    # The assembled reference operators/spectrum align with baseFlow.kappa.
    kappa = env_float("LINSTAB2D_HAMADA_KAPPA", 1.4)
    T_0 = env_float("LINSTAB2D_HAMADA_T0", 1540.0)

    stride_ksi = env_int("LINSTAB2D_HAMADA_STRIDE_KSI", 1)
    stride_eta = env_int("LINSTAB2D_HAMADA_STRIDE_ETA", 1)
    fd_order = env_int("LINSTAB2D_HAMADA_FD_ORDER", 4)
    beta = env_float("LINSTAB2D_HAMADA_BETA", 0.0)
    penalty = env_float("LINSTAB2D_HAMADA_BC_PENALTY", 1e8)

    ratio_u = env_float("LINSTAB2D_HAMADA_RATIO_UINF", 1.0)
    ratio_l = env_float("LINSTAB2D_HAMADA_RATIO_L", 0.1)
    St = env_float("LINSTAB2D_HAMADA_ST", 0.1)

    # Target angular frequency used by resolvent and shift-invert EVP.
    omega_target = 2.0 * np.pi * St * ratio_u / ratio_l

    n_modes = env_int("LINSTAB2D_HAMADA_N_MODES", 100)
    n_modes_resolvent = env_int("LINSTAB2D_HAMADA_N_MODES_RESOLVENT", 10)
    run_resolvent = env_bool("LINSTAB2D_HAMADA_RUN_RESOLVENT", False)
    run_stability = env_bool("LINSTAB2D_HAMADA_RUN_STABILITY", True)
    zero_spanwise = env_bool("LINSTAB2D_HAMADA_ZERO_W", False)
    # MATLAB export note:
    # case_inputs_used.mat stores uppercase dMUdT/d2MUdT2 as zeros; GetLinProblem.m
    # consumes those names in getCoeffsLaminar_Cartesian.m. Keep this default True
    # for strict parity with the exported reference matrices/spectrum.
    zero_mu_t_derivs = env_bool("LINSTAB2D_HAMADA_ZERO_MU_T_DERIVS", True)
    # MATLAB stability convention:
    #   L0_matlab = L0 / (-1i)  => omega = eig(L0_matlab) = eig(1j * L0)
    # Keep this as default for parity with exported MATLAB spectrum.
    use_raw_eigs = env_bool("LINSTAB2D_HAMADA_USE_RAW_EIGS", False)

    _log(
        f"config: case_dir={case_dir} fd_order={fd_order} n_modes={n_modes} "
        f"run_stability={run_stability} run_resolvent={run_resolvent} "
        f"raw_eigs={use_raw_eigs} zero_mu_t_derivs={zero_mu_t_derivs}"
    )

    # ------------------------------------------------------------------
    # 2) Load one case snapshot (x, y, rho, u, v, w, T, sponge) and
    #    compute viscosity/transport via Sutherland law.
    # ------------------------------------------------------------------
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
    )

    sutherland_air(baseflow)
    if zero_mu_t_derivs:
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0
    _log(f"loaded case arrays: shape={X.shape}, points={X.size}")

    # ------------------------------------------------------------------
    # 3) Build a computational mesh, then deform it to physical X/Y.
    # ------------------------------------------------------------------
    ny, nx = X.shape
    cmesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=nx, ny=ny, fd_order=fd_order)
    mesh = cmesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    # ------------------------------------------------------------------
    # 4) Assemble LNSE operator and impose boundary conditions.
    # ------------------------------------------------------------------
    _log("assembling LNSE operator")
    L0, idx = build_lnse_operator(mesh, baseflow, kx=beta, model="2D")
    _log(f"operator assembled: shape={L0.shape}, nnz={L0.nnz}")
    L0, bc_indices = _bc_setup(mesh, idx, L0, penalty)
    _log(f"boundary conditions applied: bc_dofs={bc_indices.size}, nnz={L0.nnz}")

    # ------------------------------------------------------------------
    # 5) Build Chu energy norm weights. These weights are required for
    #    weighted analyses (resolvent and temporal formulation).
    # ------------------------------------------------------------------
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = baseflow.RHO.ravel(order="F")[p]
    T_p = baseflow.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp)

    gains = np.array([], dtype=float)
    responses = np.empty((mesh.ngp * 5, 0), dtype=complex)
    forcings = np.empty((mesh.ngp * 5, 0), dtype=complex)

    # ------------------------------------------------------------------
    # 6) Optional resolvent analysis at omega_target.
    # ------------------------------------------------------------------
    if run_resolvent:
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
        )
        _log("resolvent analysis finished")

    omega = np.array([], dtype=complex)
    direct_modes = np.empty((mesh.ngp * 5, 0), dtype=complex)

    # ------------------------------------------------------------------
    # 7) Optional stability analysis.
    #    - raw_eigs: eig(L0) directly (diagnostic-only path, not MATLAB parity)
    #    - temporal: eig(i*L0), which matches MATLAB's L0 = L0 / -1i convention
    # ------------------------------------------------------------------
    if run_stability:
        if use_raw_eigs:
            _log(
                "running raw eigensolve on L0 "
                f"(diagnostic path): target={omega_target:.8g}, modes={n_modes}"
            )
            omega, direct_modes = _eigs_near_target(
                L0.tocsc(),
                n_modes=n_modes,
                target=omega_target,
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
            )
        _log(f"stability analysis finished: converged_modes={omega.size}")

    # ------------------------------------------------------------------
    # 8) Save full run state to one NPZ for downstream parity/plots.
    # ------------------------------------------------------------------
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
    elapsed = time.perf_counter() - t0
    _log(f"results saved to {OUTPUT_NPZ}")
    _log(f"compute finished in {elapsed:.2f}s")


def main() -> None:
    """Cache-aware entrypoint used when launched as ``python ...``."""
    try:
        np.load(OUTPUT_NPZ, allow_pickle = False)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        compute()


if __name__ == "__main__":
    main()
