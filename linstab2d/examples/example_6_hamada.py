#!/usr/bin/env python3
"""Example 6: Hamada case (2D compressible LNSE, resolvent and stability)."""

from __future__ import annotations

import os
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

from ._helpers import env_bool, env_float, env_int

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")


def _bc_setup(mesh: Mesh, idx: dict[str, np.ndarray], L0: sp.csc_matrix, penalty: float) -> tuple[sp.csc_matrix, np.ndarray]:
    L0, idx_dirichlet_lt = apply_dirichlet(L0, idx, "lt", "ruvwT", mesh.ngp, penalty=penalty)
    L0, idx_dirichlet_b = apply_dirichlet(L0, idx, "b", "ruvw", mesh.ngp, penalty=penalty)
    L0, idx_neumann_b = apply_neumann(L0, mesh, idx, "b", "T", penalty=penalty)
    L0, idx_neumann_r = apply_neumann(L0, mesh, idx, "r", "ruvwT", penalty=penalty)
    bc_all = np.unique(np.concatenate((idx_dirichlet_lt, idx_dirichlet_b, idx_neumann_b, idx_neumann_r)))
    return L0, bc_all


def compute() -> None:
    case_dir = Path(os.getenv("LINSTAB2D_HAMADA_CASE_DIR", "hamada_test"))
    Re = env_float("LINSTAB2D_HAMADA_RE", 395000.0)
    Ma = env_float("LINSTAB2D_HAMADA_MA", 2.0)
    Pr = env_float("LINSTAB2D_HAMADA_PR", 0.747)
    kappa = env_float("LINSTAB2D_HAMADA_KAPPA", 1.31)
    T_0 = env_float("LINSTAB2D_HAMADA_T0", 1540.0)

    stride_ksi = env_int("LINSTAB2D_HAMADA_STRIDE_KSI", 1)
    stride_eta = env_int("LINSTAB2D_HAMADA_STRIDE_ETA", 1)
    fd_order = env_int("LINSTAB2D_HAMADA_FD_ORDER", 4)
    beta = env_float("LINSTAB2D_HAMADA_BETA", 0.0)
    penalty = env_float("LINSTAB2D_HAMADA_BC_PENALTY", 1e8)

    ratio_u = env_float("LINSTAB2D_HAMADA_RATIO_UINF", 1.0)
    ratio_l = env_float("LINSTAB2D_HAMADA_RATIO_L", 0.1)
    St = env_float("LINSTAB2D_HAMADA_ST", 0.1)
    omega_target = 2.0 * np.pi * St * ratio_u / ratio_l

    n_modes = env_int("LINSTAB2D_HAMADA_N_MODES", 100)
    n_modes_resolvent = env_int("LINSTAB2D_HAMADA_N_MODES_RESOLVENT", 10)
    run_resolvent = env_bool("LINSTAB2D_HAMADA_RUN_RESOLVENT", False)
    run_stability = env_bool("LINSTAB2D_HAMADA_RUN_STABILITY", True)
    zero_spanwise = env_bool("LINSTAB2D_HAMADA_ZERO_W", False)
    use_raw_eigs = env_bool("LINSTAB2D_HAMADA_USE_RAW_EIGS", True)

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

    ny, nx = X.shape
    cmesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=nx, ny=ny, fd_order=fd_order)
    mesh = cmesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    L0, idx = build_lnse_operator(mesh, baseflow, kx=beta, model="2D")
    L0, bc_indices = _bc_setup(mesh, idx, L0, penalty)

    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = baseflow.RHO.ravel(order="F")[p]
    T_p = baseflow.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp)

    gains = np.array([], dtype=float)
    responses = np.empty((mesh.ngp * 5, 0), dtype=complex)
    forcings = np.empty((mesh.ngp * 5, 0), dtype=complex)
    if run_resolvent:
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

    omega = np.array([], dtype=complex)
    direct_modes = np.empty((mesh.ngp * 5, 0), dtype=complex)
    if run_stability:
        if use_raw_eigs:
            omega, direct_modes = _eigs_near_target(
                L0.tocsc(),
                n_modes=n_modes,
                target=omega_target,
            )
            order = np.argsort(np.abs(omega - omega_target))
            omega = omega[order]
            direct_modes = direct_modes[:, order]
        else:
            omega, direct_modes = temporal_eigenvalues(
                L0,
                n_modes=n_modes,
                target=omega_target,
                W_diag=W_diag,
            )

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
    print(f"Results saved to {OUTPUT_NPZ}")


def main() -> None:
    try:
        np.load(OUTPUT_NPZ, allow_pickle = False)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        compute()


if __name__ == "__main__":
    main()
