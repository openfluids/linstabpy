#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from linstabpy.baseflow import BaseFlow, sutherland_air
from linstabpy.boundary_conditions.dirichlet import apply_dirichlet
from linstabpy.boundary_conditions.neumann import apply_neumann
from linstabpy.mesh import Mesh
from linstabpy.operators.assembly import build_lnse_operator
from linstabpy.operators.energy_norm import compute_chu_norm
from linstabpy.resolvent.analysis import resolvent_analysis
from linstabpy.solvers.eigenvalue import temporal_eigenvalues

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")


def compute():

    # ---- Physical parameters ----
    Re = 395000
    Ma = 2.0
    Pr = 0.747
    kappa = 1.31
    T_0 = 1540.0
    kx = 0.0  # no spanwise wavenumber for this case

    St = 0.1
    ratio_Uinfty_to_characVelocity = 1.0
    ratio_L_to_Lx = 0.1
    freq = St * ratio_Uinfty_to_characVelocity / ratio_L_to_Lx
    omega = freq * 2 * np.pi

    n_modes = 100
    n_modes_resolvent = 10

    # ---- Grid parameters ----
    x = np.load("x.npy")
    y = np.load("y.npy")
    nx = np.shape(x)[0]
    ny = np.shape(x)[1]
    print(f"Grid: nx={nx}, ny={ny}")

    # ---- Create computational mesh ----
    xrange = (0.0, 1.0)
    yrange = (0.0, 0.5)
    fd_order = 4

    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order)

    # ---- Mesh deformation ----
    mesh = cmesh.deform(X_new=x.T, Y_new=y.T)

    # ---- Load base flow ----
    RHO = np.load("rho.npy")
    U = np.load("u.npy")
    V = np.load("v.npy")
    T = np.load("T.npy")
    W = np.zeros_like(U)

    bf = BaseFlow(Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0, RHO=RHO, U=U, V=V, W=W, T=T)

    sutherland_air(bf)

    # ---- Sponge function ----
    sponge = np.load("sponge.npy")
    sponge_amp = 1.0
    sponge *= sponge_amp
    mesh.sponge = sponge

    # ---- Assemble LNSE operator ----
    t_lnse = time.perf_counter()
    L0, idx = build_lnse_operator(mesh, bf, kx=kx)
    t_lnse = time.perf_counter() - t_lnse

    # ---- Apply Dirichlet BCs ----
    borders = "lt"
    variables = "ruvwT"
    L0, idx_dirichlet_lt = apply_dirichlet(L0, idx, borders, variables, mesh.ngp)

    borders = "b"
    variables = "ruvw"
    L0, idx_dirichlet_b = apply_dirichlet(L0, idx, borders, variables, mesh.ngp)

    # ---- Apply Neumann BCs ----
    borders = "b"
    variables = "T"
    L0, idx_neumann_b = apply_neumann(L0, mesh, idx, borders, variables, mesh.ngp)

    borders = "r"
    variables = "ruvwT"
    L0, idx_neumann_r = apply_neumann(L0, mesh, idx, borders, variables, mesh.ngp)

    # ---- Energy norm ----
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = bf.RHO.ravel(order="F")[p]
    T_p = bf.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO_p, T_p, kappa, Ma, mesh.ngp)

    # ---- Input/output matrices ----
    B_diag = np.ones(mesh.ngp * 5)
    B_diag[idx_dirichlet_lt] = 0.0
    B_diag[idx_dirichlet_b] = 0.0
    B_diag[idx_neumann_b] = 0.0
    B_diag[idx_neumann_r] = 0.0
    B_mat = sp.diags(B_diag).tocsc()
    C_mat = B_mat.copy()

    # ---- Resolvent analysis ----
    t_resolvent = time.perf_counter()
    gains, responses, forcings = resolvent_analysis(
        L0, omega, n_modes_resolvent, W_diag, B_mat=B_mat, C_mat=C_mat
    )
    t_resolvent = time.perf_counter() - t_resolvent
    print(f"build_lnse_operator={t_lnse:.2f}s, resolvent_analysis={t_resolvent:.2f}s")

    print(f"Resolvent gains: {gains}")

    L0 /= -1.0j
    # Direct and adjoint temporal spectra
    eigenvalues, direct_modes = temporal_eigenvalues(L0, n_modes, omega)

    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        omega=omega,
        St=St,
        kx=kx,
        Re=Re,
        Ma=Ma,
        eigenvalues=eigenvalues,
        direct_modes=direct_modes,
        X=mesh.X,
        Y=mesh.Y,
        U=bf.U,
        sponge=mesh.sponge,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def main():
    try:
        data = np.load(OUTPUT_NPZ)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        compute()


if __name__ == "__main__":
    main()
