#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from linstab2d.mesh import Mesh
from linstab2d.baseflow import BaseFlow, sutherland_air
from linstab2d.operators.coefficients import compute_cartesian_coefficients
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.boundary_conditions.neumann import apply_neumann
from linstab2d.solvers.resolvent import resolvent_analysis
from linstab2d.solvers.eigenvalue import adjoint_eigenvalues, temporal_eigenvalues
from linstab2d.utils.plotting import plot_flow

from linstab2d.examples._helpers import to_grid

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

    St = 0.1
    ratio_Uinfty_to_characVelocity = 1.0
    ratio_L_to_Lx = 0.1
    freq = St*ratio_Uinfty_to_characVelocity/ratio_L_to_Lx
    omega =  freq*2*np.pi

    n_modes = 100
    n_plot = n_modes

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

    print(np.shape(cmesh.X))

    # ---- Mesh deformation ----
    mesh = cmesh.deform(X_new=x.T,Y_new=y.T)

    # ---- Load base flow ----
    RHO = np.load("rho.npy")
    U = np.load("u.npy")
    V = np.load("v.npy")
    T = np.load("T.npy")
    W = np.zeros_like(U)

    bf = BaseFlow(
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        RHO=RHO,
        U=U,
        V=V,
        W=W,
        T=T)
    
    sutherland_air(bf)

    # ---- Sponge function ----
    sponge = np.load("sponge.npy")
    sponge_amp = 1.0
    sponge *= sponge_amp
    mesh.sponge = sponge

    # ---- Assemble LNSE operator ----
    kx = 0.0  # no streamwise wavenumber for this case
    t_lnse = time.perf_counter()
    L0, idx = build_lnse_operator(mesh, bf, kx=kx)
    t_lnse = time.perf_counter() - t_lnse

    # ---- Apply Dirichlet BCs ----
    borders = "lt"
    variables = "ruvwT"
    L0, bc_indices = apply_dirichlet(L0, idx, borders, variables, mesh.ngp)

    borders = "b"
    variables = "ruvw"
    L0, bc_indices = apply_dirichlet(L0, idx, borders, variables, mesh.ngp)

    # ---- Apply Neumann BCs ----
    borders = "b"
    variables = "T"
    L0, bc_indices = apply_neumann(L0, mesh, idx, borders, variables, mesh.ngp)

    borders = "r"
    variables = "ruvwT"
    L0, bc_indices = apply_neumann(L0, mesh, idx, borders, variables, mesh.ngp)

    # ---- Energy norm ----
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = bf.RHO.ravel(order="F")[p]
    T_p = bf.T.ravel(order="F")[p]
    # W_diag, invW_diag = compute_chu_norm(W_int, RHO_p, T_p, kappa, Ma, mesh.ngp)

    L0 /= -1.0j
    # Direct and adjoint temporal spectra
    omega, direct_modes = temporal_eigenvalues(
        L0,
        n_modes,
        omega
    )


    np.savez_compressed(
        OUTPUT_NPZ,
        omega=omega,
        direct_modes=direct_modes,
        X=mesh.X,
        Y=mesh.Y,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
        used_ind=mesh.used_ind,
        bc_indices=bc_indices,
        n_plot=n_plot,
    )
    print(f"Results saved to {OUTPUT_NPZ}")

def plot(data):
    """Plot temporal spectra plus direct/adjoint leading modes."""
    import matplotlib.pyplot as plt

    output_spectrum = OUTPUT_PNG

    omega = data["omega"]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.real(omega), np.imag(omega), "ob", ms=7, mew=1, label="Direct")
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.set_title("Temporal spectra")
    ax.grid(True)
    ax.legend()
    fig.savefig(output_spectrum, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(
        "Figures saved to:\n"
        f"- Spectrum: {output_spectrum}\n"
    )

def main():
    try:
        data = np.load(OUTPUT_NPZ)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        compute()
        data = np.load(OUTPUT_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
