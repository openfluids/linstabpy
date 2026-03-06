#!/usr/bin/env python3
"""
Example 1: Blasius boundary layer resolvent analysis.

Translates example_1_Blasius.m — computes resolvent gains, optimal forcings
and responses for a compressible Blasius boundary layer.

OUTPUTS: example_1_blasius.npz, example_1_blasius.png
USAGE:   python example_1_blasius.py                       # .npz exists → plot only
         rm example_1_blasius.npz && python example_1_blasius.py  # recompute
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from linstab2d.mesh import Mesh
from linstab2d.baseflow import BaseFlow, sutherland_air, load_blasius
from linstab2d.operators.coefficients import compute_cartesian_coefficients
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.solvers.resolvent import resolvent_analysis

from ._helpers import env_float, env_int

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")


def compute():
    """Run the full Blasius resolvent analysis."""
    # ---- Physical parameters ----
    Re = 6e5
    Ma = 0.3
    Pr = 0.7
    kappa = 1.4
    T_0 = 273.15

    delta = 1.72 / np.sqrt(Re)
    omega = 100 * (Re / 1e6)
    n_modes = 3

    # ---- Grid parameters (MATLAB native resolution) ----
    nx = env_int("LINSTAB2D_BLASIUS_NX", 300)
    ny = env_int("LINSTAB2D_BLASIUS_NY", 100)
    print(f"Blasius resolvent grid: nx={nx}, ny={ny}")
    dx = 0.01
    x_min = 5 * dx
    x_max = 1.25 + dx
    y_min = 0.0
    y_i = 0.01
    y_max = 0.05

    # ---- Create computational mesh ----
    xrange = (x_min, x_max)
    yrange = (-1.0, 1.0)
    fd_order = 4

    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order, alpha_filter=0.0)

    # ---- Mesh deformation (Hanifi 1996 mapping) ----
    X = cmesh.X.copy()
    Y = cmesh.Y.copy()
    a = y_i * y_max / (y_max - 2 * y_i)
    b = 1 + 2 * a / y_max
    Y_phys = a * (1 + Y) / (b - Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    # ---- Load Blasius base flow ----
    bf = load_blasius(mesh.X, Y_phys, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    # ---- Sponge function ----
    x = mesh.X
    y = mesh.Y
    xs_trans = 0.02
    xs_offset = 0.02
    ys_trans = 0.01
    ys_offset = 0.01
    sponge_amp = 5.0

    sponge = sponge_amp / 2 * np.maximum(
        np.tanh((y - y.max() + ys_offset) / ys_trans) + 1,
        np.tanh(-(x - x.min() - xs_offset) / xs_trans) + 1
        + np.tanh((x - x.max() + xs_offset) / xs_trans) + 1,
    )
    mesh.sponge = sponge

    # ---- Assemble LNSE operator ----
    kx = 0.0  # no streamwise wavenumber for this case
    t_lnse = time.perf_counter()
    L0, idx = build_lnse_operator(mesh, bf, kx=kx)
    t_lnse = time.perf_counter() - t_lnse

    # ---- Apply Dirichlet BCs ----
    bc_penalty = env_float("LINSTAB2D_BLASIUS_BC_PENALTY", 1e4)
    borders = "lrbt"
    variables = "uvwT"
    L0, bc_indices = apply_dirichlet(L0, idx, borders, variables, mesh.ngp, penalty=bc_penalty)
    print(f"Dirichlet BC penalty: {bc_penalty:.3e}")

    # ---- Energy norm ----
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = bf.RHO.ravel(order="F")[p]
    T_p = bf.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO_p, T_p, kappa, Ma, mesh.ngp)

    # ---- Input/output matrices ----
    # Force only u, v (no rho, w, T forcing)
    B_diag = np.ones(mesh.ngp * 5)
    B_diag[idx["T_j"]] = 0.0
    B_diag[idx["rho_j"]] = 0.0
    B_diag[idx["w_j"]] = 0.0
    B_diag[bc_indices] = 0.0
    B_mat = sp.diags(B_diag).tocsc()
    C_mat = B_mat.copy()

    # ---- Resolvent analysis ----
    t_resolvent = time.perf_counter()
    gains, responses, forcings = resolvent_analysis(
        L0,
        omega,
        n_modes,
        W_diag,
        B_mat=B_mat,
        C_mat=C_mat,
        filter_fn=mesh.filters.filter if mesh.filters is not None else None,
        filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
    )
    t_resolvent = time.perf_counter() - t_resolvent
    print(
        "Blasius timing: "
        f"build_lnse_operator={t_lnse:.2f}s, resolvent_analysis={t_resolvent:.2f}s"
    )

    print(f"Resolvent gains: {gains}")

    # ---- Save results ----
    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        omega=omega,
        Re=Re,
        Ma=Ma,
        X=mesh.X,
        Y=mesh.Y,
        ngp=mesh.ngp,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def plot(data):
    """Plot resolvent gains and mode shapes."""
    import matplotlib.pyplot as plt

    gains = data["gains"]
    responses = data["responses"]
    forcings = data["forcings"]
    X = data["X"]
    Y = data["Y"]
    ngp = int(data["ngp"])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i in range(min(3, len(gains))):
        # u-component forcing and response (variable index 1)
        f_u = forcings[1 * ngp : 2 * ngp, i]
        r_u = responses[1 * ngp : 2 * ngp, i]

        ny, nx = X.shape
        f_u_2d = f_u.reshape((ny, nx), order="F")
        r_u_2d = r_u.reshape((ny, nx), order="F")

        ax_f = axes[i, 0]
        cf = ax_f.contourf(X, Y, np.real(f_u_2d), levels=51, cmap="RdBu_r")
        ax_f.set_title(f"Forcing $f_u^{{({i+1})}}$, $\\sigma={gains[i]:.2f}$")
        ax_f.set_ylabel("y")
        ax_f.set_ylim(0, 0.01)
        fig.colorbar(cf, ax=ax_f)

        ax_r = axes[i, 1]
        cr = ax_r.contourf(X, Y, np.real(r_u_2d), levels=51, cmap="RdBu_r")
        ax_r.set_title(f"Response $u^{{({i+1})}}$")
        ax_r.set_ylabel("y")
        ax_r.set_ylim(0, 0.01)
        fig.colorbar(cr, ax=ax_r)

    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("x")

    fig.suptitle("Blasius Resolvent Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_PNG}")


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
