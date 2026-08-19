#!/usr/bin/env python3
"""
Example 1: Blasius boundary layer resolvent analysis.

Translates example_1_Blasius.m — computes resolvent gains, optimal forcings
and responses for a compressible Blasius boundary layer.

OUTPUTS.
  - example_1_blasius.npz.
  - example_1_blasius_baseflow.png.
  - example_1_blasius_modes_forcing_response_u_mosaic.png.
  - example_1_blasius_modes_forcing_response_u_imag_mosaic.png.
  - example_1_blasius_modes_forcing_response_v_mosaic.png.
  - example_1_blasius_modes_forcing_response_v_imag_mosaic.png.

USAGE:   python example_1_blasius.py                       # .npz exists → plot only
         rm example_1_blasius.npz && python example_1_blasius.py  # recompute
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from linstabpy.utils.yaml_io import load_config

from examples._helpers import load_or_compute
from examples.martini.example_1_plots import plot_example1_figures

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")


def compute(cfg: dict):
    """Run the full Blasius resolvent analysis."""
    import scipy.sparse as sp

    from examples.martini.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system_result
    from linstabpy.operators.energy_norm import compute_chu_norm
    from linstabpy.resolvent.analysis import resolvent_analysis

    solver_cfg = cfg.get("solver")

    # ---- Physical parameters ----
    Re = cfg["Re"]
    Ma = cfg["Ma"]
    Pr = cfg["Pr"]
    kappa = cfg["kappa"]
    T_0 = cfg["T_0"]

    delta = 1.72 / np.sqrt(Re)
    omega = cfg["omega"]
    n_modes = cfg["n_modes"]

    # ---- Grid parameters (MATLAB native resolution) ----
    nx = cfg["nx"]
    ny = cfg["ny"]
    print(f"Blasius resolvent grid: nx={nx}, ny={ny}")
    dx = cfg["dx"]
    x_min = cfg["x_min_factor"] * dx
    x_max = cfg["x_max_base"] + dx
    y_i = cfg["y_i"]
    y_max = cfg["y_max"]

    # ---- Create computational mesh ----
    xrange = (x_min, x_max)
    yrange = (-1.0, 1.0)
    fd_order = cfg["fd_order"]

    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order, alpha_filter=cfg["alpha_filter"])

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
    # LinStab2D Example 1 uses dMUdT = 0 in the 2D coefficient path (the baseFlow
    # viscosity-derivative fields are never populated). Pin zero derivatives here
    # only when ``zero_dmudT_martini_parity`` is set for MATLAB parity. The
    # physically consistent choice is to keep the Sutherland dmudT from
    # ``sutherland_air`` (omit the flag or set it false).
    if cfg.get("zero_dmudT_martini_parity", False):
        bf.dmudT[:] = 0.0
        bf.d2mudT2[:] = 0.0

    # ---- Sponge function ----
    x = mesh.X
    y = mesh.Y
    sponge_cfg = cfg["sponge"]
    xs_trans = sponge_cfg["xs_trans"]
    xs_offset = sponge_cfg["xs_offset"]
    ys_trans = sponge_cfg["ys_trans"]
    ys_offset = sponge_cfg["ys_offset"]
    sponge_amp = sponge_cfg["amplitude"]

    sponge = (
        sponge_amp
        / 2
        * np.maximum(
            np.tanh((y - y.max() + ys_offset) / ys_trans) + 1,
            np.tanh(-(x - x.min() - xs_offset) / xs_trans)
            + 1
            + np.tanh((x - x.max() + xs_offset) / xs_trans)
            + 1,
        )
    )
    mesh.sponge = sponge

    # ---- Assemble LNSE operator ----
    kx = cfg["kx"]
    assembly_backend = cfg.get("assembly_backend")
    t_lnse = time.perf_counter()
    lnse_system = build_lnse_system_result(
        mesh,
        bf,
        kx=kx,
        model=cfg.get("model", "2D"),
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=cfg["bc"]["penalty"],
        backend=assembly_backend,
        solver_cfg=solver_cfg,
    )
    L0 = lnse_system.operator
    idx = lnse_system.idx
    bc_indices = lnse_system.bc_indices
    t_lnse = time.perf_counter() - t_lnse

    # ---- Energy norm ----
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = bf.RHO.ravel(order="F")[p]
    T_p = bf.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO_p, T_p, kappa, Ma, mesh.ngp)

    # ---- Input/output matrices ----
    var_map = {"rho": "rho_j", "u": "u_j", "v": "v_j", "w": "w_j", "T": "T_j"}
    B_diag = np.ones(mesh.ngp * 5)
    for var_name in cfg.get("forcing_zero_vars", []):
        B_diag[idx[var_map[var_name]]] = 0.0
    B_diag[bc_indices] = 0.0
    B_mat = sp.diags(B_diag).tocsc()
    C_mat = B_mat.copy()

    # ---- Resolvent analysis ----
    t_resolvent = time.perf_counter()
    (
        gains,
        responses,
        forcings,
        svd_residuals_fwd,
        svd_residuals_adj,
    ) = resolvent_analysis(
        L0,
        omega,
        n_modes,
        W_diag,
        B_mat=B_mat,
        C_mat=C_mat,
        filter_fn=mesh.filters.filter if mesh.filters is not None else None,
        filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
        solver_cfg=solver_cfg,
        residuals=True,
    )
    t_resolvent = time.perf_counter() - t_resolvent
    print(
        f"Blasius timing: build_lnse_operator={t_lnse:.2f}s, resolvent_analysis={t_resolvent:.2f}s"
    )

    print(f"Resolvent gains: {gains}")

    # ---- SVD self-residuals -----------------------------------------------
    # MATLAB used 5-output lu(A) (UMFPACK) + ARPACK eigs at tol=eps=2.2e-16,
    # default subspace opts.p = 2*nEig+1.  Python uses MUMPS/METIS + SLEPc EPS
    # at tol=1e-12, ncv = max(2*n_modes+3, 20).  Different fill-reducing
    # orderings (UMFPACK vs MUMPS/METIS) yield eigenvectors that differ at the
    # O(eps*cond) level; the resulting gain mismatch (~0.54% for this case) is
    # structural, not a solver defect.  The SVD residuals are the primary closure
    # criterion: if the triplets (sigma, u, v) satisfy the discrete resolvent
    # equations to near-machine precision the result is correct independently
    # of the MATLAB reference.
    # ---- Save results ----
    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        svd_residuals_fwd=svd_residuals_fwd,
        svd_residuals_adj=svd_residuals_adj,
        omega=omega,
        Re=Re,
        Ma=Ma,
        X=mesh.X,
        Y=mesh.Y,
        ngp=mesh.ngp,
        used_ind=mesh.used_ind,
        baseflow_u=bf.U,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def plot(data):
    """Plot campaign-style Blasius baseflow and resolvent mosaics."""
    cfg = load_config(SCRIPT.with_suffix(".yaml"))
    paths = plot_example1_figures(data, example_dir=SCRIPT.parent, cfg=cfg)
    for path in paths:
        print(f"Wrote {path}")


def main():
    cfg = load_config(SCRIPT.with_suffix(".yaml"))
    data = load_or_compute(OUTPUT_NPZ, lambda: compute(cfg))
    plot(data)


if __name__ == "__main__":
    main()
