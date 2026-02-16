#!/usr/bin/env python3
"""Diagnostic: replicate MATLAB example_1_Blasius.m resolvent gains exactly.

MATLAB reference gains: [128.62, 14.54, 7.49]
Parameters: Re=6e5, Ma=0.3, omega=60, nx=300, ny=100, FDorder=4
Domain: x=[0.05, 1.26], y=[-1,1] with Hanifi deformation
Filters: alpha_filter=0 (maximum-strength explicit 10th-order)
B/C: zero out density, temperature, w blocks + Dirichlet DOFs
"""
import time
import numpy as np
import scipy.sparse as sp

from linstab2d.mesh import Mesh
from linstab2d.baseflow import load_blasius, sutherland_air
from linstab2d.operators import build_lnse_operator, compute_chu_norm
from linstab2d.boundary_conditions import apply_dirichlet
from linstab2d.solvers import resolvent_analysis


def main():
    t0 = time.perf_counter()

    # MATLAB parameters (exactly as in example_1_Blasius.m)
    Re = 6e5
    Ma = 0.3
    Pr = 0.7
    kappa = 1.4
    T_0 = 273.15

    nx, ny = 300, 100
    dx = 0.01
    x_min = 5 * dx         # 0.05
    x_max = 1.25 + dx      # 1.26
    y_i = 0.01
    y_max_phys = 0.05

    omega = 100 * (Re / 1e6)  # = 60
    n_eig = 3

    # Create mesh (alpha_filter=0 for real filtering)
    xrange = (x_min, x_max)
    yrange = (-1.0, 1.0)
    mesh = Mesh.create(xrange, yrange, nx, ny, fd_order=4, alpha_filter=0)
    print(f"Mesh: {nx}x{ny} = {mesh.ngp} points")
    print(f"  X: [{mesh.X.min():.4f}, {mesh.X.max():.4f}]")
    print(f"  Y: [{mesh.Y.min():.4f}, {mesh.Y.max():.4f}]")

    # Hanifi 1996 deformation (from MATLAB)
    X = mesh.X
    Y = mesh.Y.copy()
    a = y_i * y_max_phys / (y_max_phys - 2 * y_i)
    b = 1 + 2 * a / y_max_phys
    Y = a * (1 + Y) / (b - Y)
    mesh = mesh.deform(Y_new=Y)
    print(f"  Y deformed (Hanifi): [{mesh.Y.min():.6f}, {mesh.Y.max():.6f}]")

    # Check filters
    filt = mesh.filters
    print(f"  Filter active: {filt is not None}")
    if filt is not None:
        test_v = np.random.default_rng(0).standard_normal(mesh.ngp)
        test_filt = filt.filter(test_v)
        is_identity = np.allclose(test_v, test_filt, atol=1e-14)
        print(f"  Filter is identity: {is_identity}")

    # Base flow
    bf = load_blasius(mesh.X, mesh.Y, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)
    p = mesh.used_ind
    print(f"  Base flow: max|U|={np.max(np.abs(bf.U.ravel(order='F')[p])):.4f}")

    # Sponge (matching MATLAB)
    x = mesh.X
    y = mesh.Y
    xs_trans, xs_offset = 0.02, 0.02
    ys_trans, ys_offset = 0.01, 0.01
    sponge_amp = 5.0
    ind = mesh.used_ind
    xf = x.ravel(order="F")[ind]
    yf = y.ravel(order="F")[ind]
    term_t = np.tanh((yf - np.max(yf) + ys_offset) / ys_trans) + 1
    term_l = np.tanh(-(xf - np.min(xf) - xs_offset) / xs_trans) + 1
    term_r = np.tanh((xf - np.max(xf) + xs_offset) / xs_trans) + 1
    sponge_vals = sponge_amp / 2 * np.maximum(term_t, term_l + term_r)
    # NOTE: ravel(order="F") on C-contiguous array creates a copy;
    # must use flat array + reshape to set values correctly.
    flat = np.full(int(np.prod(mesh.X.shape)), np.nan)
    flat[ind] = sponge_vals
    sponge = flat.reshape(mesh.X.shape, order="F")
    mesh.sponge = sponge
    print(f"  Sponge: max={np.nanmax(sponge_vals):.2f}")

    # Operator assembly
    t1 = time.perf_counter()
    L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
    print(f"  L0: {L0.shape}, nnz/row={L0.nnz/L0.shape[0]:.1f} [{time.perf_counter()-t1:.1f}s]")

    # BCs (MATLAB uses 'lrbt')
    borders = "lrbt"
    vars_ = "uvwT"
    L0, bc_idx = apply_dirichlet(L0, idx, borders, vars_, mesh.ngp)
    print(f"  BCs: {len(bc_idx)} DOFs constrained")

    # Energy norm
    RHO = bf.RHO.ravel(order="F")[p]
    T_field = bf.T.ravel(order="F")[p]
    W_int = mesh.W.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO, T_field, bf.kappa, bf.Ma, mesh.ngp)
    print(f"  Chu norm: min(W)={np.min(W_diag):.2e}, max(W)={np.max(W_diag):.2e}")

    # B/C matrices (matching MATLAB: zero out T, rho, w blocks + Dirichlet DOFs)
    B_vec = np.ones(mesh.ngp * 5)
    B_vec[idx["T_j"]] = 0       # no temperature forcing
    B_vec[idx["rho_j"]] = 0     # no density forcing
    B_vec[idx["w_j"]] = 0       # no w forcing (2D problem)
    B_vec[bc_idx] = 0            # no forcing on Dirichlet BCs
    B_mat = sp.diags(B_vec, format="csc")
    C_mat = B_mat.copy()

    # Resolvent analysis
    print(f"\nResolvent analysis: omega={omega}, n_modes={n_eig}")

    # Test 1: WITH filters (MATLAB default)
    t1 = time.perf_counter()
    gains_f, _, _ = resolvent_analysis(
        L0, omega, n_eig, W_diag,
        B_mat=B_mat, C_mat=C_mat,
        filter_fn=mesh.filters.filter,
        filter_ct_fn=mesh.filters.filter_ct,
    )
    dt = time.perf_counter() - t1
    print(f"\n  WITH filters ({dt:.1f}s):")
    for i, g in enumerate(gains_f):
        print(f"    gain[{i}] = {g:.2f}")
    print(f"  Reference:  [128.62, 14.54, 7.49]")

    # Test 2: WITHOUT filters
    t1 = time.perf_counter()
    gains_nf, _, _ = resolvent_analysis(
        L0, omega, n_eig, W_diag,
        B_mat=B_mat, C_mat=C_mat,
    )
    dt = time.perf_counter() - t1
    print(f"\n  WITHOUT filters ({dt:.1f}s):")
    for i, g in enumerate(gains_nf):
        print(f"    gain[{i}] = {g:.2f}")

    # Test 3: WITHOUT B/C (full resolvent)
    t1 = time.perf_counter()
    gains_full, _, _ = resolvent_analysis(
        L0, omega, n_eig, W_diag,
        filter_fn=mesh.filters.filter,
        filter_ct_fn=mesh.filters.filter_ct,
    )
    dt = time.perf_counter() - t1
    print(f"\n  Full resolvent with filters ({dt:.1f}s):")
    for i, g in enumerate(gains_full):
        print(f"    gain[{i}] = {g:.2f}")

    total = time.perf_counter() - t0
    print(f"\nTotal: {total:.1f}s")


if __name__ == "__main__":
    main()
