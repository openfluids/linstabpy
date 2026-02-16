#!/usr/bin/env python3
"""Check the temporal spectrum of L0 near iω = 60i."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linstab2d.mesh import Mesh
from linstab2d.baseflow import load_blasius, sutherland_air
from linstab2d.operators import build_lnse_operator, compute_chu_norm
from linstab2d.boundary_conditions import apply_dirichlet


def main():
    # Same setup as MATLAB example_1
    Re, Ma, Pr, kappa, T_0 = 6e5, 0.3, 0.7, 1.4, 273.15
    nx, ny = 300, 100
    dx = 0.01
    x_min, x_max = 5 * dx, 1.25 + dx
    y_i, y_max_phys = 0.01, 0.05

    mesh = Mesh.create((x_min, x_max), (-1, 1), nx, ny, fd_order=4, alpha_filter=0)
    Y = mesh.Y.copy()
    a = y_i * y_max_phys / (y_max_phys - 2 * y_i)
    b = 1 + 2 * a / y_max_phys
    Y = a * (1 + Y) / (b - Y)
    mesh = mesh.deform(Y_new=Y)

    bf = load_blasius(mesh.X, mesh.Y, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    # Sponge
    ind = mesh.used_ind
    xf = mesh.X.ravel(order="F")[ind]
    yf = mesh.Y.ravel(order="F")[ind]
    flat = np.full(int(np.prod(mesh.X.shape)), np.nan)
    flat[ind] = 5.0 / 2 * np.maximum(
        np.tanh((yf - np.max(yf) + 0.01) / 0.01) + 1,
        np.tanh(-(xf - np.min(xf) - 0.02) / 0.02) + 1
        + np.tanh((xf - np.max(xf) + 0.02) / 0.02) + 1,
    )
    mesh.sponge = flat.reshape(mesh.X.shape, order="F")

    L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
    L0, bc_idx = apply_dirichlet(L0, idx, "lrbt", "uvwT", mesh.ngp)

    p = mesh.used_ind
    W_diag, _ = compute_chu_norm(
        mesh.W.ravel(order="F")[p],
        bf.RHO.ravel(order="F")[p],
        bf.T.ravel(order="F")[p],
        bf.kappa, bf.Ma, mesh.ngp,
    )

    omega = 60.0
    print(f"Resolvent frequency: iω = {omega}i")
    print(f"L0 shape: {L0.shape}")

    # Compute eigenvalues near several targets
    targets = [60j, -60j, 0.0, 60.0, -60.0, 30j, 100j]
    for target in targets:
        try:
            evals, _ = spla.eigs(L0, k=5, sigma=target, tol=1e-8)
            evals_sorted = evals[np.argsort(-evals.imag)]
            print(f"\n  Near σ = {target}:")
            for ev in evals_sorted:
                dist_to_iomega = abs(ev - 1j * omega)
                print(f"    λ = {ev.real:+.4f} {ev.imag:+.4f}i  "
                      f"|λ - iω| = {dist_to_iomega:.4f}")
        except Exception as e:
            print(f"\n  Near σ = {target}: FAILED ({e})")

    # Also compute the smallest singular value of (L0 - iω I)
    L_sigma = L0 - 1j * omega * sp.eye(L0.shape[0], format="csc")
    print(f"\n  Computing smallest singular value of (L0 - iωI)...")
    try:
        sv = spla.svds(L_sigma, k=1, which="SM", return_singular_vectors=False)
        print(f"  σ_min(L0 - iωI) = {sv[0]:.6e}")
        print(f"  Predicted gain ≈ 1/σ_min = {1.0/sv[0]:.2f}")
    except Exception as e:
        print(f"  svds failed: {e}")


if __name__ == "__main__":
    main()
