#!/usr/bin/env python3
"""
Example 4: Cavity global/adjoint stability analysis.

Translates example_4_Cavity.m using a masked cavity geometry, deformed mesh,
and direct + adjoint temporal eigenvalue problems.
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linstab2d.baseflow import load_cavity, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.eigenvalue import adjoint_eigenvalues, temporal_eigenvalues
from linstab2d.utils.plotting import plot_flow

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")


def _to_grid(values: np.ndarray, used_indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    flat = np.full(int(np.prod(shape)), np.nan, dtype=complex)
    flat[np.asarray(used_indices)] = np.asarray(values)
    return flat.reshape(shape, order="F")


def _to_bool(v: str | bool | None) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _to_int(v: str | None, default: int) -> int:
    if v is None:
        return default
    try:
        value = int(v)
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _to_complex(v: str | None, default: complex) -> complex:
    if v is None:
        return default
    try:
        return complex(v)
    except (TypeError, ValueError):
        print(f"Invalid complex value {v!r}; using {default}.")
        return default


def _normalize_in_weighted_inner(vectors: np.ndarray, W_diag: np.ndarray | None) -> np.ndarray:
    if W_diag is None:
        return vectors

    W = np.asarray(W_diag)
    if W.ndim != 1:
        raise ValueError("W_diag must be one-dimensional")
    if vectors.shape[0] != W.size:
        raise ValueError("W_diag length must match eigenvector length")
    if np.any(W == 0):
        raise ValueError("W_diag contains zero entries; cannot normalise")

    normed = np.array(vectors, dtype=complex, copy=True)
    for i in range(vectors.shape[1]):
        v = vectors[:, i]
        norm_sq = np.real(np.vdot(v, W * v))
        if norm_sq <= 0.0:
            continue
        normed[:, i] = v / np.sqrt(norm_sq)
    return normed


def _sort_nearest(values: np.ndarray, vectors: np.ndarray, target: complex) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.abs(values - target))
    return values[order], vectors[:, order]


def _eigs_no_shift(
    operator: sp.spmatrix,
    n_modes: int,
    target: complex,
) -> tuple[np.ndarray, np.ndarray]:
    op = sp.csc_matrix(operator)
    ndof = op.shape[0]
    k = min(max(1, n_modes), max(1, ndof - 2))
    if ndof <= 2500:
        values, vectors = np.linalg.eig(np.asarray(op.toarray()))
        return _sort_nearest(values, vectors, target)
    tol = 1e-8
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(op.shape[0]) + 1j * rng.standard_normal(op.shape[0])
    attempts = []
    for which in ("LR", "SR", "LM"):
        for ncv in (max(4 * k + 1, 20), max(2 * k + 2, 10)):
            attempts.append((which, ncv))
            try:
                values, vectors = spla.eigs(
                    op,
                    k=k,
                    which=which,
                    tol=tol,
                    ncv=ncv,
                    v0=v0,
                )
                return _sort_nearest(values, vectors, target)
            except Exception:
                continue
    raise RuntimeError(f"Non-shifted ARPACK failed after attempts: {attempts}")


def _temporal_eigen_with_fallback(
    L0: sp.spmatrix,
    n_modes: int,
    omega_target: complex,
    W_diag: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        values, vectors = temporal_eigenvalues(L0, n_modes, omega_target, W_diag=W_diag)
        return values, vectors
    except RuntimeError as exc:
        print(f"Shift-invert temporal solve failed ({exc}); using non-shifted ARPACK fallback.")
        op = 1j * sp.csc_matrix(L0)
        values, vectors = _eigs_no_shift(op, n_modes, omega_target)
        values, vectors = _sort_nearest(values, vectors, omega_target)
        return values[:n_modes], _normalize_in_weighted_inner(vectors[:, :n_modes], W_diag)


def _adjoint_eigen_with_fallback(
    L0: sp.spmatrix,
    n_modes: int,
    omega_target: complex,
    W_diag: np.ndarray,
    invW_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        values, vectors = adjoint_eigenvalues(L0, n_modes, omega_target, W_diag, invW_diag)
        return values, vectors
    except RuntimeError as exc:
        print(f"Shift-invert adjoint solve failed ({exc}); using non-shifted ARPACK fallback.")
        L0_csc = sp.csc_matrix(L0)
        W = np.asarray(W_diag)
        invW = np.asarray(invW_diag)
        adjoint_op = sp.diags(invW, format="csc") @ (L0_csc.conj().T @ sp.diags(W, format="csc"))
        values, vectors = _eigs_no_shift(adjoint_op, n_modes, omega_target)
        values, vectors = _sort_nearest(values, vectors, omega_target)
        return values[:n_modes], _normalize_in_weighted_inner(vectors[:, :n_modes], W_diag)


def compute():
    """Run cavity direct/adjoint temporal eigenproblems."""
    omega_target = _to_complex(os.getenv("LINSTAB2D_CAVITY_OMEGA_TARGET"), 1.5 + 1.0j)
    reduce_mode = (not _to_bool(os.getenv("LINSTAB2D_ALLOW_FULL_CAVITY"))) or _to_bool(os.getenv("LINSTAB2D_REDUCED_CAVITY"))
    if reduce_mode:
        print("Running in reduced memory-safe mode for Cavity (set LINSTAB2D_ALLOW_FULL_CAVITY=1 to run full-size).")

    n_modes = _to_int(
        os.getenv("LINSTAB2D_CAVITY_N_MODES_REDUCED" if reduce_mode else "LINSTAB2D_CAVITY_N_MODES"),
        4 if reduce_mode else 50,
    )
    n_plot = min(3, n_modes)
    beta = 0.0

    Nx = _to_int(os.getenv("LINSTAB2D_CAVITY_NX_REDUCED"), 31 if reduce_mode else 241)
    Ny = _to_int(os.getenv("LINSTAB2D_CAVITY_NY_REDUCED"), 11 if reduce_mode else 101)
    FDorder = 4

    x_cavity_start = 0.34029
    L_cavity = 1.9995
    xrange = (
        x_cavity_start - L_cavity,
        x_cavity_start + 2 * L_cavity,
    )
    yrange = (-1.0, 1.0)

    cmesh = Mesh.create(
        xrange,
        yrange,
        Nx,
        Ny,
        FDorder,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )
    # MATLAB convention: mask=True means "remove this point".
    # Python Mesh.mask() convention: mask=True means "keep this point".
    # Negate to convert.
    remove = (cmesh.Y < 0.0) & ((cmesh.X < 0.34) | (cmesh.X > 2.34))
    mesh = cmesh.mask(~remove)

    # --- y-deformation (mesh concentration + top stretch) ------------------
    Y = mesh.Y.copy()
    a = 0.5
    Y = (Y**3 + a * Y) / (1 + a)
    strech_n = 3
    p = Y > 0.0
    Y[p] = Y[p] + 2.0 * ((Y[p] - 0.0) / (np.max(Y[p]) - 0.0)) ** strech_n
    mesh = mesh.deform(Y_new=Y)

    # --- x-stretch on the right ------------------------------------------
    X = mesh.X.copy()
    p = X > 2.8
    X[p] = X[p] + 1.0 * ((X[p] - 2.8) / (np.max(X[p]) - 2.8)) ** strech_n
    mesh = mesh.deform(X_new=X)

    baseflow = load_cavity(mesh)
    sutherland_air(baseflow)

    xs_trans = 0.5
    xs_offset = 0.5
    ys_trans = 0.5
    ys_offset = 1.0
    sponge_amp = 5.0
    sponge = np.zeros(mesh.X.shape, dtype=float)
    ind = mesh.used_ind
    x = mesh.X.ravel(order="F")[ind]
    y = mesh.Y.ravel(order="F")[ind]
    term_x = np.tanh(-(x - np.min(x) - xs_offset) / xs_trans) + 1
    term_y = np.tanh((x - np.max(x) + xs_offset) / xs_trans) + 1
    term_t = np.tanh((y - np.max(y) + ys_offset) / ys_trans) + 1
    sponge_vals = sponge_amp / 2 * np.maximum(term_t, term_x + term_y)
    sponge.ravel(order="F")[ind] = sponge_vals
    mesh.sponge = sponge

    L0, idx = build_lnse_operator(mesh, baseflow, model="2D", kx=beta)
    borders = "lbtrm"
    vars_ = "uvwT"
    L0, bc_indices = apply_dirichlet(L0, idx, borders, vars_, mesh.ngp)

    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO = baseflow.RHO.ravel(order="F")[p]
    T = baseflow.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(
        W_int,
        RHO,
        T,
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
        model="2D",
    )

    # Direct and adjoint temporal spectra
    omega, direct_modes = _temporal_eigen_with_fallback(
        L0,
        n_modes,
        omega_target,
        W_diag=W_diag,
    )
    omega_adj_target = 1j * np.conj(omega_target)
    omega_adj, adjoint_modes = _adjoint_eigen_with_fallback(
        L0,
        n_modes,
        omega_adj_target,
        W_diag,
        invW_diag,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        omega=omega,
        omega_adj=omega_adj,
        direct_modes=direct_modes,
        adjoint_modes=adjoint_modes,
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

    output_direct = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_direct.png")
    output_adjoint = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_adjoint.png")
    output_spectrum = OUTPUT_PNG

    omega = data["omega"]
    omega_adj = data["omega_adj"]
    direct_modes = data["direct_modes"]
    adjoint_modes = data["adjoint_modes"]
    X = data["X"]
    Y = data["Y"]
    used_ind = data["used_ind"]
    idx_rho = data["idx_rho_j"]
    idx_u = data["idx_u_j"]
    idx_v = data["idx_v_j"]
    idx_w = data["idx_w_j"]
    idx_T = data["idx_T_j"]
    n_plot = int(data["n_plot"])

    direct_order = np.argsort(np.imag(omega))[::-1][:n_plot]

    # Convert adjoint eigenvalues to direct convention for plotting/sorting.
    # Python adjoint_eigenvalues returns eig(W⁻¹ L0ᴴ W) = i·conj(omega_d).
    # MATLAB returns eig(W⁻¹ (iL0)ᴴ W) = conj(omega_d).
    # Recover direct eigenvalues: omega_d_from_adj = conj(-i·omega_adj).
    omega_d_from_adj = np.conj(-1j * omega_adj)
    adj_order = np.argsort(np.imag(omega_d_from_adj))[::-1][:n_plot]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.real(omega), np.imag(omega), "ob", ms=7, mew=1, label="Direct")
    ax.plot(
        np.real(omega_d_from_adj),
        np.imag(omega_d_from_adj),
        "x",
        color="orange",
        ms=14,
        mew=2.2,
        label="Adjoint",
        zorder=4,
    )
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.set_title("Temporal spectra")
    ax.grid(True)
    ax.legend()

    for i, j in enumerate(direct_order):
        mode = direct_modes[:, int(j)]
        fig_mode, axes = plt.subplots(3, 2, figsize=(12, 10))
        fields = [
            (idx_rho, r"$\rho$"),
            (idx_u, r"$u$"),
            (idx_v, r"$v$"),
            (idx_w, r"$w$"),
            (idx_T, r"$T$"),
        ]
        for k, (idx_var, title_) in enumerate(fields):
            ax_k = axes.flat[k]
            field = _to_grid(mode[idx_var], used_ind, X.shape)
            plot_flow(X, Y, np.real(field), ax=ax_k, levels=51, cmap="RdBu_r")
            ax_k.set_title(f"Direct {title_}, ω={omega[j]:.3f}")
        axes.flat[-1].set_visible(False)

        fig_mode.suptitle(f"Direct mode {int(j) + 1}")
        fig_mode.tight_layout()
        if i == 0:
            fig_mode.savefig(output_direct, dpi=600, bbox_inches="tight")
            plt.close(fig_mode)
            print(f"Direct mode figure saved to {output_direct}")

    for i, j in enumerate(adj_order):
        mode = adjoint_modes[:, int(j)]
        fig_mode, axes = plt.subplots(3, 2, figsize=(12, 10))
        fields = [
            (idx_rho, r"$\rho$"),
            (idx_u, r"$u$"),
            (idx_v, r"$v$"),
            (idx_w, r"$w$"),
            (idx_T, r"$T$"),
        ]
        for k, (idx_var, title_) in enumerate(fields):
            ax_k = axes.flat[k]
            field = _to_grid(mode[idx_var], used_ind, X.shape)
            plot_flow(X, Y, np.real(field), ax=ax_k, levels=51, cmap="RdBu_r")
            ax_k.set_title(f"Adjoint ω*={omega_d_from_adj[j]:.3f} : {title_}")
        axes.flat[-1].set_visible(False)

        fig_mode.suptitle(f"Adjoint mode {int(j) + 1}")
        fig_mode.tight_layout()
        if i == 0:
            fig_mode.savefig(output_adjoint, dpi=600, bbox_inches="tight")
            plt.close(fig_mode)
            print(f"Adjoint mode figure saved to {output_adjoint}")

    plt.close(fig)
    fig.savefig(output_spectrum, dpi=600, bbox_inches="tight")
    print(
        "Figures saved to:\n"
        f"- Spectrum: {output_spectrum}\n"
        f"- Direct mode: {output_direct}\n"
        f"- Adjoint mode: {output_adjoint}"
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
