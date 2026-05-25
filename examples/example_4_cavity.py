#!/usr/bin/env python3
"""
Example 4: Cavity global/adjoint stability analysis.

Translates example_4_Cavity.m using a masked cavity geometry, deformed mesh,
and direct + adjoint temporal eigenvalue problems.

OUTPUTS:
  - example_4_cavity.npz
  - example_4_cavity_baseflow.png
  - example_4_cavity.png
  - example_4_cavity_direct.png
  - example_4_cavity_adjoint.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from linstabpy.utils.jsonc import load_jsonc, parse_complex

from ._helpers import (
    apply_figure_layout,
    format_complex,
    format_scientific_ticks,
    label_axes,
    load_or_compute,
    load_parity_npz,
    mesh_view,
    parity_npz_get,
    plot_scalar_panel,
    plot_signed_panel,
    save_figure,
    set_suptitle,
    to_grid,
)

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
OUTPUT_BASEFLOW = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_baseflow.png")
MATLAB_DIRECT_STEM = "example_4_Cavity_results_direct"
MATLAB_ADJOINT_STEM = "example_4_Cavity_results_adjoint"


def _load_matlab_spectra() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load extracted MATLAB direct and adjoint spectra when available."""
    try:
        direct_bundle = load_parity_npz(MATLAB_DIRECT_STEM)
        adjoint_bundle = load_parity_npz(MATLAB_ADJOINT_STEM)
    except FileNotFoundError:
        return None, None

    omega_direct = np.asarray(parity_npz_get(direct_bundle, "results_Cavity_direct/omega"), dtype=complex).reshape(-1)
    omega_adjoint = np.asarray(
        parity_npz_get(adjoint_bundle, "results_Cavity_adjoint/omegaAdj"),
        dtype=complex,
    ).reshape(-1)
    return omega_direct, np.conj(omega_adjoint)


def compute(cfg: dict):
    """Run cavity direct/adjoint temporal eigenproblems."""
    from linstabpy.baseflow import load_cavity, sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system
    from linstabpy.operators.energy_norm import compute_chu_norm
    from linstabpy.solvers.eigenvalue import adjoint_eigenvalues, temporal_eigenvalues

    solver_cfg = cfg.get("solver")
    if solver_cfg is None:
        solver_cfg = {}
    assembly_backend = cfg.get("assembly_backend", "scipy")
    if assembly_backend != "scipy":
        raise NotImplementedError(
            "example_4_cavity currently supports only assembly_backend='scipy' because the adjoint workflow remains SciPy-based"
        )
    bc_penalty = cfg["bc"].get("penalty", 1e4)

    omega_target = parse_complex(cfg["omega_target"])
    n_modes = cfg["n_modes"]
    n_plot = min(3, n_modes)
    beta = cfg["beta"]
    arnoldi_ncv = cfg["ncv"]

    nx = cfg["nx"]
    ny = cfg["ny"]
    fd_order = cfg["fd_order"]

    x_cavity_start = cfg["x_cavity_start"]
    L_cavity = cfg["L_cavity"]
    xrange = (
        x_cavity_start - L_cavity,
        x_cavity_start + 2 * L_cavity,
    )
    yrange = (-1.0, 1.0)

    cmesh = Mesh.create(
        xrange,
        yrange,
        nx,
        ny,
        fd_order,
        y_symmetry=cfg["y_symmetry"],
        x_periodicity=cfg["x_periodicity"],
        alpha_filter=cfg["alpha_filter"],
    )
    # MATLAB convention: mask=True means "remove this point".
    # Python Mesh.mask() convention: mask=True means "keep this point".
    # Negate to convert.
    mask_x_left = cfg["mask_x_left"]
    mask_x_right = cfg["mask_x_right"]
    remove = (cmesh.Y < 0.0) & ((cmesh.X < mask_x_left) | (cmesh.X > mask_x_right))
    mesh = cmesh.mask(~remove)

    # --- y-deformation (mesh concentration + top stretch) ------------------
    deform = cfg["deform"]
    Y = mesh.Y.copy()
    a = deform["y_cubic_a"]
    Y = (Y**3 + a * Y) / (1 + a)
    stretch_n = deform["stretch_n"]
    y_threshold = deform["y_threshold"]
    y_amplitude = deform["y_amplitude"]
    p = Y > y_threshold
    Y[p] = Y[p] + y_amplitude * ((Y[p] - y_threshold) / (np.max(Y[p]) - y_threshold)) ** stretch_n
    mesh = mesh.deform(Y_new=Y)

    # --- x-stretch on the right ------------------------------------------
    x_threshold = deform["x_threshold"]
    x_amplitude = deform["x_amplitude"]
    X = mesh.X.copy()
    p = X > x_threshold
    X[p] = X[p] + x_amplitude * ((X[p] - x_threshold) / (np.max(X[p]) - x_threshold)) ** stretch_n
    mesh = mesh.deform(X_new=X)

    baseflow = load_cavity(mesh)
    sutherland_air(baseflow)

    sp_cfg = cfg["sponge"]
    xs_trans = sp_cfg["xs_trans"]
    xs_offset = sp_cfg["xs_offset"]
    ys_trans = sp_cfg["ys_trans"]
    ys_offset = sp_cfg["ys_offset"]
    sponge_amp = sp_cfg["amplitude"]
    sponge = np.zeros(mesh.X.shape, dtype=float)
    ind = mesh.used_ind
    x = mesh.X.ravel(order="F")[ind]
    y = mesh.Y.ravel(order="F")[ind]
    x_min = float(np.min(mesh.X))
    x_max = float(np.max(mesh.X))
    y_max = float(np.max(mesh.Y))
    term_x = np.tanh(-(x - x_min - xs_offset) / xs_trans) + 1
    term_y = np.tanh((x - x_max + xs_offset) / xs_trans) + 1
    term_t = np.tanh((y - y_max + ys_offset) / ys_trans) + 1
    sponge_vals = sponge_amp / 2 * np.maximum(term_t, term_x + term_y)
    sponge_flat = sponge.ravel(order="F")
    sponge_flat[ind] = sponge_vals
    sponge = sponge_flat.reshape(mesh.X.shape, order="F")
    mesh.sponge = sponge

    L0, idx, bc_indices = build_lnse_system(
        mesh,
        baseflow,
        model=cfg.get("model", "2D"),
        kx=beta,
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=bc_penalty,
        backend=assembly_backend,
        solver_cfg=solver_cfg,
    )

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
    omega, direct_modes = temporal_eigenvalues(
        L0,
        n_modes,
        omega_target,
        W_diag=W_diag,
        ncv=arnoldi_ncv,
        solver_cfg=solver_cfg,
    )
    omega_adj_target = 1j * np.conj(omega_target)
    omega_adj, adjoint_modes = adjoint_eigenvalues(
        L0,
        n_modes,
        omega_adj_target,
        W_diag,
        invW_diag,
        ncv=arnoldi_ncv,
        solver_cfg=solver_cfg,
    )

    # ---- Eigenvalue self-residuals (informational; PASS gate unchanged) ----
    # Direct: temporal_eigenvalues solves iL0 q = omega q.
    # Adjoint: adjoint_eigenvalues solves W^{-1} L0^H W q = omega_adj q,
    # i.e., L0^H (W*q) = omega_adj (W*q).  Both residuals are in the plain
    # l2 norm for consistency; the cavity already passes comfortably on the
    # spectrum-match criterion and these are belt-and-suspenders only.
    iL0 = 1j * L0
    direct_residuals = np.array([
        np.linalg.norm(iL0 @ q - om * q) / (abs(om) * max(float(np.linalg.norm(q)), 1e-300))
        for om, q in zip(omega, direct_modes.T)
    ])
    # Adjoint operator: W^{-1} L0^H W — check via L0^H (W q) - omega (W q).
    adjoint_residuals = np.array([
        np.linalg.norm(L0.conj().T @ (W_diag * q) - om_adj * (W_diag * q))
        / (abs(om_adj) * max(float(np.linalg.norm(W_diag * q)), 1e-300))
        for om_adj, q in zip(omega_adj, adjoint_modes.T)
    ])
    print("Direct eigenvalue residuals ||iL0 q - omega q|| / (|omega| ||q||):")
    for i, (om, res) in enumerate(zip(omega, direct_residuals)):
        print(f"  [{i}]  omega={om.real:.5f}{om.imag:+.5f}j   res={res:.3e}")
    print("Adjoint eigenvalue residuals ||L0^H (Wq) - omega_adj (Wq)|| / (|omega_adj| ||Wq||):")
    for i, (om, res) in enumerate(zip(omega_adj, adjoint_residuals)):
        print(f"  [{i}]  omega_adj={om.real:.5f}{om.imag:+.5f}j   res={res:.3e}")

    np.savez_compressed(
        OUTPUT_NPZ,
        omega=omega,
        omega_adj=omega_adj,
        direct_modes=direct_modes,
        adjoint_modes=adjoint_modes,
        direct_residuals=direct_residuals,
        adjoint_residuals=adjoint_residuals,
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
        baseflow_u=baseflow.U,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def _load_baseflow_u(data) -> np.ndarray:
    """Reload the cavity baseflow when plotting from an older cache."""
    if "baseflow_u" in data:
        return np.asarray(data["baseflow_u"], dtype=float)
    try:
        bundle = load_parity_npz("example_4_Cavity_results_base")
    except FileNotFoundError:
        bundle = None
    if bundle is not None:
        return np.asarray(parity_npz_get(bundle, "results_Cavity_base/baseFlow/U"), dtype=float).T
    from linstabpy.baseflow import load_cavity

    return np.asarray(load_cavity(mesh_view(data["X"], data["Y"], used_ind=data["used_ind"])).U, dtype=float)


def _style_cavity_axis(ax) -> None:
    """Apply common styling for cavity plots with masked walls."""
    ax.set_facecolor("#8e8e8e")
    ax.set_xlim(float(np.min(ax.get_xlim())), float(np.max(ax.get_xlim())))


def _cavity_baseflow_plot_field(X: np.ndarray, Y: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Mask wall-line cells that show spurious corner values in the baseflow preview."""
    cavity_left = 0.34029
    cavity_right = 2.34
    y_floor = float(np.nanmin(Y))
    x_spacing = float(np.min(np.diff(np.unique(np.asarray(X, dtype=float)[0, :]))))
    y_spacing = float(np.min(np.diff(np.unique(np.asarray(Y, dtype=float)[:, 0]))))
    x_tol = 0.55 * x_spacing
    y_tol = 0.55 * y_spacing

    plot_field = np.array(field, dtype=float, copy=True)
    floor_mask = (Y <= y_floor + y_tol) & (X >= cavity_left - x_tol) & (X <= cavity_right + x_tol)
    left_wall_mask = (np.abs(X - cavity_left) <= x_tol) & (Y <= 0.0 + y_tol)
    right_wall_mask = (np.abs(X - cavity_right) <= x_tol) & (Y <= 0.0 + y_tol)
    plot_field[floor_mask | left_wall_mask | right_wall_mask] = np.nan
    return plot_field


def _draw_cavity_outline(ax) -> None:
    """Draw the cavity floor and sidewalls so masked wall cells remain visible."""
    cavity_left = 0.34029
    cavity_right = 2.34
    y_floor = -1.0
    ax.plot([cavity_left, cavity_left], [0.0, y_floor], color="black", lw=1.1)
    ax.plot([cavity_left, cavity_right], [y_floor, y_floor], color="black", lw=1.1)
    ax.plot([cavity_right, cavity_right], [y_floor, 0.0], color="black", lw=1.1)


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
    baseflow_u = _load_baseflow_u(data)
    summary_modes = min(2, n_plot)
    plot_xlim = (-1.65, 5.34)
    plot_ylim = (-1.0, 0.6)
    matlab_direct, matlab_adjoint = _load_matlab_spectra()

    direct_order = np.argsort(np.imag(omega))[::-1][:n_plot]

    # Convert adjoint eigenvalues to direct convention for plotting/sorting.
    # Python adjoint_eigenvalues returns eig(W⁻¹ L0ᴴ W) = i·conj(omega_d).
    # MATLAB returns eig(W⁻¹ (iL0)ᴴ W) = conj(omega_d).
    # Recover direct eigenvalues: omega_d_from_adj = conj(-i·omega_adj).
    omega_d_from_adj = np.conj(-1j * omega_adj)
    adj_order = np.argsort(np.imag(omega_d_from_adj))[::-1][:n_plot]

    fig_base, ax_base = plt.subplots(1, 1, figsize=(11, 3.6))
    plot_scalar_panel(
        ax_base,
        X,
        Y,
        _cavity_baseflow_plot_field(X, Y, baseflow_u),
        title=r"Cavity baseflow $u/U_\infty$",
        cmap="turbo",
        xlim=plot_xlim,
        ylim=plot_ylim,
        aspect="auto",
        levels=101,
        colorbar_kwargs={"pad": 0.012},
        title_pad=4.0,
    )
    ax_base.set_facecolor("#8e8e8e")
    _draw_cavity_outline(ax_base)
    label_axes(ax_base)
    save_figure(fig_base, OUTPUT_BASEFLOW)

    fig = plt.figure(figsize=(12.5, 8.3))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.25, 1.0, 1.0),
        hspace=0.34,
        wspace=0.18,
    )
    ax = fig.add_subplot(grid[0, :])
    if matlab_direct is not None:
        ax.plot(
            np.real(matlab_direct),
            np.imag(matlab_direct),
            "o",
            mfc="white",
            mec="#2a5caa",
            mew=1.2,
            ms=7,
            label="MATLAB direct",
            zorder=2,
        )
    ax.plot(
        np.real(omega),
        np.imag(omega),
        "o",
        color="#2a5caa",
        ms=6,
        mew=1,
        label="Python direct",
        zorder=3,
    )
    if matlab_adjoint is not None:
        ax.plot(
            np.real(matlab_adjoint),
            np.imag(matlab_adjoint),
            "s",
            mfc="white",
            mec="#d99100",
            mew=1.2,
            ms=6.5,
            label="MATLAB adjoint",
            zorder=2,
        )
    ax.plot(
        np.real(omega_d_from_adj),
        np.imag(omega_d_from_adj),
        "x",
        color="#d99100",
        ms=10,
        mew=2.0,
        label="Python adjoint",
        zorder=4,
    )
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.set_title("Temporal spectra", pad=4.0)
    ax.grid(True, alpha=0.3)
    format_scientific_ticks(ax)
    ax.legend(loc="upper right", frameon=False)

    for i, j in enumerate(direct_order):
        mode = direct_modes[:, int(j)]
        fig_mode, axes = plt.subplots(
            3,
            2,
            figsize=(12, 10),
            gridspec_kw={"wspace": 0.18, "hspace": 0.28},
        )
        fields = [
            (idx_rho, r"$\rho$"),
            (idx_u, r"$u$"),
            (idx_v, r"$v$"),
            (idx_w, r"$w$"),
            (idx_T, r"$T$"),
        ]
        for k, (idx_var, title_) in enumerate(fields):
            ax_k = axes.flat[k]
            field = to_grid(mode[idx_var], used_ind, X.shape)
            plot_signed_panel(
                ax_k,
                X,
                Y,
                np.real(field),
                title=rf"Direct {title_}, $\omega = {format_complex(omega[j])}$",
                xlim=plot_xlim,
                ylim=plot_ylim,
                aspect="auto",
                title_pad=4.0,
            )
            ax_k.set_facecolor("#8e8e8e")
            label_axes(ax_k)
        axes.flat[-1].set_visible(False)

        set_suptitle(
            fig_mode,
            f"Direct mode {int(j) + 1} (rank {i + 1} by growth rate)",
            y=0.985,
            fontsize=15,
        )
        apply_figure_layout(fig_mode, rect=(0.0, 0.0, 1.0, 0.965), pad=0.7)
        if i == 0:
            save_figure(fig_mode, output_direct)
        else:
            plt.close(fig_mode)

    for i, j in enumerate(adj_order):
        mode = adjoint_modes[:, int(j)]
        fig_mode, axes = plt.subplots(
            3,
            2,
            figsize=(12, 10),
            gridspec_kw={"wspace": 0.18, "hspace": 0.28},
        )
        fields = [
            (idx_rho, r"$\rho$"),
            (idx_u, r"$u$"),
            (idx_v, r"$v$"),
            (idx_w, r"$w$"),
            (idx_T, r"$T$"),
        ]
        for k, (idx_var, title_) in enumerate(fields):
            ax_k = axes.flat[k]
            field = to_grid(mode[idx_var], used_ind, X.shape)
            plot_signed_panel(
                ax_k,
                X,
                Y,
                np.real(field),
                title=rf"Adjoint {title_}, $\omega = {format_complex(omega_d_from_adj[j])}$",
                xlim=plot_xlim,
                ylim=plot_ylim,
                aspect="auto",
                title_pad=4.0,
            )
            ax_k.set_facecolor("#8e8e8e")
            label_axes(ax_k)
        axes.flat[-1].set_visible(False)

        set_suptitle(
            fig_mode,
            f"Adjoint mode {int(j) + 1} (rank {i + 1} by growth rate)",
            y=0.985,
            fontsize=15,
        )
        apply_figure_layout(fig_mode, rect=(0.0, 0.0, 1.0, 0.965), pad=0.7)
        if i == 0:
            save_figure(fig_mode, output_adjoint)
        else:
            plt.close(fig_mode)

    for row, order, modes, label in [
        (1, adj_order[:summary_modes], adjoint_modes, "Adjoint"),
        (2, direct_order[:summary_modes], direct_modes, "Direct"),
    ]:
        for col, j in enumerate(order):
            ax_mode = fig.add_subplot(grid[row, col])
            field = to_grid(modes[:, int(j)][idx_u], used_ind, X.shape)
            plot_signed_panel(
                ax_mode,
                X,
                Y,
                np.real(field),
                title=(
                    rf"{label} $u$, $\omega = {format_complex(omega_d_from_adj[j])}$"
                    if label == "Adjoint"
                    else rf"{label} $u$, $\omega = {format_complex(omega[j])}$"
                ),
                xlim=plot_xlim,
                ylim=plot_ylim,
                aspect="auto",
                colorbar=False,
                title_pad=4.0,
            )
            ax_mode.set_facecolor("#8e8e8e")
            ax_mode.set_ylabel(r"$y$")
            if row == 2:
                ax_mode.set_xlabel(r"$x$")
            else:
                ax_mode.set_xticklabels([])

    set_suptitle(fig, "Cavity direct and adjoint mode support", y=0.972)
    apply_figure_layout(fig, rect=(0.0, 0.0, 1.0, 0.94), pad=0.7)
    save_figure(fig, output_spectrum)


def main():
    cfg = load_jsonc(SCRIPT.with_suffix(".jsonc"))
    data = load_or_compute(OUTPUT_NPZ, lambda: compute(cfg))
    plot(data)


if __name__ == "__main__":
    main()
