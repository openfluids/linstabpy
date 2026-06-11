#!/usr/bin/env python3
"""
Example 5: Paraboloid resolvent analysis.

Translates example_5_Paraboloid.m using a curvilinear transformation to a
parabolic physical grid and an axisymmetric model.

OUTPUTS:
  - example_5_paraboloid.npz
  - example_5_paraboloid_baseflow.png
  - example_5_paraboloid.png
  - example_5_paraboloid_support_components.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from linstabpy.utils.jsonc import load_jsonc

from ._helpers import (
    apply_figure_layout,
    label_axes,
    load_or_compute,
    load_parity_npz,
    mesh_view,
    parity_npz_get,
    plot_gain_bars,
    plot_scalar_panel,
    plot_signed_panel,
    save_figure,
    set_suptitle,
)

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
OUTPUT_BASEFLOW = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_baseflow.png")
OUTPUT_SUPPORT = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_support_components.png")
MATLAB_RESULTS_STEM = "example_5_Paraboloid_results"


def _load_matlab_gains() -> np.ndarray | None:
    """Load MATLAB gain references when the extracted parity bundle is available."""
    try:
        bundle = load_parity_npz(MATLAB_RESULTS_STEM)
    except FileNotFoundError:
        return None
    return np.asarray(parity_npz_get(bundle, "results_Paraboloid/gains"), dtype=float).reshape(-1)


def compute(cfg: dict):
    """Run resolvent analysis on the paraboloid-mapped cavity.

    Uses m = 5 azimuthal Fourier mode and axy model.
    """
    import scipy.sparse as sp

    from linstabpy.baseflow import load_paraboloid, sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system_result
    from linstabpy.operators.energy_norm import compute_chu_norm
    from linstabpy.resolvent.analysis import resolvent_analysis
    from linstabpy.solvers.petsc_utils import scipy_to_petsc

    solver_cfg = cfg.get("solver")

    omega = cfg["omega"]
    omega_shift_fallback = cfg["omega_shift_fallback"]
    use_shift_fallback = cfg["use_omega_shift_fallback"]
    n_modes = cfg["n_modes"]
    m = cfg["m"]
    fd_order = cfg["fd_order"]

    Nx = cfg["nx"]
    Ny = cfg["ny"]
    Lx = cfg["Lx"]
    Ly = cfg["Ly"]

    xrange = tuple(cfg["xrange"])
    yrange = tuple(cfg["yrange"])
    cmesh = Mesh.create(
        xrange,
        yrange,
        Nx,
        Ny,
        fd_order,
        y_symmetry=cfg["y_symmetry"],
        x_periodicity=cfg["x_periodicity"],
        alpha_filter=cfg["alpha_filter"],
    )

    X = cmesh.X.copy()
    x_nonzero = X != 0.0
    X_mapped = np.zeros_like(X)
    X_mapped[x_nonzero] = 1 / (1 / (X[x_nonzero] ** 4) * np.sign(X[x_nonzero]) + 1 / X[x_nonzero])
    X = X_mapped
    X[np.isnan(X)] = 0.0
    X = X / np.max(np.abs(X))
    Y = cmesh.Y.copy()
    Y = 1 - (1 - Y) ** 1.4
    Y = 1 - (1 - Y) ** 1.4
    Y = np.maximum(Y, 1e-10)

    # Scale to physical domain before Z² conformal mapping (MATLAB convention)
    X = X * Lx - 0.5
    Y = Y * Ly

    Z = X + 1j * Y
    Xp = -np.real(Z**2)
    Yp = -np.imag(Z**2)
    mesh = cmesh.deform(X_new=Xp, Y_new=Yp)

    baseflow = load_paraboloid(mesh)
    sutherland_air(baseflow)

    xs = -0.5 - Lx + cfg["sponge"]["xs_offset"]
    xs_trans = Lx * cfg["sponge"]["xs_trans_fraction"]
    ys = Ly - cfg["sponge"]["ys_offset"]
    ys_trans = cfg["sponge"]["ys_trans"]
    sponge_amp = cfg["sponge"]["amplitude"]
    # MATLAB defines the sponge on the pre-conformal (x, y) grid, before the
    # final Z^2 mapping to the physical mesh.
    sponge_x = np.tanh(-(X - xs) / xs_trans) + 1
    sponge_y = np.tanh((Y - ys) / ys_trans) + 1
    mesh.sponge = sponge_amp / 2 * np.maximum(sponge_x, sponge_y)

    assembly_backend = cfg.get("assembly_backend")
    bc_penalty = cfg["bc"].get("penalty", 1e4)
    lnse_system = build_lnse_system_result(
        mesh,
        baseflow,
        model=cfg["model"],
        m=m,
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=bc_penalty,
        backend=assembly_backend,
        solver_cfg=solver_cfg,
    )
    L0 = lnse_system.operator
    idx = lnse_system.idx
    bc_indices = lnse_system.bc_indices

    used_ind = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[used_ind]
    rho = baseflow.RHO.ravel(order="F")[used_ind]
    T = baseflow.T.ravel(order="F")[used_ind]
    R = mesh.Y.ravel(order="F")[used_ind]
    W_diag, _ = compute_chu_norm(
        W_int,
        rho,
        T,
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
        model=cfg["model"],
        R=R,
    )

    B_diag = np.ones(5 * mesh.ngp)
    var_map = {"rho": "rho_j", "u": "u_j", "v": "v_j", "w": "w_j", "T": "T_j"}
    for var_name in cfg.get("forcing_zero_vars", []):
        B_diag[idx[var_map[var_name]]] = 0.0
    B_diag[bc_indices] = 0.0
    B_mat_sp = sp.diags(B_diag).tocsc()
    B_mat = scipy_to_petsc(B_mat_sp, solver_cfg) if hasattr(L0, "getInfo") else B_mat_sp
    C_mat = B_mat.copy()

    # Paraboloid at ω=0 can be singular depending on discretization and
    # boundary conditions; a small imaginary shift is a conservative fallback
    # to keep this example numerically robust in reduced-memory settings.
    solve_freqs = [complex(omega)]
    if use_shift_fallback and abs(omega) < 1e-12 and omega_shift_fallback > 0.0:
        solve_freqs.append(1j * omega_shift_fallback)
    last_error: Exception | None = None
    gains: np.ndarray
    responses: np.ndarray
    forcings: np.ndarray
    omega_used = solve_freqs[0]

    for candidate in solve_freqs:
        try:
            (
                gains,
                responses,
                forcings,
                svd_residuals_fwd,
                svd_residuals_adj,
            ) = resolvent_analysis(
                L0,
                candidate,
                n_modes,
                W_diag,
                B_mat=B_mat,
                C_mat=C_mat,
                filter_fn=mesh.filters.filter if mesh.filters is not None else None,
                filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
                solver_cfg=solver_cfg,
                residuals=True,
            )
            omega_used = candidate
            if candidate != solve_freqs[0]:
                print(
                    f"Using fallback frequency ω={candidate:.3g} to avoid singular/ill-conditioned solve at ω={solve_freqs[0]:.3g}."
                )
            break
        except RuntimeError as exc:
            last_error = exc
            if candidate == solve_freqs[0] and len(solve_freqs) > 1:
                print(
                    "Paraboloid resolvent failed at ω=0; retrying with small imaginary"
                    f" shift ω={solve_freqs[1]:.3g}."
                )
                continue
            raise

    if "gains" not in locals():
        raise RuntimeError(
            "Paraboloid resolvent failed for all frequency attempts."
        ) from last_error

    # ---- SVD self-residuals -----------------------------------------------
    # MATLAB used 5-output lu(A) (UMFPACK) + ARPACK eigs at tol=eps=2.2e-16,
    # default subspace opts.p = 2*nEig+1.  Python uses MUMPS/METIS + SLEPc EPS
    # at tol=1e-12, ncv = max(2*n_modes+3, 20).  Different fill-reducing
    # orderings yield gain mismatch ~0.04% for this case; structurally
    # unavoidable.  The residuals above are computed with the same shifted
    # factorization used for the accepted candidate frequency.

    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        svd_residuals_fwd=svd_residuals_fwd,
        svd_residuals_adj=svd_residuals_adj,
        idx_u_j=idx["u_j"],
        omega=omega,
        omega_used_real=float(np.real(omega_used)),
        omega_used_imag=float(np.imag(omega_used)),
        m=m,
        X=mesh.X,
        Y=mesh.Y,
        n_modes=n_modes,
        baseflow_u=baseflow.U,
        baseflow_v=baseflow.V,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def _upscale_field(
    X: np.ndarray,
    Y: np.ndarray,
    field: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a field for smoother presentation plots."""
    from scipy.ndimage import zoom

    if factor <= 1:
        return X, Y, field
    return (
        zoom(X, factor, order=1),
        zoom(Y, factor, order=1),
        zoom(field, factor, order=1),
    )


def _body_outline(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mapped body boundary from the rightmost mesh column."""
    x_body = np.asarray(X[:, -1], dtype=float)
    y_body = np.asarray(Y[:, -1], dtype=float)
    order = np.argsort(y_body)
    return x_body[order], y_body[order]


def _load_baseflow_fields(data) -> tuple[np.ndarray, np.ndarray]:
    """Reload the paraboloid baseflow when plotting from an older cache."""
    if "baseflow_u" in data and "baseflow_v" in data:
        return np.asarray(data["baseflow_u"], dtype=float), np.asarray(
            data["baseflow_v"], dtype=float
        )
    try:
        bundle = load_parity_npz(MATLAB_RESULTS_STEM)
    except FileNotFoundError:
        bundle = None
    if bundle is not None:
        base_u = np.asarray(parity_npz_get(bundle, "results_Paraboloid/baseFlow/U"), dtype=float).T
        base_v = np.asarray(parity_npz_get(bundle, "results_Paraboloid/baseFlow/V"), dtype=float).T
        return base_u, base_v
    from linstabpy.baseflow import load_paraboloid

    baseflow = load_paraboloid(mesh_view(data["X"], data["Y"]))
    return np.asarray(baseflow.U, dtype=float), np.asarray(baseflow.V, dtype=float)


def plot(data):
    """Plot paper-style and support figures for the paraboloid case."""
    import matplotlib.pyplot as plt

    gains = np.asarray(data["gains"], dtype=float).reshape(-1)
    responses = np.asarray(data["responses"], dtype=complex)
    forcings = np.asarray(data["forcings"], dtype=complex)
    idx_u = np.asarray(data["idx_u_j"], dtype=int)
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    n_modes = int(data["n_modes"])
    refine = 3
    x_body, y_body = _body_outline(X, Y)
    base_u, base_v = _load_baseflow_fields(data)
    speed = np.sqrt(base_u**2 + base_v**2)

    n_plot = min(n_modes, 3)
    crop_xlim = (float(np.min(X)), float(np.max(X)))
    crop_ylim = (0.0, 2.0)

    fig_base, ax_base = plt.subplots(1, 1, figsize=(11, 4.2))
    plot_scalar_panel(
        ax_base,
        X,
        Y,
        speed,
        title=r"Paraboloid baseflow speed $\sqrt{u^2+v^2}$",
        cmap="turbo",
        xlim=crop_xlim,
        ylim=(0.0, 1.6),
        aspect="auto",
        levels=101,
        colorbar_kwargs={"pad": 0.012},
        title_pad=4.0,
    )
    ax_base.plot(x_body, y_body, color="black", lw=1.6)
    label_axes(ax_base)
    save_figure(fig_base, OUTPUT_BASEFLOW)

    fig = plt.figure(figsize=(12.5, 8.0))
    grid = fig.add_gridspec(
        n_plot,
        3,
        width_ratios=(1.0, 2.0, 2.0),
        hspace=0.28,
        wspace=0.24,
    )
    ax_gain = fig.add_subplot(grid[:, 0])
    plot_gain_bars(ax_gain, gains[:n_plot], matlab_gains=_load_matlab_gains())

    for i in range(n_plot):
        f_u = np.asarray(forcings[idx_u, i], dtype=complex).reshape(X.shape, order="F")
        r_u = np.asarray(responses[idx_u, i], dtype=complex).reshape(X.shape, order="F")
        X_plot_f, Y_plot_f, f_plot = _upscale_field(X, Y, np.real(f_u), refine)
        X_plot_r, Y_plot_r, r_plot = _upscale_field(X, Y, np.real(r_u), refine)
        ax_f = fig.add_subplot(grid[i, 1])
        ax_r = fig.add_subplot(grid[i, 2])
        plot_signed_panel(
            ax_f,
            X_plot_f,
            Y_plot_f,
            f_plot,
            title=rf"$f_u^{{({i + 1})}}$",
            xlim=crop_xlim,
            ylim=crop_ylim,
            aspect="auto",
            colorbar=False,
            title_pad=4.0,
        )
        plot_signed_panel(
            ax_r,
            X_plot_r,
            Y_plot_r,
            r_plot,
            title=rf"$u^{{({i + 1})}}$",
            xlim=crop_xlim,
            ylim=crop_ylim,
            aspect="auto",
            colorbar=False,
            title_pad=4.0,
        )
        ax_f.plot(x_body, y_body, color="black", lw=1.2)
        ax_r.plot(x_body, y_body, color="black", lw=1.2)
        ax_f.text(
            0.03,
            0.88,
            rf"$\sigma_{i + 1}={gains[i]:.2f}$",
            transform=ax_f.transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ax_f.set_ylabel(r"$y$")
        ax_r.set_ylabel(r"$y$")
        if i < n_plot - 1:
            ax_f.set_xticklabels([])
            ax_r.set_xticklabels([])
        else:
            ax_f.set_xlabel(r"$x$")
            ax_r.set_xlabel(r"$x$")
    set_suptitle(fig, r"Paraboloid resolvent summary at $m = 5$", y=0.972)
    apply_figure_layout(fig, rect=(0.0, 0.0, 1.0, 0.94), pad=0.7)
    save_figure(fig, OUTPUT_PNG)

    fig_support, axes_support = plt.subplots(
        n_plot,
        2,
        figsize=(12.2, 4.2 * n_plot),
        gridspec_kw={"wspace": 0.18, "hspace": 0.32},
    )
    if n_plot == 1:
        axes_support = np.array([axes_support])
    for i in range(n_plot):
        f_u = np.asarray(forcings[idx_u, i], dtype=complex).reshape(X.shape, order="F")
        r_u = np.asarray(responses[idx_u, i], dtype=complex).reshape(X.shape, order="F")
        X_plot_f, Y_plot_f, f_plot = _upscale_field(X, Y, np.real(f_u), refine)
        X_plot_r, Y_plot_r, r_plot = _upscale_field(X, Y, np.real(r_u), refine)
        plot_signed_panel(
            axes_support[i, 0],
            X_plot_f,
            Y_plot_f,
            f_plot,
            title=rf"Full-domain $f_u^{{({i + 1})}}$",
            xlim=(float(np.min(X)), float(np.max(X))),
            ylim=(0.0, float(np.max(Y))),
            aspect="auto",
            title_pad=4.0,
        )
        plot_signed_panel(
            axes_support[i, 1],
            X_plot_r,
            Y_plot_r,
            r_plot,
            title=rf"Full-domain $u^{{({i + 1})}}$",
            xlim=(float(np.min(X)), float(np.max(X))),
            ylim=(0.0, float(np.max(Y))),
            aspect="auto",
            title_pad=4.0,
        )
        axes_support[i, 0].plot(x_body, y_body, color="black", lw=1.2)
        axes_support[i, 1].plot(x_body, y_body, color="black", lw=1.2)
        label_axes(axes_support[i, 0])
        label_axes(axes_support[i, 1])
    set_suptitle(fig_support, "Paraboloid support panels", y=0.975)
    apply_figure_layout(fig_support, rect=(0.0, 0.0, 1.0, 0.945), pad=0.7)
    save_figure(fig_support, OUTPUT_SUPPORT)


def main():
    cfg = load_jsonc(SCRIPT.with_suffix(".jsonc"))
    data = load_or_compute(OUTPUT_NPZ, lambda: compute(cfg), verbose=False)
    plot(data)


if __name__ == "__main__":
    main()
