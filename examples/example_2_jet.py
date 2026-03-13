#!/usr/bin/env python3
"""
Example 2: Jet resolvent analysis.

Translates example_2_Jet.m for an axisymmetric (m=0) jet base flow.

OUTPUTS:
  - example_2_jet.npz
  - example_2_jet.png
  - example_2_jet_baseflow.png
  - example_2_jet_support_components.png
USAGE:   python example_2_jet.py                       # .npz exists → plot only
         rm example_2_jet.npz && python example_2_jet.py  # recompute
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from linstab2d.utils.jsonc import load_jsonc
from linstab2d.boundary_conditions.recipe import apply_bc_recipe

from ._helpers import (
    label_axes,
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
REFERENCE_MAT = SCRIPT.with_name("example_2_reference.mat")
MATLAB_RESULTS_STEM = "example_2_Jet_results"

def _load_matlab_gains() -> np.ndarray | None:
    """Load MATLAB gain references when the extracted parity bundle is available."""
    try:
        bundle = load_parity_npz(MATLAB_RESULTS_STEM)
    except FileNotFoundError:
        return None
    return np.asarray(parity_npz_get(bundle, "results_Jet/S"), dtype=float).reshape(-1)


def _as_grid(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel(order="F")
    if arr.size != np.prod(shape):
        raise ValueError(f"{name} has size {arr.size}, expected {shape[0] * shape[1]}")
    return arr.reshape(shape, order="F")


def _reference_or_warn(label: str, actual: np.ndarray, key: str) -> None:
    from scipy.io import loadmat

    if not REFERENCE_MAT.exists():
        return
    try:
        reference = loadmat(REFERENCE_MAT)
    except Exception as exc:
        print(f"Reference comparison skipped ({REFERENCE_MAT.name} load failed): {exc}")
        return
    if key not in reference:
        print(f"Reference comparison skipped ({REFERENCE_MAT.name} has no '{key}' field)")
        return

    expected = np.asarray(reference[key]).reshape(-1)
    try:
        np.testing.assert_allclose(
            actual,
            expected[: actual.size],
            rtol=1e-6,
            atol=1e-6,
        )
    except AssertionError as exc:
        raise RuntimeError(f"MATLAB reference comparison failed for {label}") from exc
    print(f"Reference comparison passed for {label} ({REFERENCE_MAT.name})")


def _load_sponge(mesh, r_1D: np.ndarray, z_1D: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.io import loadmat

    from linstab2d.baseflow.loaders import _BASEFLOW_DATA_DIR

    mat = loadmat(_BASEFLOW_DATA_DIR / "M09jet.mat")
    source_shape = (int(r_1D.size), int(z_1D.size))
    sponge = _as_grid(mat["spongeFun"], source_shape, "spongeFun")
    if tuple(mesh.X.shape) == source_shape:
        return sponge

    # Mesh grid is built as np.meshgrid(z_1D, r_1D, indexing="xy"), so axis 0 is r and axis 1 is z.
    interpolator = RegularGridInterpolator(
        (r_1D.astype(float), z_1D.astype(float)),
        sponge,
        bounds_error=False,
        fill_value=np.nan,
    )
    query = np.column_stack(
        (
            mesh.Y.ravel(order="F").astype(float),
            mesh.X.ravel(order="F").astype(float),
        )
    )
    return interpolator(query).reshape(mesh.X.shape, order="F")


def _print_velocity_scales(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> None:
    print(
        "Jet velocity scales: "
        f"|U|_max={np.max(np.abs(U)):.3e}, "
        f"|V|_max={np.max(np.abs(V)):.3e}, "
        f"|W|_max={np.max(np.abs(W)):.3e}"
    )


def _dominant_component(fields: dict[str, np.ndarray]) -> str:
    norms = {name: np.linalg.norm(np.abs(field)) for name, field in fields.items()}
    return max(norms, key=norms.get)


def compute(cfg: dict):
    """Run jet resolvent analysis."""
    import scipy.sparse as sp
    from scipy.io import loadmat

    from linstab2d.baseflow import load_jet, sutherland_air
    from linstab2d.baseflow.loaders import _BASEFLOW_DATA_DIR
    from linstab2d.mesh import Mesh
    from linstab2d.operators.assembly import build_lnse_operator
    from linstab2d.operators.energy_norm import compute_chu_norm
    from linstab2d.solvers.resolvent import compute_resolvent_svd_residuals, resolvent_analysis

    solver_cfg = cfg.get("solver")

    St = cfg["St"]
    omega = 2 * np.pi * St
    coarsen = max(1, cfg["coarsen"])
    n_modes = cfg["n_modes"]
    m = cfg["m"]
    fd_order = cfg["fd_order"]
    model = cfg["model"]

    data = loadmat(_BASEFLOW_DATA_DIR / "M09jet.mat")
    r_1D = np.asarray(data["r_1D"], dtype=float).ravel()
    z_1D = np.asarray(data["z_1D"], dtype=float).ravel()
    nr = int(r_1D[::coarsen].size)
    nz = int(z_1D[::coarsen].size)
    print(f"Jet mesh grid points: nz={nz}, nr={nr}, coarsen={coarsen}")
    r_1D_coarse = r_1D[::coarsen]
    z_1D_coarse = z_1D[::coarsen]

    xrange = (0.0, 1.0)
    yrange = (0.0, 1.0)

    cmesh = Mesh.create(
        xrange,
        yrange,
        nz,
        nr,
        fd_order,
        y_symmetry=cfg["y_symmetry"],
        x_periodicity=cfg["x_periodicity"],
        alpha_filter=cfg["alpha_filter"],
    )
    X = np.meshgrid(z_1D_coarse, r_1D_coarse, indexing="xy")[0]
    Y = np.meshgrid(z_1D_coarse, r_1D_coarse, indexing="xy")[1]
    mesh = cmesh.deform(X_new=X, Y_new=Y)

    baseflow = load_jet(mesh)
    sutherland_air(baseflow)
    _print_velocity_scales(
        baseflow.U,
        baseflow.V,
        baseflow.W,
    )
    mesh.sponge = _load_sponge(mesh, r_1D, z_1D)

    t_lnse = time.perf_counter()
    L0, idx = build_lnse_operator(mesh, baseflow, model=model, m=m)
    t_lnse = time.perf_counter() - t_lnse

    bc_recipe = cfg["bc"]["recipe"]
    L0, _ = apply_bc_recipe(L0, mesh, idx, bc_recipe)

    used_ind = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[used_ind]
    rho_p = baseflow.RHO.ravel(order="F")[used_ind]
    T_p = baseflow.T.ravel(order="F")[used_ind]
    R = mesh.Y.ravel(order="F")[used_ind]
    W_diag, _ = compute_chu_norm(
        W_int,
        rho_p,
        T_p,
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
        model=model,
        R=R,
    )

    sponge0 = (mesh.sponge.ravel(order="F")[used_ind] == 0.0).astype(float)
    # Match the MATLAB example literally: kron(B(:), ones(5,1)).
    # This repeats each grid-point mask entry across the flattened 5-field
    # vector in the same order as the reference workflow.
    B_diag = np.repeat(sponge0, 5)
    B_mat = sp.diags(B_diag).tocsc()

    t_resolvent = time.perf_counter()
    gains, responses, forcings = resolvent_analysis(
        L0,
        omega,
        n_modes,
        W_diag,
        B_mat=B_mat,
        C_mat=B_mat,
        filter_fn=mesh.filters.filter if mesh.filters is not None else None,
        filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
        solver_cfg=solver_cfg,
    )
    t_resolvent = time.perf_counter() - t_resolvent
    print(f"Resolvent gains: {gains}")
    print(f"Stage 1C timing: build_lnse_operator={t_lnse:.2f}s, resolvent_analysis={t_resolvent:.2f}s")

    if REFERENCE_MAT.exists():
        _reference_or_warn("resolvent gains", gains, "gains")

    # ---- SVD self-residuals -----------------------------------------------
    # MATLAB used 5-output lu(A) (UMFPACK) + ARPACK eigs at tol=eps=2.2e-16,
    # default subspace opts.p = 2*nEig+1.  Python uses MUMPS/METIS + SLEPc EPS
    # at tol=1e-12, ncv = max(2*n_modes+3, 20).  Different fill-reducing
    # orderings yield gain mismatch ~0.20% for this case; structurally
    # unavoidable.  Residuals below are the primary closure criterion.
    svd_residuals_fwd, svd_residuals_adj = compute_resolvent_svd_residuals(
        L0,
        omega,
        B_mat,
        B_mat,
        gains,
        forcings,
        responses,
        W_diag,
        filter_fn=mesh.filters.filter if mesh.filters is not None else None,
        filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
        solver_cfg=solver_cfg,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        svd_residuals_fwd=svd_residuals_fwd,
        svd_residuals_adj=svd_residuals_adj,
        omega=omega,
        n_modes=n_modes,
        X=mesh.X,
        Y=mesh.Y,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
        ngp=mesh.ngp,
        baseflow_u=baseflow.U,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def _load_baseflow_u(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Reload the jet baseflow when plotting from an older cache."""
    if "baseflow_u" in data:
        return np.asarray(data["baseflow_u"], dtype=float)
    try:
        bundle = load_parity_npz(MATLAB_RESULTS_STEM)
    except FileNotFoundError:
        bundle = None
    if bundle is not None:
        return np.asarray(parity_npz_get(bundle, "results_Jet/baseFlow/U"), dtype=float).T
    from linstab2d.baseflow import load_jet

    return np.asarray(load_jet(mesh_view(data["X"], data["Y"])).U, dtype=float)


def _load_index_vector(
    data: np.lib.npyio.NpzFile,
    key: str,
    ngp: int,
    start: int,
) -> np.ndarray:
    if key in data:
        idx = np.asarray(data[key], dtype=int)
    else:
        idx = np.arange(start, start + ngp, dtype=int)
    return idx


def _read_field_components(
    field_data: np.ndarray,
    idx: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    return np.asarray(field_data[idx, :], dtype=complex).reshape((shape[0], shape[1], -1), order="F")


def plot(data: np.lib.npyio.NpzFile) -> None:
    """Plot paper-style and support figures for the jet example."""
    import matplotlib.pyplot as plt

    gains = np.asarray(data["gains"], dtype=float).reshape(-1)
    responses = np.asarray(data["responses"], dtype=complex)
    forcings = np.asarray(data["forcings"], dtype=complex)
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    shape = X.shape
    ngp = int(data["ngp"])
    idx_rho = _load_index_vector(data, "idx_rho_j", ngp, 0)
    idx_u = _load_index_vector(data, "idx_u_j", ngp, ngp)
    idx_v = _load_index_vector(data, "idx_v_j", ngp, 2 * ngp)
    idx_w = _load_index_vector(data, "idx_w_j", ngp, 3 * ngp)
    idx_T = _load_index_vector(data, "idx_T_j", ngp, 4 * ngp)

    f_rho = _read_field_components(forcings, idx_rho, shape)
    f_u = _read_field_components(forcings, idx_u, shape)
    f_v = _read_field_components(forcings, idx_v, shape)
    f_w = _read_field_components(forcings, idx_w, shape)
    f_T = _read_field_components(forcings, idx_T, shape)

    r_rho = _read_field_components(responses, idx_rho, shape)
    r_u = _read_field_components(responses, idx_u, shape)
    r_v = _read_field_components(responses, idx_v, shape)
    r_w = _read_field_components(responses, idx_w, shape)
    r_T = _read_field_components(responses, idx_T, shape)

    n_modes = min(int(data["n_modes"]), len(gains))
    baseflow_u = _load_baseflow_u(data)
    base_xlim = (0.0, 30.0)
    base_ylim = (0.0, 3.0)
    mode_xlim = (0.0, 20.0)
    mode_ylim = (0.0, 1.05)

    fig_base, ax_base = plt.subplots(1, 1, figsize=(11, 3.3))
    plot_scalar_panel(
        ax_base,
        X,
        Y,
        baseflow_u / max(float(np.max(baseflow_u)), 1e-12),
        title=r"Axisymmetric jet baseflow $u/U_j$",
        cmap="turbo",
        xlim=base_xlim,
        ylim=base_ylim,
        aspect="auto",
        levels=101,
        colorbar_kwargs={"pad": 0.012},
        title_pad=4.0,
    )
    label_axes(ax_base)
    ax_base.annotate(
        r"$U_j$",
        xy=(0.22, 0.18),
        xytext=(0.08, 0.18),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "black"},
        fontsize=16,
        va="center",
    )
    save_figure(fig_base, OUTPUT_BASEFLOW)

    fig = plt.figure(figsize=(12.5, 7.8))
    grid = fig.add_gridspec(
        n_modes,
        3,
        width_ratios=(1.0, 2.2, 2.2),
        hspace=0.28,
        wspace=0.22,
    )
    ax_gain = fig.add_subplot(grid[:, 0])
    plot_gain_bars(ax_gain, gains[:n_modes], matlab_gains=_load_matlab_gains())
    for i in range(n_modes):
        ax_f = fig.add_subplot(grid[i, 1])
        ax_r = fig.add_subplot(grid[i, 2])
        plot_signed_panel(
            ax_f,
            X,
            Y,
            np.real(f_u[:, :, i]),
            title=rf"$f_u^{{({i + 1})}}$",
            xlim=mode_xlim,
            ylim=mode_ylim,
            aspect="auto",
            colorbar=False,
            title_pad=4.0,
        )
        plot_signed_panel(
            ax_r,
            X,
            Y,
            np.real(r_u[:, :, i]),
            title=rf"$u^{{({i + 1})}}$",
            xlim=mode_xlim,
            ylim=mode_ylim,
            aspect="auto",
            colorbar=False,
            title_pad=4.0,
        )
        ax_f.text(
            0.02,
            0.88,
            rf"$\sigma_{i + 1}={gains[i]:.2f}$",
            transform=ax_f.transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ax_f.set_ylabel(r"$y$")
        ax_r.set_ylabel(r"$y$")
        if i < n_modes - 1:
            ax_f.set_xticklabels([])
            ax_r.set_xticklabels([])
        else:
            ax_f.set_xlabel(r"$x$")
            ax_r.set_xlabel(r"$x$")
    set_suptitle(fig, r"Jet resolvent summary at $St = 0.39$", y=0.972)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.7)
    save_figure(fig, OUTPUT_PNG)

    fig_support, axes_support = plt.subplots(
        n_modes,
        2,
        figsize=(12.2, 3.8 * n_modes),
        gridspec_kw={"wspace": 0.18, "hspace": 0.34},
    )
    if n_modes == 1:
        axes_support = np.array([axes_support])
    forcing_fields = {"rho": f_rho, "u": f_u, "v": f_v, "w": f_w, "T": f_T}
    response_fields = {"rho": r_rho, "u": r_u, "v": r_v, "w": r_w, "T": r_T}
    for i in range(n_modes):
        f_name = _dominant_component(forcing_fields)
        r_name = _dominant_component(response_fields)
        plot_signed_panel(
            axes_support[i, 0],
            X,
            Y,
            np.real(forcing_fields[f_name][:, :, i]),
            title=rf"Full-domain $f_{{{f_name}}}^{{({i + 1})}}$",
            xlim=(0.0, float(np.max(X))),
            ylim=(0.0, 3.0),
            aspect="auto",
            robust_percentile=98.0,
            title_pad=4.0,
        )
        plot_signed_panel(
            axes_support[i, 1],
            X,
            Y,
            np.real(response_fields[r_name][:, :, i]),
            title=rf"Full-domain ${r_name}^{{({i + 1})}}$",
            xlim=(0.0, float(np.max(X))),
            ylim=(0.0, 3.0),
            aspect="auto",
            robust_percentile=98.0,
            title_pad=4.0,
        )
        label_axes(axes_support[i, 0])
        label_axes(axes_support[i, 1])
    set_suptitle(
        fig_support,
        "Jet support panels with full-domain component selection",
        y=0.975,
    )
    fig_support.tight_layout(rect=(0.0, 0.0, 1.0, 0.945), pad=0.7)
    save_figure(fig_support, OUTPUT_SUPPORT)


def main():
    cfg = load_jsonc(SCRIPT.with_suffix(".jsonc"))
    try:
        data = np.load(OUTPUT_NPZ)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        compute(cfg)
        data = np.load(OUTPUT_NPZ)

    plot(data)


if __name__ == "__main__":
    main()
