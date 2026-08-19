"""Presentation figures for Martini Example 5 (paraboloid resolvent).

Matches the thesis/campaign layout under
``cases/Martini2024/comparison/plots/example_5_paraboloid_resolvent/``.

- ``example_5_paraboloid_baseflow.png`` — baseflow :math:`|\\mathbf{u}_b|`.
- ``example_5_paraboloid_modes_forcing_response_{u,v,w}[_imag]_mosaic.png`` — forcing/response ranks 1–3
  (LinStab2D | linstabpy | error triptych).

Uses the **fixed** ``Mesh.deform`` metric terms (see ``example_5_D2y_asymm_fix.md``).

LinStab2D references store **pre-** :math:`C` responses and export forcings as
:math:`B v` (``returnPhysicalSpace``). Mosaics multiply MATLAB responses by
:math:`C=B` and Python forcings by the same diagonal :math:`B` before error
panels (like Examples 1–2). Python NPZ responses are already post-:math:`C`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from linstabpy.utils.yaml_io import load_config

from examples.martini.compare_martini_examples import (
    LINSTAB2D_RESULTS_DIRNAME,
    _mode_matrix,
    _npz_get,
    load_matlab_bundle,
)
from examples.martini.plot_compare_utils import (
    apply_diagonal_c,
    compact_uvw_component_idx,
    ensure_matplotlib,
    load_python_npz,
    match_modes_by_gains,
    matlab_component_grid,
    maybe_compact_python_resolvent_uvw,
    patch_matlab_compact_uvw_idx,
    plot_mode_triptych_mosaic,
    python_component_grid,
    scaled_python_mode,
)

PYTHON_NPZ = "example_5_paraboloid.npz"
PYTHON_YAML = "example_5_paraboloid.yaml"
MATLAB_STEM = "example_5_Paraboloid_results"
IDX_PREFIX = "results_Paraboloid/idx"
PLOT_GRID_PREFIX = "results_Paraboloid/plot_grid"

FIGURE_PREFIX = "example_5_paraboloid"
OUT_BASEFLOW = f"{FIGURE_PREFIX}_baseflow.png"

N_MODES = 3
DEFAULT_COMPONENTS = ("u", "v", "w")
DEFAULT_PARTS = ("real", "imag")
MOSAIC_YLIM = (0.0, 2.0)
BASEFLOW_YLIM = (0.0, 2.0)
MOSAIC_ROW_HEIGHT = 1.55

TITLE_FS = 26
LABEL_FS = 24
TICK_FS = 20
CBAR_FS = 18
DPI = 400


def linstab2d_results_dir(example_dir: Path) -> Path:
    return example_dir / LINSTAB2D_RESULTS_DIRNAME


def ex5_observation_mask_diag(
    cfg: dict[str, Any],
    *,
    n_mode_dofs: int | None = None,
) -> np.ndarray:
    """Block-correct :math:`B=C` diagonal for Ex.~5 (zeros rho,T and Dirichlet DOFs)."""
    from examples.martini.matlab_baseflow import load_paraboloid
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system_result

    solver_cfg = cfg.get("solver")
    Nx = cfg["nx"]
    Ny = cfg["ny"]
    Lx = cfg["Lx"]
    Ly = cfg["Ly"]

    cmesh = Mesh.create(
        tuple(cfg["xrange"]),
        tuple(cfg["yrange"]),
        Nx,
        Ny,
        cfg["fd_order"],
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
    X = X * Lx - 0.5
    Y = Y * Ly

    Z = X + 1j * Y
    mesh = cmesh.deform(X_new=-np.real(Z**2), Y_new=-np.imag(Z**2))

    baseflow = load_paraboloid(mesh)
    sutherland_air(baseflow)

    xs = -0.5 - Lx + cfg["sponge"]["xs_offset"]
    xs_trans = Lx * cfg["sponge"]["xs_trans_fraction"]
    ys = Ly - cfg["sponge"]["ys_offset"]
    ys_trans = cfg["sponge"]["ys_trans"]
    sponge_x = np.tanh(-(X - xs) / xs_trans) + 1
    sponge_y = np.tanh((Y - ys) / ys_trans) + 1
    mesh.sponge = cfg["sponge"]["amplitude"] / 2 * np.maximum(sponge_x, sponge_y)

    lnse_system = build_lnse_system_result(
        mesh,
        baseflow,
        model=cfg["model"],
        m=cfg["m"],
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=cfg["bc"].get("penalty", 1e4),
        backend=cfg.get("assembly_backend"),
        solver_cfg=solver_cfg,
    )
    idx = lnse_system.idx
    bc_indices = lnse_system.bc_indices

    var_map = {"rho": "rho_j", "u": "u_j", "v": "v_j", "w": "w_j", "T": "T_j"}
    c_diag = np.ones(5 * mesh.ngp)
    for var_name in cfg.get("forcing_zero_vars", []):
        c_diag[idx[var_map[var_name]]] = 0.0
    c_diag[bc_indices] = 0.0

    if n_mode_dofs is None or n_mode_dofs == c_diag.size:
        return c_diag
    uvw_dofs = int(idx["u_j"].size + idx["v_j"].size + idx["w_j"].size)
    if n_mode_dofs == uvw_dofs:
        return np.concatenate(
            [
                c_diag[idx["u_j"]],
                c_diag[idx["v_j"]],
                c_diag[idx["w_j"]],
            ]
        )
    raise ValueError(
        f"Ex5 observation mask length {c_diag.size} (full) / {uvw_dofs} (u+v+w) "
        f"does not match mode DOFs {n_mode_dofs}"
    )


def mosaic_name(component: str, part: str) -> str:
    if part == "real":
        return f"{FIGURE_PREFIX}_modes_forcing_response_{component}_mosaic.png"
    return f"{FIGURE_PREFIX}_modes_forcing_response_{component}_imag_mosaic.png"


def _part_tex(part: str) -> str:
    if part == "imag":
        return r"\Im"
    return r"\Re"


def _body_outline(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_body = np.asarray(X[:, -1], dtype=float)
    y_body = np.asarray(Y[:, -1], dtype=float)
    order = np.argsort(y_body)
    return x_body[order], y_body[order]


def _domain_xlim(X: np.ndarray) -> tuple[float, float]:
    return (float(np.nanmin(X)), float(np.nanmax(X)))


def plot_baseflow(example_dir: Path, out_path: Path) -> None:
    plt = ensure_matplotlib()
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    py = load_python_npz(example_dir / PYTHON_NPZ)
    X = np.asarray(py["X"], dtype=float)
    Y = np.asarray(py["Y"], dtype=float)
    u = np.asarray(py["baseflow_u"], dtype=float)
    v = np.asarray(py["baseflow_v"], dtype=float)
    speed = np.sqrt(u**2 + v**2)
    x_body, y_body = _body_outline(X, Y)
    xlim = _domain_xlim(X)

    finite = speed[np.isfinite(speed)]
    v_lo = 0.0
    v_hi = float(np.max(finite))
    pad = 0.02 * max(v_hi - v_lo, 1.0)
    v_hi += pad

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    levels = np.linspace(v_lo, v_hi, 101)
    cf = ax.contourf(X, Y, speed, levels=levels, cmap="turbo", extend="neither")
    ax.plot(x_body, y_body, color="black", lw=1.4, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*BASEFLOW_YLIM)
    ax.set_aspect("auto")
    ax.set_xlabel(r"$x$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$y$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    cbar = fig.colorbar(cf, ax=ax, pad=0.015)
    cbar.locator = MaxNLocator(nbins=6)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_useOffset(False)
    cbar.formatter = formatter
    cbar.update_ticks()
    cbar.set_label(r"$|\mathbf{u}_b|$", fontsize=CBAR_FS)
    cbar.ax.tick_params(labelsize=CBAR_FS)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _component_pair(
    *,
    component: str,
    part: str,
    modes_mat: np.ndarray,
    modes_py: np.ndarray,
    mat_i: int,
    py_i: int,
    matlab: dict,
    X_py: np.ndarray,
    Y_py: np.ndarray,
    used_py: np.ndarray,
    idx_comp: np.ndarray,
    one_based_idx: bool,
) -> dict:
    X_mat, Y_mat, grid_mat = matlab_component_grid(
        modes_mat,
        mat_i,
        matlab,
        idx_prefix=IDX_PREFIX,
        plot_grid_prefix=PLOT_GRID_PREFIX,
        component=component,
        part=part,
    )
    py_scaled = scaled_python_mode(modes_py, modes_mat, py_i, mat_i)
    tmp = _mode_matrix(modes_py).copy()
    tmp[:, py_i] = py_scaled
    grid_py = python_component_grid(
        tmp,
        py_i,
        X=X_py,
        used_ind=used_py,
        idx_comp=idx_comp,
        one_based_idx=one_based_idx,
        part=part,
    )
    return {
        "X_mat": X_mat,
        "Y_mat": Y_mat,
        "matlab_grid": grid_mat,
        "X_py": X_py,
        "Y_py": Y_py,
        "python_grid": grid_py,
    }


def build_rows(
    example_dir: Path,
    component: str,
    part: str,
) -> tuple[list[dict], str, tuple[float, float]]:
    component = component.strip().lower()
    part = part.strip().lower()
    if part not in {"real", "imag"}:
        raise ValueError(f"Unsupported part {part!r} (use real or imag)")
    op = _part_tex(part)

    py = load_python_npz(example_dir / PYTHON_NPZ)
    matlab = load_matlab_bundle(example_dir, MATLAB_STEM)
    cfg = load_config(example_dir / PYTHON_YAML)

    gains_mat = np.asarray(_npz_get(matlab, "results_Paraboloid/gains"), dtype=float).reshape(-1)
    gains_py = np.asarray(py["gains"], dtype=float).reshape(-1)
    responses_mat = _mode_matrix(_npz_get(matlab, "results_Paraboloid/responses"))
    b_c_diag = ex5_observation_mask_diag(cfg, n_mode_dofs=responses_mat.shape[0])
    responses_mat = apply_diagonal_c(responses_mat, b_c_diag)
    forces_mat = _mode_matrix(_npz_get(matlab, "results_Paraboloid/forces"))
    b_c_diag_full = ex5_observation_mask_diag(cfg)
    forces_py = apply_diagonal_c(_mode_matrix(py["forcings"]), b_c_diag_full)
    responses_py = py["responses"]
    u_j = np.asarray(_npz_get(matlab, f"{IDX_PREFIX}/u_j"), dtype=int).reshape(-1)
    v_j = np.asarray(_npz_get(matlab, f"{IDX_PREFIX}/v_j"), dtype=int).reshape(-1)
    w_j = np.asarray(_npz_get(matlab, f"{IDX_PREFIX}/w_j"), dtype=int).reshape(-1)
    forces_py = maybe_compact_python_resolvent_uvw(
        forces_py,
        forces_mat,
        u_j=u_j,
        v_j=v_j,
        w_j=w_j,
    )
    responses_py = maybe_compact_python_resolvent_uvw(
        responses_py,
        responses_mat,
        u_j=u_j,
        v_j=v_j,
        w_j=w_j,
    )

    mode_pairs = match_modes_by_gains(gains_mat, gains_py, N_MODES)
    if len(mode_pairs) < N_MODES:
        raise RuntimeError(f"Only matched {len(mode_pairs)}/{N_MODES} modes")

    X_py = np.asarray(py["X"], dtype=float)
    Y_py = np.asarray(py["Y"], dtype=float)
    used_py = (
        np.asarray(py["used_ind"], dtype=int)
        if "used_ind" in py
        else np.arange(int(np.prod(X_py.shape)), dtype=int)
    )
    n_gp = int(u_j.size)
    if forces_mat.shape[0] == 3 * n_gp:
        matlab_plot = patch_matlab_compact_uvw_idx(
            matlab, idx_prefix=IDX_PREFIX, n_gp=n_gp
        )
        idx_comp = compact_uvw_component_idx(n_gp, component, one_based=True)
        one_based = True
    else:
        matlab_plot = matlab
        idx_key = f"idx_{component}_j"
        idx_comp = (
            py[idx_key]
            if idx_key in py
            else _npz_get(matlab, f"{IDX_PREFIX}/{component}_j")
        )
        one_based = idx_key not in py
    xlim = _domain_xlim(X_py)

    rows: list[dict] = []
    for rank, (mat_i, py_i) in enumerate(mode_pairs, start=1):
        force = _component_pair(
            component=component,
            part=part,
            modes_mat=forces_mat,
            modes_py=forces_py,
            mat_i=mat_i,
            py_i=py_i,
            matlab=matlab_plot,
            X_py=X_py,
            Y_py=Y_py,
            used_py=used_py,
            idx_comp=idx_comp,
            one_based_idx=one_based,
        )
        force["row_label"] = rf"${op}(f_{{{component}}}^{{({rank})}})$"
        force["ref_gain"] = float(gains_mat[mat_i])
        force["py_gain"] = float(gains_py[py_i])
        rows.append(force)

        resp = _component_pair(
            component=component,
            part=part,
            modes_mat=responses_mat,
            modes_py=responses_py,
            mat_i=mat_i,
            py_i=py_i,
            matlab=matlab_plot,
            X_py=X_py,
            Y_py=Y_py,
            used_py=used_py,
            idx_comp=idx_comp,
            one_based_idx=one_based,
        )
        resp["row_label"] = rf"${op}({component}^{{({rank})}})$"
        rows.append(resp)

    return rows, "", xlim


def plot_example5_figures(
    example_dir: Path,
    *,
    components: tuple[str, ...] = DEFAULT_COMPONENTS,
    parts: tuple[str, ...] = DEFAULT_PARTS,
) -> list[Path]:
    """Write Example 5 campaign figures next to the driver."""
    example_dir = example_dir.resolve()
    written: list[Path] = []

    baseflow_path = example_dir / OUT_BASEFLOW
    plot_baseflow(example_dir, baseflow_path)
    written.append(baseflow_path)

    for component in components:
        for part in parts:
            rows, title, xlim = build_rows(example_dir, component, part)
            out_path = example_dir / mosaic_name(component, part)
            plot_mode_triptych_mosaic(
                out_path=out_path,
                title=title,
                rows=rows,
                xlim=xlim,
                ylim=MOSAIC_YLIM,
                dpi=DPI,
                row_height=MOSAIC_ROW_HEIGHT,
            )
            written.append(out_path)

    return written


def main() -> None:
    """Load the Example 5 NPZ and write the presentation PNGs."""
    example_dir = Path(__file__).resolve().parent
    for path in plot_example5_figures(example_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
