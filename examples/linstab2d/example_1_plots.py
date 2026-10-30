"""Presentation figures for Martini Example 1 (Blasius resolvent).

Figures:

- ``example_1_blasius_baseflow.png`` — Blasius baseflow :math:`u/U_\\infty`.
- ``example_1_blasius_modes_forcing_response_{u,v}_mosaic.png`` — LinStab2D | linstabpy | error (Re).
- ``example_1_blasius_modes_forcing_response_{u,v}_imag_mosaic.png`` — same for Im.

Only :math:`u` and :math:`v` receive forcing and observation
(:math:`\\rho,w,T` masked in :math:`B=C`).
MATLAB responses are pre-:math:`C`.
Mosaics apply the same diagonal :math:`C=B` before plotting responses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.linstab2d.compare_martini_examples import (
    LINSTAB2D_RESULTS_DIRNAME,
    _mode_matrix,
    _npz_get,
    load_matlab_bundle,
)
from examples.linstab2d.plot_compare_utils import (
    apply_diagonal_c,
    compact_uv_component_idx,
    ensure_matplotlib,
    match_modes_by_gains,
    matlab_component_grid,
    maybe_compact_python_resolvent_uv,
    patch_matlab_compact_uv_idx,
    plot_mode_triptych_mosaic,
    python_component_grid,
    scaled_python_mode,
)
from linstabpy.utils.yaml_io import load_config

MATLAB_EX1_STEM = "example_1_Blasius_results"
IDX_PREFIX = "results_Blasius/idx"
PLOT_GRID_PREFIX = "results_Blasius/plot_grid"

FIGURE_PREFIX = "example_1_blasius"
OUT_BASEFLOW = f"{FIGURE_PREFIX}_baseflow.png"

DEFAULT_COMPONENTS = ("u", "v")
DEFAULT_PARTS = ("real", "imag")
N_MODES = 3
YLIM_BY_COMPONENT = {
    "u": (0.0, 0.01),
    "v": (0.0, 0.01),
}
DEFAULT_YLIM = (0.0, 0.01)
BASEFLOW_YLIM = (0.0, 0.015)

TITLE_FS = 26
LABEL_FS = 24
TICK_FS = 20
CBAR_FS = 18
ANNOT_FS = 20
DPI = 400


def linstab2d_results_dir(example_dir: Path) -> Path:
    return example_dir / LINSTAB2D_RESULTS_DIRNAME


def load_matlab_ex1_fixture(example_dir: Path) -> dict[str, np.ndarray]:
    return load_matlab_bundle(example_dir, MATLAB_EX1_STEM)


def _ex1_lnse_system(cfg: dict[str, Any]):
    """Build the Blasius LNSE system used for Ex.~1 observation masks."""
    from examples.linstab2d.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system_result

    solver_cfg = cfg.get("solver")
    dx = cfg["dx"]
    xrange = (cfg["x_min_factor"] * dx, cfg["x_max_base"] + dx)
    cmesh = Mesh.create(
        xrange,
        (-1.0, 1.0),
        cfg["nx"],
        cfg["ny"],
        cfg["fd_order"],
        alpha_filter=cfg["alpha_filter"],
    )
    Y = cmesh.Y.copy()
    y_i, y_max = cfg["y_i"], cfg["y_max"]
    a = y_i * y_max / (y_max - 2 * y_i)
    b = 1 + 2 * a / y_max
    Y_phys = a * (1 + Y) / (b - Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    bf = load_blasius(
        mesh.X,
        Y_phys,
        Re=cfg["Re"],
        Ma=cfg["Ma"],
        Pr=cfg["Pr"],
        kappa=cfg["kappa"],
        T_0=cfg["T_0"],
    )
    sutherland_air(bf)
    if cfg.get("zero_dmudT_martini_parity", False):
        bf.dmudT[:] = 0.0
        bf.d2mudT2[:] = 0.0

    x, y = mesh.X, mesh.Y
    sc = cfg["sponge"]
    mesh.sponge = (
        sc["amplitude"]
        / 2
        * np.maximum(
            np.tanh((y - y.max() + sc["ys_offset"]) / sc["ys_trans"]) + 1,
            np.tanh(-(x - x.min() - sc["xs_offset"]) / sc["xs_trans"])
            + 1
            + np.tanh((x - x.max() + sc["xs_offset"]) / sc["xs_trans"])
            + 1,
        )
    )

    return build_lnse_system_result(
        mesh,
        bf,
        kx=cfg["kx"],
        model=cfg.get("model", "2D"),
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=cfg["bc"]["penalty"],
        backend=cfg.get("assembly_backend"),
        solver_cfg=solver_cfg,
    )


def ex1_uv_indices(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Global u/v DOF indices for compacting full-stack Python resolvent modes."""
    idx = _ex1_lnse_system(cfg).idx
    return idx["u_j"], idx["v_j"]


def ex1_observation_mask_diag(
    cfg: dict[str, Any],
    *,
    n_mode_dofs: int | None = None,
) -> np.ndarray:
    """Block-correct :math:`B=C` diagonal for Ex.~1 (zeros rho,w,T and Dirichlet DOFs)."""
    lnse_system = _ex1_lnse_system(cfg)
    idx = lnse_system.idx
    bc_indices = lnse_system.bc_indices

    var_map = {"rho": "rho_j", "u": "u_j", "v": "v_j", "w": "w_j", "T": "T_j"}
    b_diag = np.ones(int(idx["u_j"].size) * 5)
    for var_name in cfg.get("forcing_zero_vars", []):
        b_diag[idx[var_map[var_name]]] = 0.0
    b_diag[bc_indices] = 0.0

    if n_mode_dofs is None or n_mode_dofs == b_diag.size:
        return b_diag
    uv_dofs = int(idx["u_j"].size + idx["v_j"].size)
    if n_mode_dofs == uv_dofs:
        return np.concatenate([b_diag[idx["u_j"]], b_diag[idx["v_j"]]])
    raise ValueError(
        f"Ex1 observation mask length {b_diag.size} (full) / {uv_dofs} (u+v) "
        f"does not match mode DOFs {n_mode_dofs}"
    )


def _part_tex(part: str) -> str:
    if part == "imag":
        return r"\Im"
    return r"\Re"


def mosaic_name(component: str, part: str) -> str:
    if part == "real":
        return f"{FIGURE_PREFIX}_modes_forcing_response_{component}_mosaic.png"
    if part == "imag":
        return f"{FIGURE_PREFIX}_modes_forcing_response_{component}_imag_mosaic.png"
    return f"{FIGURE_PREFIX}_modes_forcing_response_{component}_{part}_mosaic.png"


def plot_baseflow(
    data: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Blasius :math:`u/U_\\infty` with campaign typography."""
    plt = ensure_matplotlib()
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    if "baseflow_u" not in data:
        raise KeyError(f"Missing baseflow_u in cached NPZ for {out_path}")
    u = np.asarray(data["baseflow_u"], dtype=float)

    fig, ax = plt.subplots(figsize=(12.0, 3.4))
    levels = np.linspace(0.0, 1.0, 101)
    cf = ax.contourf(X, Y, u, levels=levels, cmap="turbo", extend="neither")
    ax.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
    ax.set_ylim(BASEFLOW_YLIM)
    ax.set_aspect("auto")
    ax.set_xlabel(r"$x$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$y$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.annotate(
        r"$U_\infty$",
        xy=(0.78, 0.82),
        xytext=(0.58, 0.82),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "black"},
        fontsize=ANNOT_FS,
        va="center",
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.015)
    cbar.set_ticks(np.linspace(0.0, 1.0, 5))
    cbar.locator = MaxNLocator(nbins=5)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_useOffset(False)
    cbar.formatter = formatter
    cbar.update_ticks()
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
    matlab: dict[str, np.ndarray],
    X_py: np.ndarray,
    Y_py: np.ndarray,
    used_py: np.ndarray,
    idx_comp: np.ndarray,
    one_based_idx: bool,
) -> dict[str, Any]:
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


def build_mosaic_rows(
    *,
    python_data: dict[str, np.ndarray],
    matlab: dict[str, np.ndarray],
    cfg: dict[str, Any],
    component: str,
    part: str,
) -> list[dict[str, Any]]:
    component = component.strip().lower()
    if component not in {"u", "v"}:
        raise ValueError(
            f"Unsupported component {component!r} for Ex1 presentation "
            "(only u,v are forced; rho/w/T are masked)"
        )
    part = part.strip().lower()
    if part not in {"real", "imag"}:
        raise ValueError(f"Unsupported part {part!r} (use real or imag)")
    op = _part_tex(part)

    gains_mat = np.asarray(_npz_get(matlab, "results_Blasius/gains"), dtype=float).reshape(-1)
    gains_py = np.asarray(python_data["gains"], dtype=float).reshape(-1)
    responses_mat = _mode_matrix(_npz_get(matlab, "results_Blasius/responses"))
    responses_mat = apply_diagonal_c(
        responses_mat,
        ex1_observation_mask_diag(cfg, n_mode_dofs=responses_mat.shape[0]),
    )
    forces_mat = _mode_matrix(_npz_get(matlab, "results_Blasius/forces"))
    responses_py = python_data["responses"]
    forces_py = python_data["forcings"]
    u_j = np.asarray(_npz_get(matlab, f"{IDX_PREFIX}/u_j"), dtype=int).reshape(-1)
    v_j = np.asarray(_npz_get(matlab, f"{IDX_PREFIX}/v_j"), dtype=int).reshape(-1)
    n_gp = int(u_j.size)
    forces_py = maybe_compact_python_resolvent_uv(
        forces_py,
        forces_mat,
        u_j=u_j,
        v_j=v_j,
    )
    responses_py = maybe_compact_python_resolvent_uv(
        responses_py,
        responses_mat,
        u_j=u_j,
        v_j=v_j,
    )

    mode_pairs = match_modes_by_gains(gains_mat, gains_py, N_MODES)
    X_py = np.asarray(python_data["X"], dtype=float)
    Y_py = np.asarray(python_data["Y"], dtype=float)
    used_py = (
        np.asarray(python_data["used_ind"], dtype=int)
        if "used_ind" in python_data
        else np.arange(int(np.prod(X_py.shape)), dtype=int)
    )
    if forces_mat.shape[0] == 2 * n_gp:
        matlab_plot = patch_matlab_compact_uv_idx(matlab, idx_prefix=IDX_PREFIX, n_gp=n_gp)
        idx_comp = compact_uv_component_idx(n_gp, component, one_based=True)
        one_based = True
    else:
        matlab_plot = matlab
        idx_key = f"idx_{component}_j"
        idx_comp = (
            python_data[idx_key]
            if idx_key in python_data
            else _npz_get(matlab, f"{IDX_PREFIX}/{component}_j")
        )
        one_based = idx_key not in python_data

    rows: list[dict[str, Any]] = []
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

    return rows


def plot_mosaic(
    *,
    python_data: dict[str, np.ndarray],
    matlab: dict[str, np.ndarray],
    cfg: dict[str, Any],
    component: str,
    part: str,
    out_path: Path,
) -> None:
    rows = build_mosaic_rows(
        python_data=python_data,
        matlab=matlab,
        cfg=cfg,
        component=component,
        part=part,
    )
    ylim = YLIM_BY_COMPONENT.get(component, DEFAULT_YLIM)
    plot_mode_triptych_mosaic(
        out_path=out_path,
        title="",
        rows=rows,
        ylim=ylim,
        dpi=DPI,
    )


def plot_example1_figures(
    data: dict[str, np.ndarray],
    *,
    example_dir: Path,
    cfg: dict | None = None,
) -> list[Path]:
    """Write all Example 1 presentation PNGs next to the driver script."""
    if cfg is None:
        cfg = load_config(example_dir / "example_1_blasius.yaml")

    matlab = load_matlab_ex1_fixture(example_dir)
    outputs: list[Path] = []

    baseflow_path = example_dir / OUT_BASEFLOW
    plot_baseflow(data, baseflow_path)
    outputs.append(baseflow_path)

    for component in DEFAULT_COMPONENTS:
        for part in DEFAULT_PARTS:
            out_path = example_dir / mosaic_name(component, part)
            plot_mosaic(
                python_data=data,
                matlab=matlab,
                cfg=cfg,
                component=component,
                part=part,
                out_path=out_path,
            )
            outputs.append(out_path)

    return outputs


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def main() -> None:
    """Load the Example 1 NPZ and write the presentation PNGs."""
    example_dir = Path(__file__).resolve().parent
    data = _load_npz(example_dir / "example_1_blasius.npz")
    for path in plot_example1_figures(data, example_dir=example_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
