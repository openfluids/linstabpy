# SPDX-License-Identifier: GPL-3.0-only
"""Presentation figures for Martini Example 2 (jet resolvent).

Uses published ``kron``/``repeat`` :math:`B=C` parity
(MATLAB ``kron(B(:), ones(5,1))``).

- ``example_2_jet_baseflow.png`` — jet axial velocity :math:`u`.
- ``example_2_jet_modes_forcing_response_{u,v}[_imag]_mosaic.png`` — LinStab2D | linstabpy | error.

MATLAB responses are pre-:math:`C`; mosaics apply the same diagonal :math:`C=B`
before error panels. Forcing rows are left as exported (:math:`B v`).
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
    jet_resolvent_response_c_diag,
    jet_sponge_zero_mask,
    match_modes_by_gains,
    matlab_component_grid,
    maybe_compact_python_resolvent_uv,
    patch_matlab_compact_uv_idx,
    plot_mode_triptych_mosaic,
    python_component_grid,
    scaled_python_mode,
)

MATLAB_EX2_STEM = "example_2_Jet_results"
IDX_PREFIX = "results_Jet/idx"
PLOT_GRID_PREFIX = "results_Jet/plot_grid"

FIGURE_PREFIX = "example_2_jet"
OUT_BASEFLOW = f"{FIGURE_PREFIX}_baseflow.png"
DEFAULT_COMPONENTS = ("u", "v")
DEFAULT_PARTS = ("real", "imag")
N_MODES = 3
BC_MASK_LAYOUT = "kron"

BASEFLOW_XLIM = (0.0, 30.0)
BASEFLOW_YLIM = (0.0, 3.0)
MOSAIC_XLIM = (0.0, 20.0)
MOSAIC_YLIM = (0.0, 1.05)
MOSAIC_ROW_HEIGHT = 1.55

LABEL_FS = 24
TICK_FS = 20
CBAR_FS = 18
DPI = 400


def linstab2d_results_dir(example_dir: Path) -> Path:
    return example_dir / LINSTAB2D_RESULTS_DIRNAME


def load_matlab_ex2_fixture(example_dir: Path) -> dict[str, np.ndarray]:
    return load_matlab_bundle(example_dir, MATLAB_EX2_STEM)


def _matlab_gains(matlab: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("results_Jet/gains", "results_Jet/S"):
        try:
            return np.asarray(_npz_get(matlab, key), dtype=float).reshape(-1)
        except KeyError:
            continue
    raise KeyError("Missing MATLAB jet gains (results_Jet/gains or results_Jet/S)")


def _part_tex(part: str) -> str:
    if part == "imag":
        return r"\Im"
    return r"\Re"


def _canonical_component(raw: str) -> str:
    key = raw.strip().lower()
    if key in {"u", "v"}:
        return key
    raise ValueError(f"Unsupported component {raw!r} (use u or v)")


def mosaic_name(component: str, part: str) -> str:
    component = _canonical_component(component)
    if part == "real":
        return f"{FIGURE_PREFIX}_modes_forcing_response_{component}_mosaic.png"
    if part == "imag":
        return f"{FIGURE_PREFIX}_modes_forcing_response_{component}_imag_mosaic.png"
    return f"{FIGURE_PREFIX}_modes_forcing_response_{component}_{part}_mosaic.png"


def plot_baseflow(data: dict[str, np.ndarray], out_path: Path) -> None:
    """Jet axial velocity with campaign typography."""
    plt = ensure_matplotlib()
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    if "baseflow_u" not in data:
        raise KeyError(f"Missing baseflow_u in cached NPZ for {out_path}")
    u = np.asarray(data["baseflow_u"], dtype=float)

    finite = u[np.isfinite(u)]
    v_lo = float(np.min(finite))
    v_hi = float(np.max(finite))
    pad = 0.02 * max(v_hi - v_lo, 1.0)
    v_lo -= pad
    v_hi += pad

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    levels = np.linspace(v_lo, v_hi, 101)
    cf = ax.contourf(X, Y, u, levels=levels, cmap="turbo", extend="neither")
    ax.set_xlim(*BASEFLOW_XLIM)
    ax.set_ylim(*BASEFLOW_YLIM)
    ax.set_aspect("auto")
    ax.set_xlabel(r"$x$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$r$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    cbar = fig.colorbar(cf, ax=ax, pad=0.015)
    cbar.locator = MaxNLocator(nbins=6)
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
    component: str,
    part: str,
    bc_mask_layout: str = BC_MASK_LAYOUT,
) -> list[dict[str, Any]]:
    component = _canonical_component(component)
    part = part.strip().lower()
    if part not in {"real", "imag"}:
        raise ValueError(f"Unsupported part {part!r} (use real or imag)")
    op = _part_tex(part)
    if component not in DEFAULT_COMPONENTS:
        raise ValueError(
            f"Unsupported component {component!r} for Ex2 presentation "
            "(only u,v are forced; rho/w/T are masked)"
        )

    gains_mat = _matlab_gains(matlab)
    gains_py = np.asarray(python_data["gains"], dtype=float).reshape(-1)
    responses_mat = _mode_matrix(_npz_get(matlab, "results_Jet/responses"))
    forces_mat = _mode_matrix(_npz_get(matlab, "results_Jet/forces"))
    responses_py = _mode_matrix(python_data["responses"])
    forces_py = _mode_matrix(python_data["forcings"])
    u_j = np.asarray(_npz_get(matlab, f"{IDX_PREFIX}/u_j"), dtype=int).reshape(-1)
    v_j = np.asarray(_npz_get(matlab, f"{IDX_PREFIX}/v_j"), dtype=int).reshape(-1)
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

    X_py = np.asarray(python_data["X"], dtype=float)
    Y_py = np.asarray(python_data["Y"], dtype=float)
    used_py = (
        np.asarray(python_data["used_ind"], dtype=int)
        if "used_ind" in python_data
        else np.arange(int(np.prod(X_py.shape)), dtype=int)
    )

    sponge0 = jet_sponge_zero_mask(X_py, Y_py, used_ind=used_py)
    mode_dofs = int(responses_mat.shape[0])
    c_diag = jet_resolvent_response_c_diag(
        sponge0,
        mode_dofs=mode_dofs,
        u_j=u_j,
        v_j=v_j,
        layout=bc_mask_layout,
    )
    responses_mat = apply_diagonal_c(responses_mat, c_diag)

    mode_pairs = match_modes_by_gains(gains_mat, gains_py, N_MODES)
    if len(mode_pairs) < N_MODES:
        raise RuntimeError(f"Only matched {len(mode_pairs)}/{N_MODES} jet modes")

    n_gp_idx = int(u_j.size)
    if forces_mat.shape[0] == 2 * n_gp_idx:
        matlab_plot = patch_matlab_compact_uv_idx(matlab, idx_prefix=IDX_PREFIX, n_gp=n_gp_idx)
        idx_comp = compact_uv_component_idx(n_gp_idx, component, one_based=True)
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
    component: str,
    part: str,
    out_path: Path,
    bc_mask_layout: str = BC_MASK_LAYOUT,
) -> None:
    rows = build_mosaic_rows(
        python_data=python_data,
        matlab=matlab,
        component=component,
        part=part,
        bc_mask_layout=bc_mask_layout,
    )
    plot_mode_triptych_mosaic(
        out_path=out_path,
        title="",
        rows=rows,
        xlim=MOSAIC_XLIM,
        ylim=MOSAIC_YLIM,
        dpi=DPI,
        row_height=MOSAIC_ROW_HEIGHT,
    )


def plot_example2_figures(
    data: dict[str, np.ndarray],
    *,
    example_dir: Path,
    bc_mask_layout: str = BC_MASK_LAYOUT,
) -> list[Path]:
    """Write all Example 2 presentation PNGs next to the driver script."""
    matlab = load_matlab_ex2_fixture(example_dir)
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
                component=component,
                part=part,
                out_path=out_path,
                bc_mask_layout=bc_mask_layout,
            )
            outputs.append(out_path)

    return outputs


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def main() -> None:
    """Load the Example 2 NPZ and write the presentation PNGs."""
    example_dir = Path(__file__).resolve().parent
    data = _load_npz(example_dir / "example_2_jet.npz")
    bc_layout = BC_MASK_LAYOUT
    if "bc_mask_layout" in data:
        bc_layout = str(np.asarray(data["bc_mask_layout"]).reshape(-1)[0])
    for path in plot_example2_figures(data, example_dir=example_dir, bc_mask_layout=bc_layout):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
