"""Presentation figures for Martini Example 4 (cavity direct / adjoint).

Matches the thesis/campaign layout under
``cases/Martini2024/comparison/plots/example_4_cavity/``.

- ``example_4_cavity_baseflow.png`` — cavity baseflow :math:`u/U_\\infty`.
- ``example_4_cavity_spectrum_direct.png`` / ``example_4_cavity_spectrum_adjoint.png`` — LinStab2D vs linstabpy.
- ``example_4_cavity_modes_{direct,adjoint}_{u,v,w}[_imag]_mosaic.png`` — ranks 1–3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.martini.compare_martini_examples import (
    LINSTAB2D_RESULTS_DIRNAME,
    _best_complex_scale,
    _best_spectrum_pairs,
    _mode_matrix,
    _npz_get,
    load_matlab_bundle,
)
from examples.martini.plot_compare_utils import (
    compact_resolvent_modes_uvw,
    complex_vector,
    ensure_matplotlib,
    load_python_npz,
    matlab_component_grid,
    plot_mode_triptych_mosaic,
    plot_spectrum_overlay,
    python_component_grid,
)

PYTHON_NPZ = "example_4_cavity.npz"
MATLAB_DIRECT_STEM = "example_4_Cavity_results_direct"
MATLAB_ADJOINT_STEM = "example_4_Cavity_results_adjoint"
IDX_PREFIX = "results_Cavity_direct/idx"
PLOT_GRID_PREFIX = "results_Cavity_direct/plot_grid"
SAVED_MODE_INDICES_DIRECT = "results_Cavity_direct/saved_mode_indices"
SAVED_MODE_INDICES_ADJOINT = "results_Cavity_adjoint/saved_mode_indices"

FIGURE_PREFIX = "example_4_cavity"
OUT_BASEFLOW = f"{FIGURE_PREFIX}_baseflow.png"
OUT_SPEC_DIRECT = f"{FIGURE_PREFIX}_spectrum_direct.png"
OUT_SPEC_ADJOINT = f"{FIGURE_PREFIX}_spectrum_adjoint.png"

N_MODES = 3
DEFAULT_COMPONENTS = ("u", "v", "w")
DEFAULT_PARTS = ("real", "imag")
_COMPONENT_TEX = {
    "u": "u",
    "v": "v",
    "w": "w",
}
PLOT_XLIM = (-1.65, 5.34)
PLOT_YLIM = (-1.0, 0.6)
MOSAIC_YLIM = (-1.0, 0.2)
MOSAIC_ROW_HEIGHT = 1.55

TITLE_FS = 26
LABEL_FS = 24
TICK_FS = 20
CBAR_FS = 18
DPI = 400
SPEC_MARKER = {"marker_size": 10.0, "marker_edgewidth": 3.0, "dpi": 300}


def linstab2d_results_dir(example_dir: Path) -> Path:
    return example_dir / LINSTAB2D_RESULTS_DIRNAME


def load_matlab_ex4_fixture(example_dir: Path, stem: str) -> dict[str, np.ndarray]:
    return load_matlab_bundle(example_dir, stem)


def mosaic_name(tag: str, component: str, part: str) -> str:
    if part == "real":
        return f"{FIGURE_PREFIX}_modes_{tag}_{component}_mosaic.png"
    return f"{FIGURE_PREFIX}_modes_{tag}_{component}_{part}_mosaic.png"


def _part_tex(part: str) -> str:
    if part == "imag":
        return r"\Im"
    return r"\Re"


def _component_tex(component: str) -> str:
    return _COMPONENT_TEX.get(component.strip().lower(), component)


def _cavity_baseflow_plot_field(X: np.ndarray, Y: np.ndarray, field: np.ndarray) -> np.ndarray:
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
    cavity_left = 0.34029
    cavity_right = 2.34
    y_floor = -1.0
    ax.plot([cavity_left, cavity_left], [0.0, y_floor], color="black", lw=1.1)
    ax.plot([cavity_left, cavity_right], [y_floor, y_floor], color="black", lw=1.1)
    ax.plot([cavity_right, cavity_right], [y_floor, 0.0], color="black", lw=1.1)


def plot_baseflow(example_dir: Path, out_path: Path) -> None:
    plt = ensure_matplotlib()
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    py = load_python_npz(example_dir / PYTHON_NPZ)
    X = np.asarray(py["X"], dtype=float)
    Y = np.asarray(py["Y"], dtype=float)
    u = _cavity_baseflow_plot_field(X, Y, np.asarray(py["baseflow_u"], dtype=float))
    finite = u[np.isfinite(u)]
    u_min = float(np.min(finite))
    u_max = float(np.max(finite))
    v_lo = min(u_min, 0.0)
    v_hi = max(u_max, 1.0)
    pad = 0.02 * max(v_hi - v_lo, 1.0)

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    ax.set_facecolor("#8e8e8e")
    cf = ax.contourf(X, Y, u, levels=np.linspace(v_lo - pad, v_hi + pad, 101), cmap="turbo")
    _draw_cavity_outline(ax)
    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_aspect("auto")
    ax.set_xlabel(r"$x$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$y$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.annotate(
        r"$U_\infty$",
        xy=(0.82, 0.78),
        xytext=(0.62, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "black"},
        fontsize=20,
        va="center",
    )
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
    print(f"Wrote {out_path} (u in [{u_min:.3f}, {u_max:.3f}])")


def _adjoint_omega_direct_convention_python(omega_adj: np.ndarray) -> np.ndarray:
    return np.conjugate(-1j * complex_vector(omega_adj))


def _adjoint_omega_direct_convention_matlab(omega_adj: np.ndarray) -> np.ndarray:
    return np.conjugate(complex_vector(omega_adj))


def _matlab_uvw_idx_labels(py: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global u/v/w DOF labels for compact compare (LinStab2D = linstabpy + 1)."""
    return (
        np.asarray(py["idx_u_j"], dtype=int) + 1,
        np.asarray(py["idx_v_j"], dtype=int) + 1,
        np.asarray(py["idx_w_j"], dtype=int) + 1,
    )


def _scale_python_mode_for_slim_matlab(
    modes_py: np.ndarray,
    modes_mat: np.ndarray,
    *,
    py_index: int,
    mat_col: int,
    u_j: np.ndarray,
    v_j: np.ndarray,
    w_j: np.ndarray,
) -> np.ndarray:
    """Match slim MATLAB u+v+w columns; grid the scaled full-stack Python column."""
    py_mat = _mode_matrix(modes_py)
    ref = _mode_matrix(modes_mat)[:, mat_col]
    cand = compact_resolvent_modes_uvw(py_mat, u_j, v_j, w_j)[:, py_index]
    return py_mat[:, py_index] * _best_complex_scale(ref, cand)


def _matlab_mode_pairs(
    matlab: dict[str, np.ndarray],
    *,
    saved_key: str,
    omega_mat: np.ndarray,
    omega_py: np.ndarray,
    n_modes: int,
) -> list[tuple[int, int]]:
    saved = np.asarray(_npz_get(matlab, saved_key), dtype=int).reshape(-1)
    mat_eig = saved - 1 if int(saved.min()) >= 1 else saved
    if mat_eig.size < n_modes:
        raise RuntimeError(f"Expected {n_modes} saved MATLAB modes, got {mat_eig.size}")

    pair_map = {
        int(mat_i): int(py_i)
        for mat_i, py_i, _, _ in _best_spectrum_pairs(
            omega_mat,
            omega_py,
            max(int(omega_mat.size), int(omega_py.size)),
        )
    }
    pairs: list[tuple[int, int]] = []
    for col, eig_i in enumerate(mat_eig[:n_modes]):
        eig_i = int(eig_i)
        if eig_i not in pair_map:
            raise RuntimeError(f"No Python pair for saved MATLAB mode index {eig_i + 1}")
        pairs.append((col, pair_map[eig_i]))
    return pairs


def _component_row(
    *,
    component: str,
    part: str,
    rank: int,
    modes_mat: np.ndarray,
    modes_py: np.ndarray,
    mat_col: int,
    py_i: int,
    matlab_grid_bundle: dict,
    X_py: np.ndarray,
    Y_py: np.ndarray,
    used_py: np.ndarray,
    idx_comp: np.ndarray,
    one_based_idx: bool,
    u_j: np.ndarray,
    v_j: np.ndarray,
    w_j: np.ndarray,
) -> dict:
    op = _part_tex(part)
    X_mat, Y_mat, grid_mat = matlab_component_grid(
        modes_mat,
        mat_col,
        matlab_grid_bundle,
        idx_prefix=IDX_PREFIX,
        plot_grid_prefix=PLOT_GRID_PREFIX,
        component=component,
        part=part,
    )
    py_scaled = _scale_python_mode_for_slim_matlab(
        modes_py,
        modes_mat,
        py_index=py_i,
        mat_col=mat_col,
        u_j=u_j,
        v_j=v_j,
        w_j=w_j,
    )
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
    var = _component_tex(component)
    return {
        "row_label": rf"${op}({var}^{{({rank})}})$",
        "X_mat": X_mat,
        "Y_mat": Y_mat,
        "matlab_grid": grid_mat,
        "X_py": X_py,
        "Y_py": Y_py,
        "python_grid": grid_py,
        "ref_gain": None,
        "py_gain": None,
    }


def build_branch_rows(
    example_dir: Path,
    tag: str,
    component: str,
    part: str,
    n_modes: int = N_MODES,
) -> list[dict]:
    component = component if component != "t" else "T"
    part = part if part in {"real", "imag"} else "real"

    py = load_python_npz(example_dir / PYTHON_NPZ)
    matlab_d = load_matlab_ex4_fixture(example_dir, MATLAB_DIRECT_STEM)
    matlab_a = load_matlab_ex4_fixture(example_dir, MATLAB_ADJOINT_STEM)

    if tag == "direct":
        omega_mat = complex_vector(_npz_get(matlab_d, "results_Cavity_direct/omega"))
        omega_py = complex_vector(py["omega"])
        modes_mat = _npz_get(matlab_d, "results_Cavity_direct/V")
        modes_py = py["direct_modes"]
        grid_bundle = matlab_d
        saved_key = SAVED_MODE_INDICES_DIRECT
    elif tag == "adjoint":
        omega_mat = _adjoint_omega_direct_convention_matlab(
            _npz_get(matlab_a, "results_Cavity_adjoint/omegaAdj")
        )
        omega_py = _adjoint_omega_direct_convention_python(py["omega_adj"])
        modes_mat = _npz_get(matlab_a, "results_Cavity_adjoint/VAdj")
        modes_py = py["adjoint_modes"]
        grid_bundle = matlab_d
        saved_key = SAVED_MODE_INDICES_ADJOINT
    else:
        raise ValueError(f"Unknown branch tag {tag!r}")

    mode_pairs = _matlab_mode_pairs(
        matlab_d if tag == "direct" else matlab_a,
        saved_key=saved_key,
        omega_mat=omega_mat,
        omega_py=omega_py,
        n_modes=n_modes,
    )
    if len(mode_pairs) < n_modes:
        raise RuntimeError(f"Only matched {len(mode_pairs)}/{n_modes} {tag} modes")

    X_py = np.asarray(py["X"], dtype=float)
    Y_py = np.asarray(py["Y"], dtype=float)
    used_py = np.asarray(py["used_ind"], dtype=int)
    idx_key = f"idx_{component}_j"
    idx_comp = (
        py[idx_key] if idx_key in py else _npz_get(grid_bundle, f"{IDX_PREFIX}/{component}_j")
    )
    one_based = idx_key in py
    u_j, v_j, w_j = _matlab_uvw_idx_labels(py)

    rows: list[dict] = []
    for rank, (mat_col, py_i) in enumerate(mode_pairs, start=1):
        rows.append(
            _component_row(
                component=component,
                part=part,
                rank=rank,
                modes_mat=modes_mat,
                modes_py=modes_py,
                mat_col=mat_col,
                py_i=py_i,
                matlab_grid_bundle=grid_bundle,
                X_py=X_py,
                Y_py=Y_py,
                used_py=used_py,
                idx_comp=idx_comp,
                one_based_idx=one_based,
                u_j=u_j,
                v_j=v_j,
                w_j=w_j,
            )
        )
    return rows


def plot_spectra(example_dir: Path) -> list[Path]:
    py = load_python_npz(example_dir / PYTHON_NPZ)
    matlab_d = load_matlab_ex4_fixture(example_dir, MATLAB_DIRECT_STEM)
    matlab_a = load_matlab_ex4_fixture(example_dir, MATLAB_ADJOINT_STEM)
    style = SPEC_MARKER

    out_direct = example_dir / OUT_SPEC_DIRECT
    plot_spectrum_overlay(
        out_path=out_direct,
        title="",
        matlab_eigs=complex_vector(_npz_get(matlab_d, "results_Cavity_direct/omega")),
        python_eigs=complex_vector(py["omega"]),
        xlabel=r"$\Re(\omega)$",
        ylabel=r"$\Im(\omega)$",
        n_annotate=6,
        **style,
    )
    print(f"Wrote {out_direct}")

    out_adj = example_dir / OUT_SPEC_ADJOINT
    plot_spectrum_overlay(
        out_path=out_adj,
        title="",
        matlab_eigs=_adjoint_omega_direct_convention_matlab(
            _npz_get(matlab_a, "results_Cavity_adjoint/omegaAdj")
        ),
        python_eigs=_adjoint_omega_direct_convention_python(py["omega_adj"]),
        xlabel=r"$\Re(\omega)$",
        ylabel=r"$\Im(\omega)$",
        n_annotate=6,
        **style,
    )
    print(f"Wrote {out_adj}")
    return [out_direct, out_adj]


def plot_example4_figures(
    example_dir: Path,
    *,
    components: tuple[str, ...] = DEFAULT_COMPONENTS,
    parts: tuple[str, ...] = DEFAULT_PARTS,
    n_modes: int = N_MODES,
    skip_baseflow: bool = False,
    skip_spectra: bool = False,
    skip_modes: bool = False,
) -> list[Path]:
    """Write all Example 4 presentation PNGs next to the driver script."""
    outputs: list[Path] = []

    if not skip_baseflow:
        baseflow_path = example_dir / OUT_BASEFLOW
        plot_baseflow(example_dir, baseflow_path)
        outputs.append(baseflow_path)

    if not skip_spectra:
        outputs.extend(plot_spectra(example_dir))

    if not skip_modes:
        for tag in ("direct", "adjoint"):
            for component in components:
                for part in parts:
                    rows = build_branch_rows(example_dir, tag, component, part, n_modes=n_modes)
                    out_path = example_dir / mosaic_name(tag, component, part)
                    plot_mode_triptych_mosaic(
                        out_path=out_path,
                        title="",
                        rows=rows,
                        xlim=PLOT_XLIM,
                        ylim=MOSAIC_YLIM,
                        dpi=DPI,
                        row_height=MOSAIC_ROW_HEIGHT,
                    )
                    print(f"Wrote {out_path}")
                    outputs.append(out_path)

    return outputs


def main() -> None:
    """Load the Example 4 NPZ and write the presentation PNGs."""
    example_dir = Path(__file__).resolve().parent
    for path in plot_example4_figures(example_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
