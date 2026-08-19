"""Presentation figures for Martini Example 3 (streaky BL / Floquet).

Matches the thesis/campaign layout under
``cases/Martini2024/comparison/plots/example_3_streaky_bl/``.

- ``example_3_streaky_bl_baseflow.png`` — streak :math:`W` (from :math:`L_x=2\pi` case).
- ``example_3_streaky_bl_spectrum_temporal_{Lx2pi,Lxpi_nfloq0,Lxpi_nfloq2}.png``.
- ``example_3_streaky_bl_spectrum_temporal_Lxpi_spatial_run.png``, ``example_3_streaky_bl_spectrum_spatial_Lxpi.png``.
- ``example_3_streaky_bl_modes_temporal_{tag}_w[_imag]_mosaic.png`` — LinStab2D | linstabpy | error.
- ``example_3_streaky_bl_modes_spatial_Lxpi_w[_imag]_mosaic.png`` — linstabpy-only spatial modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.martini.compare_martini_examples import (
    LINSTAB2D_RESULTS_DIRNAME,
    _best_spectrum_pairs,
    _mode_matrix,
    _npz_get,
    load_matlab_bundle,
)
from examples.martini.plot_compare_utils import (
    complex_vector,
    ensure_matplotlib,
    extract_component,
    load_python_npz,
    matlab_component_grid,
    maybe_compact_python_modes_to_matlab,
    plot_mode_triptych_mosaic,
    plot_spectrum_overlay,
    python_component_grid,
    scaled_python_mode,
)

FIGURE_PREFIX = "example_3_streaky_bl"
OUT_BASEFLOW = f"{FIGURE_PREFIX}_baseflow.png"

IDX_PREFIX = "results_StreakyBL_temporal/idx"
PLOT_GRID_PREFIX = "results_StreakyBL_temporal/plot_grid"
OMEGA_KEY_MAT = "results_StreakyBL_temporal/omega"
MODES_KEY_MAT = "results_StreakyBL_temporal/V"
SAVED_MODE_INDICES_KEY = "results_StreakyBL_temporal/saved_mode_indices"
ALPHA_KEY_MAT = "results_StreakyBL_spatial/alphas"

TEMPORAL_CASES = (
    {
        "tag": "Lx2pi",
        "python_npz": "example_3_paper_temporal_Lx2pi_nP2.npz",
        "matlab_stem": "example_3_paper_temporal_Lx2pi_nP2_results_temporal",
        "out_spectrum": f"{FIGURE_PREFIX}_spectrum_temporal_Lx2pi.png",
        "n_annotate": 8,
    },
    {
        "tag": "Lxpi_nfloq0",
        "python_npz": "example_3_paper_temporal_Lxpi_nfloq0.npz",
        "matlab_stem": "example_3_paper_temporal_Lxpi_nfloq0_results_temporal",
        "out_spectrum": f"{FIGURE_PREFIX}_spectrum_temporal_Lxpi_nfloq0.png",
        "n_annotate": 8,
    },
    {
        "tag": "Lxpi_nfloq2",
        "python_npz": "example_3_paper_temporal_Lxpi_nfloq2.npz",
        "matlab_stem": "example_3_paper_temporal_Lxpi_nfloq2_results_temporal",
        "out_spectrum": f"{FIGURE_PREFIX}_spectrum_temporal_Lxpi_nfloq2.png",
        "n_annotate": 8,
    },
)

SPATIAL_CASE = {
    "python_npz": "example_3_paper_spatial_Lxpi_nP2.npz",
    "matlab_temporal_stem": "example_3_paper_spatial_Lxpi_nP2_results_temporal",
    "matlab_spatial_stem": "example_3_paper_spatial_Lxpi_nP2_results_spatial",
    "out_spectrum_temporal": f"{FIGURE_PREFIX}_spectrum_temporal_Lxpi_spatial_run.png",
    "out_spectrum_spatial": f"{FIGURE_PREFIX}_spectrum_spatial_Lxpi.png",
}

MODE_CASES = TEMPORAL_CASES
BASEFLOW_PYTHON_NPZ = TEMPORAL_CASES[0]["python_npz"]

N_MODES = 3
DEFAULT_COMPONENTS = ("w",)
DEFAULT_PARTS = ("real", "imag")
MOSAIC_YLIM = (0.0, 5.0)
MOSAIC_ROW_HEIGHT = 1.55

TITLE_FS = 26
LABEL_FS = 24
TICK_FS = 20
CBAR_FS = 18
DPI = 400
SPEC_MARKER = {"marker_size": 10.0, "marker_edgewidth": 3.0, "dpi": 300}


def linstab2d_results_dir(example_dir: Path) -> Path:
    return example_dir / LINSTAB2D_RESULTS_DIRNAME


def load_matlab_ex3_fixture(example_dir: Path, stem: str) -> dict[str, np.ndarray]:
    return load_matlab_bundle(example_dir, stem)


def mosaic_name(tag: str, component: str, part: str) -> str:
    if part == "real":
        return f"{FIGURE_PREFIX}_modes_temporal_{tag}_{component}_mosaic.png"
    return f"{FIGURE_PREFIX}_modes_temporal_{tag}_{component}_{part}_mosaic.png"


def spatial_mosaic_name(component: str, part: str) -> str:
    if part == "real":
        return f"{FIGURE_PREFIX}_modes_spatial_Lxpi_{component}_mosaic.png"
    return f"{FIGURE_PREFIX}_modes_spatial_Lxpi_{component}_{part}_mosaic.png"


def _part_tex(part: str) -> str:
    if part == "imag":
        return r"\Im"
    return r"\Re"


def _spectrum_style() -> dict[str, Any]:
    return {
        **SPEC_MARKER,
        "title_fs": TITLE_FS,
        "label_fs": LABEL_FS,
        "tick_fs": TICK_FS,
        "legend_fs": 18,
    }


def plot_baseflow(example_dir: Path, out_path: Path) -> None:
    plt = ensure_matplotlib()
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    py = load_python_npz(example_dir / BASEFLOW_PYTHON_NPZ)
    X = np.asarray(py["X"], dtype=float)
    Y = np.asarray(py["Y"], dtype=float)
    if "baseflow_w" in py:
        field = np.asarray(py["baseflow_w"], dtype=float)
        field_name = "W"
    else:
        field = np.asarray(py["baseflow_u"], dtype=float)
        field_name = "U"

    finite = field[np.isfinite(field)]
    v_lo = float(np.min(finite))
    v_hi = float(np.max(finite))
    pad = 0.02 * max(v_hi - v_lo, 1.0)

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    cf = ax.contourf(
        X,
        Y,
        field,
        levels=np.linspace(v_lo - pad, v_hi + pad, 101),
        cmap="turbo",
        extend="neither",
    )
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
    cbar.ax.tick_params(labelsize=CBAR_FS)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Wrote {out_path} ({field_name} in "
        f"[{float(np.min(finite)):.3f}, {float(np.max(finite)):.3f}])"
    )


def plot_temporal_spectra(example_dir: Path) -> list[Path]:
    style = _spectrum_style()
    outputs: list[Path] = []
    for case in TEMPORAL_CASES:
        py = load_python_npz(example_dir / case["python_npz"])
        matlab = load_matlab_ex3_fixture(example_dir, case["matlab_stem"])
        out_path = example_dir / case["out_spectrum"]
        plot_spectrum_overlay(
            out_path=out_path,
            title="",
            matlab_eigs=complex_vector(_npz_get(matlab, OMEGA_KEY_MAT)),
            python_eigs=complex_vector(py["temporal_omega"]),
            xlabel=r"$\Re(\omega)$",
            ylabel=r"$\Im(\omega)$",
            n_annotate=int(case["n_annotate"]),
            matlab_zoom_inset=False,
            **style,
        )
        print(f"Wrote {out_path}")
        outputs.append(out_path)
    return outputs


def plot_spatial_spectra(example_dir: Path) -> list[Path]:
    style = _spectrum_style()
    case = SPATIAL_CASE
    py = load_python_npz(example_dir / case["python_npz"])
    matlab_t = load_matlab_ex3_fixture(example_dir, case["matlab_temporal_stem"])
    matlab_s = load_matlab_ex3_fixture(example_dir, case["matlab_spatial_stem"])

    out_t = example_dir / case["out_spectrum_temporal"]
    plot_spectrum_overlay(
        out_path=out_t,
        title="",
        matlab_eigs=complex_vector(_npz_get(matlab_t, OMEGA_KEY_MAT)),
        python_eigs=complex_vector(py["temporal_omega"]),
        xlabel=r"$\Re(\omega)$",
        ylabel=r"$\Im(\omega)$",
        n_annotate=0,
        matlab_zoom_inset=False,
        **style,
    )
    print(f"Wrote {out_t}")

    out_s = example_dir / case["out_spectrum_spatial"]
    plot_spectrum_overlay(
        out_path=out_s,
        title="",
        matlab_eigs=complex_vector(_npz_get(matlab_s, ALPHA_KEY_MAT)),
        python_eigs=complex_vector(py["spatial_alpha"]),
        xlabel=r"$\Re(\alpha)$",
        ylabel=r"$\Im(\alpha)$",
        n_annotate=0,
        matlab_zoom_inset=False,
        **style,
    )
    print(f"Wrote {out_s}")
    return [out_t, out_s]


def _matlab_mode_pairs(
    matlab: dict[str, np.ndarray],
    omega_mat: np.ndarray,
    omega_py: np.ndarray,
) -> list[tuple[int, int]]:
    """Return ``(matlab_V_column, python_mode_index)`` using saved eig indices."""
    saved = np.asarray(_npz_get(matlab, SAVED_MODE_INDICES_KEY), dtype=int).reshape(-1)
    mat_eig = saved - 1 if int(saved.min()) >= 1 else saved
    if mat_eig.size < N_MODES:
        raise RuntimeError(f"Expected {N_MODES} saved MATLAB modes, got {mat_eig.size}")

    pair_map = {
        int(mat_i): int(py_i)
        for mat_i, py_i, _, _ in _best_spectrum_pairs(
            omega_mat,
            omega_py,
            max(int(omega_mat.size), int(omega_py.size)),
        )
    }
    pairs: list[tuple[int, int]] = []
    for col, eig_i in enumerate(mat_eig[:N_MODES]):
        eig_i = int(eig_i)
        if eig_i not in pair_map:
            raise RuntimeError(f"No Python pair for saved MATLAB mode index {eig_i + 1}")
        pairs.append((col, pair_map[eig_i]))
    return pairs


def build_mode_rows(
    example_dir: Path,
    case: dict[str, str],
    component: str,
    part: str,
) -> list[dict]:
    component = component if component != "t" else "T"
    op = _part_tex(part)

    py = load_python_npz(example_dir / case["python_npz"])
    matlab = load_matlab_ex3_fixture(example_dir, case["matlab_stem"])

    omega_mat = complex_vector(_npz_get(matlab, OMEGA_KEY_MAT))
    omega_py = complex_vector(py["temporal_omega"])
    modes_mat = _npz_get(matlab, MODES_KEY_MAT)
    modes_py = py["temporal_modes"]

    mode_pairs = _matlab_mode_pairs(matlab, omega_mat, omega_py)
    if len(mode_pairs) < N_MODES:
        raise RuntimeError(f"Only matched {len(mode_pairs)}/{N_MODES} modes for {case['tag']}")

    saved = np.asarray(_npz_get(matlab, SAVED_MODE_INDICES_KEY), dtype=int).reshape(-1)
    mat_eig = saved - 1 if int(saved.min()) >= 1 else saved

    X_py = np.asarray(py["X"], dtype=float)
    Y_py = np.asarray(py["Y"], dtype=float)
    used_py = (
        np.asarray(py["used_ind"], dtype=int)
        if "used_ind" in py
        else np.arange(int(np.prod(X_py.shape)), dtype=int)
    )
    idx_key = f"idx_{component}_j"
    idx_comp = (
        py[idx_key]
        if idx_key in py
        else _npz_get(matlab, f"{IDX_PREFIX}/{component}_j")
    )
    one_based = idx_key not in py
    modes_py, idx_comp, one_based = maybe_compact_python_modes_to_matlab(
        modes_py,
        modes_mat,
        idx_comp,
        one_based_idx=one_based,
    )

    var = component
    rows: list[dict] = []
    for rank, (mat_col, py_i) in enumerate(mode_pairs, start=1):
        X_mat, Y_mat, grid_mat = matlab_component_grid(
            modes_mat,
            mat_col,
            matlab,
            idx_prefix=IDX_PREFIX,
            plot_grid_prefix=PLOT_GRID_PREFIX,
            component=component,
            part=part,
        )
        py_scaled = scaled_python_mode(modes_py, modes_mat, py_i, mat_col)
        tmp = _mode_matrix(modes_py).copy()
        tmp[:, py_i] = py_scaled
        grid_py = python_component_grid(
            tmp,
            py_i,
            X=X_py,
            used_ind=used_py,
            idx_comp=idx_comp,
            one_based_idx=one_based,
            part=part,
        )
        om_m = omega_mat[int(mat_eig[mat_col])]
        om_p = omega_py[py_i]
        rows.append(
            {
                "row_label": rf"${op}({var}^{{({rank})}})$",
                "X_mat": X_mat,
                "Y_mat": Y_mat,
                "matlab_grid": grid_mat,
                "X_py": X_py,
                "Y_py": Y_py,
                "python_grid": grid_py,
                "ref_gain": None,
                "py_gain": None,
                "banner_ref": rf"$\omega={om_m.real:.4f}{om_m.imag:+.4f}\mathrm{{i}}$",
                "banner_py": rf"$\omega={om_p.real:.4f}{om_p.imag:+.4f}\mathrm{{i}}$",
            }
        )
    return rows


def _physical_spatial_modes(py: dict[str, np.ndarray]) -> np.ndarray:
    modes = _mode_matrix(py["spatial_modes"])
    idx_keys = [k for k in py if k.startswith("idx_") and k.endswith("_j")]
    n_states = int(max(int(np.max(np.asarray(py[k]))) for k in idx_keys) + 1)
    if modes.shape[0] > n_states:
        modes = modes[:n_states, :]
    return modes


def _phase_align_peak(
    vec: np.ndarray,
    idx_comp: np.ndarray,
    *,
    one_based: bool,
) -> np.ndarray:
    comp = extract_component(vec, idx_comp, one_based=one_based)
    peak = comp[int(np.nanargmax(np.abs(comp)))]
    if not np.isfinite(peak) or abs(peak) == 0.0:
        return np.asarray(vec, dtype=complex)
    return np.asarray(vec, dtype=complex) * np.exp(-1j * np.angle(peak))


def build_spatial_mode_rows(
    example_dir: Path,
    component: str,
    part: str,
) -> list[dict]:
    component = component if component != "t" else "T"
    op = _part_tex(part)

    py = load_python_npz(example_dir / SPATIAL_CASE["python_npz"])
    if "spatial_modes" not in py or "spatial_alpha" not in py:
        raise RuntimeError("spatial NPZ has no spatial_modes / spatial_alpha")

    alpha = complex_vector(py["spatial_alpha"])
    target = complex(np.asarray(py["alpha_target"]).reshape(-1)[0])
    order = np.argsort(np.abs(alpha - target))[:N_MODES]
    modes = _physical_spatial_modes(py)

    X = np.asarray(py["X"], dtype=float)
    Y = np.asarray(py["Y"], dtype=float)
    used = (
        np.asarray(py["used_ind"], dtype=int)
        if "used_ind" in py
        else np.arange(int(np.prod(X.shape)), dtype=int)
    )
    idx_key = f"idx_{component}_j"
    if idx_key not in py:
        raise RuntimeError(f"spatial NPZ missing {idx_key}")
    idx_comp = py[idx_key]

    var = component
    rows: list[dict] = []
    tmp = modes.copy()
    for rank, py_i in enumerate((int(i) for i in order), start=1):
        tmp[:, py_i] = _phase_align_peak(modes[:, py_i], idx_comp, one_based=False)
        grid = python_component_grid(
            tmp,
            py_i,
            X=X,
            used_ind=used,
            idx_comp=idx_comp,
            one_based_idx=False,
            part=part,
        )
        a = alpha[py_i]
        rows.append(
            {
                "row_label": rf"${op}({var}^{{({rank})}})$",
                "X": X,
                "Y": Y,
                "grid": grid,
                "banner": rf"$\alpha={a.real:.4f}{a.imag:+.4f}\mathrm{{i}}$",
            }
        )
    return rows


def plot_python_mode_stack(
    *,
    out_path: Path,
    rows: list[dict],
    ylim: tuple[float, float] | None,
) -> None:
    plt = ensure_matplotlib()
    n_rows = len(rows)
    fig_h = 0.85 + n_rows * MOSAIC_ROW_HEIGHT
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, fig_h), squeeze=False, sharex=True)
    for r, (row, ax) in enumerate(zip(rows, axes[:, 0])):
        X = np.asarray(row["X"], dtype=float)
        Y = np.asarray(row["Y"], dtype=float)
        grid = np.asarray(row["grid"], dtype=float)
        peak = max(float(np.nanmax(np.abs(grid))), 1e-30)
        im = ax.pcolormesh(
            X, Y, grid / peak, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto"
        )
        ax.set_aspect("auto")
        if ylim is not None:
            y0, y1 = float(ylim[0]), float(ylim[1])
            ax.set_ylim(y0, y1)
            ax.set_yticks([y0, 0.5 * (y0 + y1), y1])
        ax.tick_params(labelsize=TICK_FS)
        ax.set_title(str(row["banner"]), fontsize=18, pad=4.0)
        raw = str(row.get("row_label", f"row {r + 1}")).strip()
        side = rf"$y-{raw[1:-1]}$" if raw.startswith("$") and raw.endswith("$") else rf"$y$-{raw}"
        ax.set_ylabel(side, fontsize=18, labelpad=8)
        if r == n_rows - 1:
            ax.set_xlabel(r"$x$", fontsize=LABEL_FS)
        cbar = fig.colorbar(im, ax=ax, pad=0.015)
        cbar.ax.tick_params(labelsize=CBAR_FS)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_example3_figures(
    example_dir: Path,
    *,
    components: tuple[str, ...] = DEFAULT_COMPONENTS,
    parts: tuple[str, ...] = DEFAULT_PARTS,
    skip_baseflow: bool = False,
    skip_spectra: bool = False,
    skip_modes: bool = False,
) -> list[Path]:
    """Write all Example 3 presentation PNGs next to the driver script."""
    outputs: list[Path] = []

    if not skip_baseflow:
        baseflow_path = example_dir / OUT_BASEFLOW
        plot_baseflow(example_dir, baseflow_path)
        outputs.append(baseflow_path)

    if not skip_spectra:
        outputs.extend(plot_temporal_spectra(example_dir))
        outputs.extend(plot_spatial_spectra(example_dir))

    if not skip_modes:
        for case in MODE_CASES:
            for component in components:
                for part in parts:
                    rows = build_mode_rows(example_dir, case, component, part)
                    out_path = example_dir / mosaic_name(case["tag"], component, part)
                    plot_mode_triptych_mosaic(
                        out_path=out_path,
                        title="",
                        rows=rows,
                        ylim=MOSAIC_YLIM,
                        dpi=DPI,
                        row_height=MOSAIC_ROW_HEIGHT,
                    )
                    print(f"Wrote {out_path}")
                    outputs.append(out_path)

        for component in components:
            for part in parts:
                rows = build_spatial_mode_rows(example_dir, component, part)
                out_path = example_dir / spatial_mosaic_name(component, part)
                plot_python_mode_stack(out_path=out_path, rows=rows, ylim=MOSAIC_YLIM)
                print(f"Wrote {out_path}")
                outputs.append(out_path)

    return outputs
