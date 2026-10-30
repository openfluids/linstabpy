"""Presentation figures for Martini Example 0 (curvilinear derivative).

Figures:

- ``example_0_curvilinear_mesh.png``
- ``example_0_curvilinear_validation.png``
- ``example_0_field_df_triptych.png`` (LinStab2D reference vs Python NPZ).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples._helpers import parity_npz_get
from examples.linstab2d.compare_martini_examples import load_matlab_bundle
from examples.linstab2d.plot_compare_utils import ensure_matplotlib
from linstabpy.utils.yaml_io import load_config

LINSTAB2D_RESULTS_DIRNAME = "LinStab2DResults"
MATLAB_EX0_STEM = "example_0_TestCurvilinearDerivative_results"

OUT_MESH = "example_0_curvilinear_mesh.png"
OUT_VALIDATION = "example_0_curvilinear_validation.png"
OUT_FIELD_DF = "example_0_field_df_triptych.png"

DPI = 400
LABEL_FS = 14
TITLE_FS = 13
TICK_FS = 11

REF_CODE_LABEL = "LinStab2D"
PY_CODE_LABEL = "linstabpy"


def linstab2d_results_dir(example_dir: Path) -> Path:
    """Directory for shipped LinStab2D reference NPZ bundles."""
    return example_dir / LINSTAB2D_RESULTS_DIRNAME


def matlab_ex0_fixture_path(example_dir: Path) -> Path:
    """Default path to the Example 0 LinStab2D reference NPZ."""
    return linstab2d_results_dir(example_dir) / f"{MATLAB_EX0_STEM}.npz"


def load_matlab_ex0_fixture(example_dir: Path) -> dict[str, np.ndarray]:
    return load_matlab_bundle(example_dir, MATLAB_EX0_STEM)


def _signed_cmap(plt):
    from matplotlib.colors import ListedColormap

    colors = plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 257))
    mid = colors.shape[0] // 2
    colors[mid - 4 : mid + 5, :3] = 1.0
    return ListedColormap(colors, name="rdbu_white_center")


def _style_cbar(cbar) -> None:
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.locator = MaxNLocator(nbins=5)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_useOffset(False)
    cbar.formatter = formatter
    cbar.update_ticks()


def _contour_panel(
    ax,
    X,
    Y,
    field,
    *,
    cmap,
    levels,
    vmin,
    vmax,
    title,
    xlabel=r"$x$",
    ylabel=r"$y$",
    extend="neither",
):
    levels_arr = np.linspace(vmin, vmax, levels)
    cf = ax.contourf(
        X,
        Y,
        field,
        levels=levels_arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend=extend,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=TITLE_FS, pad=3.0)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    return cf


def _draw_mesh_lines(ax, Xg, Yg, *, step: int) -> None:
    ax.plot(Xg[:, ::step], Yg[:, ::step], color="#4a5878", lw=0.7, alpha=0.9)
    ax.plot(Xg[::step, :].T, Yg[::step, :].T, color="#4a5878", lw=0.7, alpha=0.9)
    ax.plot(Xg[:, 0], Yg[:, 0], color="#394264", lw=1.0)
    ax.plot(Xg[:, -1], Yg[:, -1], color="#394264", lw=1.0)
    ax.plot(Xg[0, :], Yg[0, :], color="#394264", lw=1.0)
    ax.plot(Xg[-1, :], Yg[-1, :], color="#394264", lw=1.0)


def _computational_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    x0, x1 = cfg["xrange"]
    y_min, y_max = cfg["yrange"]
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    x = np.linspace(x0, x1, nx)
    if cfg.get("y_symmetry", False) and np.isclose(y_min, 0.0) and not np.isclose(y_max, 0.0):
        dy = y_max / (ny - 0.5)
        y_min = y_min + dy / 2.0
    y = np.linspace(y_min, y_max, ny)
    return np.meshgrid(x, y)


def plot_validation(data: dict[str, np.ndarray], out_path: Path) -> None:
    """Analytical | numerical | |error| (%) — campaign layout."""
    plt = ensure_matplotlib()

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    df = np.asarray(data["df"], dtype=float)
    df_an = np.asarray(data["df_an"], dtype=float)
    error = df - df_an

    df_lim = float(np.nanmax(np.abs(np.concatenate([df_an.ravel(), df.ravel()]))))
    df_lim = max(df_lim, 1e-30)
    err_abs = np.abs(error)
    err_pct = 100.0 * err_abs / df_lim
    err_vmax = (
        float(np.nanmax(err_pct[np.isfinite(err_pct)])) if np.any(np.isfinite(err_pct)) else 1.0
    )
    err_vmax = max(err_vmax, 1e-30)

    title_fs = 22
    label_fs = 26
    tick_fs = 22
    cbar_fs = 20
    phys_xlim = (-0.5, float(np.nanmax(x)))
    phys_ylim = (0.0, 1.4)

    gap_12 = 0.01
    gap_23 = 0.075
    gap_cbar = 0.005
    cbar_w = 0.02
    left, bottom, top = 0.08, 0.14, 0.84

    fig = plt.figure(figsize=(14.5, 4.6))
    ax0 = fig.add_axes([left, bottom, 0.22, top - bottom])
    ax1 = fig.add_axes([left, bottom, 0.22, top - bottom])
    ax2 = fig.add_axes([left, bottom, 0.22, top - bottom])
    cax_f = fig.add_axes([left, bottom, cbar_w, top - bottom])
    cax_e = fig.add_axes([left, bottom, cbar_w, top - bottom])

    kw = dict(cmap="RdBu_r", vmin=-df_lim, vmax=df_lim, shading="auto")
    im0 = ax0.pcolormesh(x, y, df_an, **kw)
    ax1.pcolormesh(x, y, df, **kw)
    im2 = ax2.pcolormesh(x, y, err_pct, cmap="magma", shading="auto", vmin=0.0, vmax=err_vmax)

    ax0.set_title(r"$\partial_x f_{\mathrm{analytical}}$", fontsize=title_fs)
    ax1.set_title(r"$\partial_x f_{\mathrm{numerical}}$", fontsize=title_fs)
    ax2.set_title(r"$|\mathrm{error}|$ (\%)", fontsize=title_fs)

    for ax in (ax0, ax1, ax2):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$", fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlim(phys_xlim)
        ax.set_ylim(phys_ylim)

    ax0.set_ylabel(r"$y$", fontsize=label_fs)
    ax0.tick_params(axis="y", left=True, labelleft=True)
    for ax in (ax1, ax2):
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=True, labelleft=False)

    cbar0 = fig.colorbar(im0, cax=cax_f)
    cbar0.ax.tick_params(labelsize=cbar_fs)
    cbar2 = fig.colorbar(im2, cax=cax_e)
    cbar2.ax.tick_params(labelsize=cbar_fs)

    fig.canvas.draw()

    def _size(ax):
        b = ax.get_position()
        return b.width, b.height

    w0, h0 = _size(ax0)
    w1, h1 = _size(ax1)
    w2, h2 = w1, h1
    y0 = bottom + 0.5 * ((top - bottom) - h0)
    y1 = bottom + 0.5 * ((top - bottom) - h1)
    y2 = bottom + 0.5 * ((top - bottom) - h2)

    x0 = left
    x1 = x0 + w0 + gap_12
    xf = x1 + w1 + gap_cbar
    x2 = x1 + w1 + gap_23
    xe = x2 + w2 + gap_cbar

    ax0.set_position([x0, y0, w0, h0])
    ax1.set_position([x1, y1, w1, h1])
    cax_f.set_position([xf, y1, cbar_w, h1])
    ax2.set_position([x2, y2, w2, h2])
    cax_e.set_position([xe, y2, cbar_w, h2])

    panel_mid = 0.5 * (x0 + x2 + w2)
    shift = 0.5 - panel_mid
    for ax in (ax0, ax1, ax2, cax_f, cax_e):
        pos = ax.get_position()
        ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mesh(data: dict[str, np.ndarray], cfg: dict, out_path: Path) -> None:
    """Computational (xi, eta) | physical (x, y) | f=cos(x)."""
    plt = ensure_matplotlib()

    Xi, Eta = _computational_grid(cfg)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    f = np.asarray(data["f"], dtype=float)
    Xphys, Yphys = x.T, y.T
    f_vmin = float(np.nanmin(f))
    f_vmax = float(np.nanmax(f))

    gap_12 = 0.06
    gap_23 = 0.01
    gap_3c = 0.005
    cbar_w = 0.025
    left, bottom, top = 0.08, 0.16, 0.86

    fig = plt.figure(figsize=(9.2, 3.6))
    ax0 = fig.add_axes([left, bottom, 0.25, top - bottom])
    ax1 = fig.add_axes([left, bottom, 0.25, top - bottom])
    ax2 = fig.add_axes([left, bottom, 0.25, top - bottom])
    cax2 = fig.add_axes([left, bottom, cbar_w, top - bottom])

    _draw_mesh_lines(ax0, Xi, Eta, step=4)
    ax0.set_title(r"Computational mesh", fontsize=TITLE_FS, pad=4.0)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel(r"$\xi$", fontsize=LABEL_FS, labelpad=4)
    ax0.set_ylabel(r"$\eta$", fontsize=LABEL_FS)
    ax0.tick_params(labelsize=TICK_FS)

    phys_xlim = (-0.5, float(np.nanmax(Xphys)))
    phys_ylim = (0.0, 1.4)

    _draw_mesh_lines(ax1, Xphys, Yphys, step=1)
    ax1.set_title(r"Mapped physical mesh", fontsize=TITLE_FS, pad=4.0)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"$x$", fontsize=LABEL_FS, labelpad=4)
    ax1.set_ylabel(r"$y$", fontsize=LABEL_FS)
    ax1.tick_params(labelsize=TICK_FS)
    ax1.set_xlim(phys_xlim)
    ax1.set_ylim(phys_ylim)

    cf = _contour_panel(
        ax2,
        Xphys,
        Yphys,
        f.T,
        cmap="viridis",
        levels=81,
        vmin=f_vmin,
        vmax=f_vmax,
        title=r"$f(x,y)=\cos(x)$",
        xlabel=r"$x$",
        ylabel="",
    )
    ax2.set_ylabel("")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(phys_xlim)
    ax2.set_ylim(phys_ylim)
    ax2.set_yticks(ax1.get_yticks())
    ax2.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        labelleft=False,
        direction="out",
        length=5.0,
        width=1.0,
        labelsize=TICK_FS,
    )
    _style_cbar(fig.colorbar(cf, cax=cax2, orientation="vertical"))

    fig.canvas.draw()

    def _size(ax):
        b = ax.get_position()
        return b.width, b.height

    w0, h0 = _size(ax0)
    w1, h1 = _size(ax1)
    w2, h2 = w1, h1
    y0 = bottom + 0.5 * ((top - bottom) - h0)
    y1 = bottom + 0.5 * ((top - bottom) - h1)
    y2 = bottom + 0.5 * ((top - bottom) - h2)
    x0 = left
    x1 = x0 + w0 + gap_12
    x2 = x1 + w1 + gap_23
    xc = x2 + w2 + gap_3c
    ax0.set_position([x0, y0, w0, h0])
    ax1.set_position([x1, y1, w1, h1])
    ax2.set_position([x2, y2, w2, h2])
    cax2.set_position([xc, y2, cbar_w, h2])
    ax2.set_yticks(ax1.get_yticks())
    ax2.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        labelleft=False,
        direction="out",
        length=5.0,
        width=1.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_field_df_triptych(
    *,
    python_data: dict[str, np.ndarray],
    matlab_bundle: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """LinStab2D vs linstabpy ``df`` triptych."""
    plt = ensure_matplotlib()

    title_fs = 22
    label_fs = 26
    tick_fs = 22
    cbar_fs = 20

    X_py = np.asarray(python_data["x"], dtype=float)
    Y_py = np.asarray(python_data["y"], dtype=float)
    python_field = np.asarray(python_data["df"], dtype=float)

    X_mat = np.asarray(parity_npz_get(matlab_bundle, "results_TestCurv/x"), dtype=float)
    Y_mat = np.asarray(parity_npz_get(matlab_bundle, "results_TestCurv/y"), dtype=float)
    matlab_field = np.asarray(parity_npz_get(matlab_bundle, "results_TestCurv/df"), dtype=float)

    if matlab_field.shape != python_field.shape and matlab_field.T.shape == python_field.shape:
        matlab_field = matlab_field.T
        X_mat = X_mat.T
        Y_mat = Y_mat.T
    elif (
        matlab_field.shape == python_field.shape
        and not np.allclose(matlab_field, python_field, rtol=0, atol=1e-9, equal_nan=True)
        and np.allclose(matlab_field.T, python_field, rtol=0, atol=1e-9, equal_nan=True)
    ):
        matlab_field = matlab_field.T
        X_mat = X_mat.T
        Y_mat = Y_mat.T

    err = np.abs(matlab_field - python_field)
    vmax = float(np.nanmax(np.abs(np.stack([matlab_field, python_field]))))
    vmax = max(vmax, 1e-30)
    err_pct = 100.0 * err / vmax
    err_vmax = (
        float(np.nanmax(err_pct[np.isfinite(err_pct)])) if np.any(np.isfinite(err_pct)) else 1.0
    )
    err_vmax = max(err_vmax, 1e-30)

    phys_xlim = (-0.5, float(np.nanmax(np.concatenate([X_mat.ravel(), X_py.ravel()]))))
    phys_ylim = (0.0, 1.4)
    gap_12 = 0.01
    gap_23 = 0.075
    gap_cbar = 0.005
    cbar_w = 0.02
    left, bottom, top = 0.08, 0.14, 0.84

    fig = plt.figure(figsize=(14.5, 4.6))
    ax0 = fig.add_axes([left, bottom, 0.22, top - bottom])
    ax1 = fig.add_axes([left, bottom, 0.22, top - bottom])
    ax2 = fig.add_axes([left, bottom, 0.22, top - bottom])
    cax_f = fig.add_axes([left, bottom, cbar_w, top - bottom])
    cax_e = fig.add_axes([left, bottom, cbar_w, top - bottom])

    kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    im0 = ax0.pcolormesh(X_mat, Y_mat, matlab_field, **kw)
    ax1.pcolormesh(X_py, Y_py, python_field, **kw)
    im2 = ax2.pcolormesh(X_py, Y_py, err_pct, cmap="magma", shading="auto", vmin=0.0, vmax=err_vmax)

    ax0.set_title(REF_CODE_LABEL, fontsize=title_fs)
    ax1.set_title(PY_CODE_LABEL, fontsize=title_fs)
    ax2.set_title(r"$|\mathrm{error}|$ (\%)", fontsize=title_fs)

    for ax in (ax0, ax1, ax2):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$", fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlim(phys_xlim)
        ax.set_ylim(phys_ylim)

    ax0.set_ylabel(r"$y$", fontsize=label_fs)
    ax0.tick_params(axis="y", left=True, labelleft=True)
    for ax in (ax1, ax2):
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=True, labelleft=False)

    cbar0 = fig.colorbar(im0, cax=cax_f)
    cbar0.ax.tick_params(labelsize=cbar_fs)
    cbar2 = fig.colorbar(im2, cax=cax_e)
    cbar2.ax.tick_params(labelsize=cbar_fs)

    fig.canvas.draw()

    def _size(ax):
        b = ax.get_position()
        return b.width, b.height

    w0, h0 = _size(ax0)
    w1, h1 = _size(ax1)
    w2, h2 = w1, h1
    y0 = bottom + 0.5 * ((top - bottom) - h0)
    y1 = bottom + 0.5 * ((top - bottom) - h1)
    y2 = bottom + 0.5 * ((top - bottom) - h2)

    x0 = left
    x1 = x0 + w0 + gap_12
    xf = x1 + w1 + gap_cbar
    x2 = x1 + w1 + gap_23
    xe = x2 + w2 + gap_cbar

    ax0.set_position([x0, y0, w0, h0])
    ax1.set_position([x1, y1, w1, h1])
    cax_f.set_position([xf, y1, cbar_w, h1])
    ax2.set_position([x2, y2, w2, h2])
    cax_e.set_position([xe, y2, cbar_w, h2])

    panel_mid = 0.5 * (x0 + x2 + w2)
    shift = 0.5 - panel_mid
    for ax in (ax0, ax1, ax2, cax_f, cax_e):
        pos = ax.get_position()
        ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_example0_figures(
    data: dict[str, np.ndarray],
    *,
    example_dir: Path,
    cfg: dict | None = None,
) -> tuple[Path, Path, Path]:
    """Write all three Example 0 presentation PNGs next to the driver script."""
    if cfg is None:
        cfg = load_config(example_dir / "example_0_curvilinear_validation.yaml")

    mesh_path = example_dir / OUT_MESH
    validation_path = example_dir / OUT_VALIDATION
    triptych_path = example_dir / OUT_FIELD_DF

    plot_mesh(data, cfg, mesh_path)
    plot_validation(data, validation_path)
    matlab_bundle = load_matlab_ex0_fixture(example_dir)
    plot_field_df_triptych(
        python_data=data,
        matlab_bundle=matlab_bundle,
        out_path=triptych_path,
    )
    return mesh_path, validation_path, triptych_path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def main() -> None:
    """Load the Example 0 NPZ and write the three presentation PNGs."""
    example_dir = Path(__file__).resolve().parent
    data = _load_npz(example_dir / "example_0_curvilinear_validation.npz")
    for path in plot_example0_figures(data, example_dir=example_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
