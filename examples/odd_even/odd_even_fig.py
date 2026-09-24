# SPDX-License-Identifier: GPL-3.0-only
"""Plot the odd-even figure from the run record of ``odd_even.py``.

Panels:

- (a) curvilinear mesh, coarse, epsilon = 0. It shows the odd-even mode.
- (b) curvilinear mesh, coarse, epsilon = 0.05. Dissipation removes the mode.
- (c) curvilinear mesh, fine, epsilon = 0. Refinement removes the mode.
- (d) fraction of the response energy at the grid scale against epsilon.

Panels (a) to (c) show the real part of the streamwise response velocity.
We divide each field by its largest magnitude. We do no solve here.

Run from the repository root, after ``odd_even.py``::

    python -m examples.odd_even.odd_even_fig
"""

from __future__ import annotations

import csv
import logging

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from examples.figure_style import apply
from examples.odd_even.odd_even import CSV_PATH, EPS_LADDER, HERE, LX, NPZ_PATH, RESOLUTIONS

LOG = logging.getLogger(__name__)

PNG_PATH = HERE / "odd_even.png"
EPS_SHOWN = 0.05
DPI = 400


def mesh_size(nx: int, ny: int) -> str:
    r"""Return the node count as a math string.

    Parameters
    ----------
    nx, ny : int
        Number of nodes along x and along y.

    Returns
    -------
    str
        For example ``$81 \times 41$``.
    """
    return rf"${nx} \times {ny}$"


def plot_field(
    ax: Axes, data: np.lib.npyio.NpzFile, nx: int, ny: int, eps: float, letter: str
) -> None:
    """Plot the normalised streamwise response of one curvilinear case.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes.
    data : numpy.lib.npyio.NpzFile
        The fields that ``odd_even.py`` writes.
    nx, ny : int
        Number of nodes along x and along y.
    eps : float
        Operator-dissipation coefficient of the case.
    letter : str
        Panel letter.
    """
    res = f"{nx}x{ny}"
    u = np.real(data[f"u_curvilinear_{res}_{eps:g}"])
    u /= np.max(np.abs(u))
    ax.pcolormesh(
        data[f"X_curvilinear_{res}"],
        data[f"Y_curvilinear_{res}"],
        u,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        shading="nearest",
        rasterized=True,
    )
    ax.set_title(
        rf"({letter}) curvilinear, $\varepsilon = {eps:g}$, {mesh_size(nx, ny)}", loc="left"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(0.0, LX)
    ax.set_ylim(-0.2, 0.6)


def plot_fraction(ax: Axes, rows: list[dict[str, str]]) -> None:
    """Plot the grid-scale energy fraction against epsilon for the four ladders.

    The epsilon samples are not evenly spaced. The panel puts them at equal
    intervals and labels each tick with its value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes.
    rows : list of dict
        The rows of ``odd_even.csv``.
    """
    (nx_c, ny_c), (nx_f, ny_f) = RESOLUTIONS
    # Thick lines for the coarse mesh are drawn first, so they stay at the back.
    styles = (
        ("rectilinear", nx_c, ny_c, "Cartesian mesh", "o", 2.6),
        ("curvilinear", nx_c, ny_c, "curvilinear mesh", "s", 2.6),
        ("rectilinear", nx_f, ny_f, "Cartesian mesh, refined", "o", 1.2),
        ("curvilinear", nx_f, ny_f, "curvilinear mesh, refined", "s", 1.2),
    )
    for zorder, (kind, nx, ny, label, marker, lw) in enumerate(styles, start=1):
        points = sorted(
            (float(r["epsilon"]), float(r["r_n"]))
            for r in rows
            if r["mesh_kind"] == kind and int(r["nx"]) == nx and int(r["ny"]) == ny
        )
        positions = [EPS_LADDER.index(eps) for eps, _ in points]
        ax.plot(
            positions,
            [fraction for _, fraction in points],
            marker=marker,
            markersize=4.0,
            linewidth=lw,
            label=label,
            zorder=zorder,
        )
    ax.set_xticks(range(len(EPS_LADDER)))
    ax.set_xticklabels([rf"${eps:g}$" for eps in EPS_LADDER])
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("energy fraction at the grid scale")
    ax.set_title("(d) grid-scale content of the response", loc="left")
    ax.legend(loc="best", frameon=True)


def main() -> int:
    """Write ``odd_even.png`` from ``odd_even.csv`` and ``output/odd_even.npz``.

    Returns
    -------
    int
        Exit status 0.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply()
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    (nx_c, ny_c), (nx_f, ny_f) = RESOLUTIONS
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained")
    with np.load(NPZ_PATH) as data:
        plot_field(axes[0, 0], data, nx_c, ny_c, 0.0, "a")
        plot_field(axes[0, 1], data, nx_c, ny_c, EPS_SHOWN, "b")
        plot_field(axes[1, 0], data, nx_f, ny_f, 0.0, "c")
    plot_fraction(axes[1, 1], rows)
    fig.savefig(PNG_PATH, dpi=DPI)
    plt.close(fig)
    LOG.info("wrote %s", PNG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
