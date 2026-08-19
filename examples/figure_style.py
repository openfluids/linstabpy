# SPDX-License-Identifier: GPL-3.0-only
"""Journal figure style for every example PNG.

Probe TeX by writing a PNG of ``$x$``. Use usetex when that works.
Else use Computer Modern mathtext. A failed usetex save retries once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIGURE_DPI = 300
_APPLIED = False

# Debian/Ubuntu packages for matplotlib usetex (see examples/song/README.md).
LATEX_SYSTEM_PACKAGES_DEBIAN = (
    "texlive-latex-extra",
    "texlive-fonts-recommended",
    "dvipng",
    "ghostscript",
)


def latex_available() -> bool:
    """Return True when Agg usetex can write a PNG of ``$x$``."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    previous = matplotlib.rcParams.get("text.usetex", False)
    fig = None
    try:
        matplotlib.rcParams["text.usetex"] = True
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$x$")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.tell() > 0
    except Exception:
        return False
    finally:
        matplotlib.rcParams["text.usetex"] = previous
        if fig is not None:
            plt.close(fig)


def apply() -> None:
    """Set rcParams once. Call at the top of every figure script."""
    global _APPLIED
    if _APPLIED:
        return
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
            "font.size": 9,
            "text.usetex": latex_available(),
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.linewidth": 0.6,
            "axes.titlepad": 3.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.borderaxespad": 0.3,
            "figure.titlesize": 9,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
        }
    )
    _APPLIED = True


def plotting_prerequisites_message() -> str:
    """Human-readable install hint when LaTeX is missing."""
    pkgs = " ".join(LATEX_SYSTEM_PACKAGES_DEBIAN)
    return (
        "Example figures use matplotlib usetex (Computer Modern). Install a TeX stack, e.g.\n"
        f"  sudo apt install {' '.join(LATEX_SYSTEM_PACKAGES_DEBIAN)}\n"
        "On Fedora: texlive-scheme-medium dvipng ghostscript."
    )


def check_plot_environment(*, require_latex: bool = False) -> bool:
    """Apply journal rcParams and optionally require a working LaTeX toolchain."""
    apply()
    tex_ok = latex_available()
    if require_latex and not tex_ok:
        raise RuntimeError(plotting_prerequisites_message())
    if not tex_ok:
        import warnings

        warnings.warn(
            "LaTeX not available; writing figures with matplotlib mathtext fallback.",
            UserWarning,
            stacklevel=2,
        )
    return tex_ok


def signed_cmap():
    """Diverging map with a white band at zero."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 257))
    mid = colors.shape[0] // 2
    colors[mid - 3 : mid + 4, :3] = 1.0
    return ListedColormap(colors, name="rdbu_zero_white")


def save(fig, path: str | Path, *, dpi: int | None = None, pad_inches: float = 0.03) -> Path:
    """Write ``path`` at 300 dpi. Retry once with mathtext if usetex fails."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    apply()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dpi = FIGURE_DPI if dpi is None else int(dpi)
    try:
        fig.savefig(out, dpi=save_dpi, bbox_inches="tight", pad_inches=pad_inches)
    except Exception:
        if not mpl.rcParams.get("text.usetex"):
            raise
        mpl.rcParams["text.usetex"] = False
        fig.savefig(out, dpi=save_dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return out


def thin_colorbar(fig, mappable, ax, label: str = ""):
    """Attach a narrow colorbar to one axes."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.06)
    cbar = fig.colorbar(mappable, cax=cax)
    if label:
        cbar.set_label(label)
    cbar.ax.tick_params(length=2.0, width=0.5, labelsize=7)
    return cbar


def y_limit_for_support(y: np.ndarray, *fields: np.ndarray, frac: float = 0.995) -> float:
    """Return a y-max that holds ``frac`` of the peak |field| support."""
    y1 = np.asarray(y)
    if y1.ndim == 2:
        y_line = y1[:, 0] if y1.shape[1] else y1.ravel()
        amp = np.zeros(y1.shape[0])
        for field in fields:
            amp = np.maximum(amp, np.max(np.abs(np.asarray(field)), axis=1))
    else:
        y_line = y1
        amp = np.zeros_like(y_line, dtype=float)
        for field in fields:
            amp = np.maximum(amp, np.abs(np.asarray(field)).reshape(-1))
    peak = float(np.max(amp))
    if peak <= 0.0:
        return float(np.max(y_line))
    mask = amp >= (1.0 - frac) * peak
    if not np.any(mask):
        return float(np.max(y_line))
    return float(np.max(y_line[mask])) * 1.08
