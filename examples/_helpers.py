"""Shared helpers for example scripts.

Grid reconstruction, plotting, and other utilities used across multiple examples.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Re-export from canonical location for callers that import from here.
from linstabpy.utils.yaml_io import load_config  # noqa: F401


def load_or_compute(
    npz_path,
    compute_fn,
    *,
    required_keys=None,
    verbose=True,
    **np_load_kwargs,
):
    """Return ``np.load(npz_path)``; on miss (absent/truncated/unreadable/schema) call ``compute_fn`` and retry."""

    def _load():
        return np.load(npz_path, **np_load_kwargs)

    try:
        data = _load()
        if required_keys is not None and not required_keys.issubset(set(data.files)):
            data.close()
            raise KeyError("stale cache schema")
    except (FileNotFoundError, EOFError, OSError, ValueError, KeyError) as exc:
        if verbose:
            print(format_cache_recompute_message(npz_path, exc))
        compute_fn()
        data = _load()
    return data


from examples.figure_style import apply as _apply_figure_style
from examples.figure_style import signed_cmap as signed_zero_white_cmap  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
PARITY_NPZ_DIR = REPO_ROOT / "validation" / "martini2024" / "fixtures"
LEGACY_PARITY_NPZ_DIR = REPO_ROOT / "tools" / "parity" / "npz"
_STYLE_CONFIGURED = False


def repo_relative_path(path: str | Path) -> str:
    """Return a repo-relative display path when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def format_cache_recompute_message(
    npz_path: str | Path,
    exc: Exception | None,
) -> str:
    """Describe a cache miss or stale cache in user-facing terms."""
    cache_path = repo_relative_path(npz_path)
    if exc is None:
        return f"Cache miss or stale cache for {cache_path}; recomputing local results."
    if isinstance(exc, FileNotFoundError):
        return f"Cache miss for {cache_path}; recomputing local results."
    if isinstance(exc, EOFError):
        return f"Truncated cache for {cache_path}; recomputing local results."
    if isinstance(exc, KeyError) and exc.args == ("stale cache schema",):
        return f"Stale cache schema for {cache_path}; recomputing local results."

    reason = "Unreadable cache" if isinstance(exc, OSError) else "Invalid cache contents"
    return f"{reason} for {cache_path} ({exc}); recomputing local results."


def to_grid(
    values: np.ndarray,
    used_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Scatter a DOF vector back onto the full 2D grid (NaN-filled)."""
    flat = np.full(int(np.prod(shape)), np.nan, dtype=complex)
    flat[np.asarray(used_indices)] = np.asarray(values)
    return flat.reshape(shape, order="F")


def leading_mode_real_frequency(
    omega: np.ndarray,
    imag_shift: float = 0.0,
) -> complex:
    """Return the real part of the least-damped mode plus an imag shift."""
    values = np.asarray(omega, dtype=complex).reshape(-1)
    if values.size == 0:
        raise ValueError("omega must contain at least one eigenvalue")
    lead = values[int(np.argmax(np.imag(values)))]
    return complex(float(np.real(lead)), float(imag_shift))


def save_figure(fig, path: str | Path) -> Path:
    """Save a matplotlib figure with the repo-wide example defaults."""
    from examples.figure_style import save as _save_figure

    configure_example_style()
    output = _save_figure(fig, path)
    print(f"Figure saved to {repo_relative_path(output)}")
    return output


def apply_figure_layout(
    fig,
    *,
    rect: tuple[float, float, float, float] | None = None,
    pad: float = 0.25,
    h_pad: float | None = None,
    w_pad: float | None = None,
) -> None:
    """Apply tight layout while suppressing Matplotlib's mixed-axes warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "This figure includes Axes that are not compatible with tight_layout, "
                "so results might be incorrect."
            ),
            category=UserWarning,
        )
        fig.tight_layout(rect=rect, pad=min(pad, 0.25), h_pad=h_pad, w_pad=w_pad)


def configure_example_style() -> None:
    """Apply a consistent math-text-heavy style across all example figures."""
    global _STYLE_CONFIGURED
    if _STYLE_CONFIGURED:
        return

    _apply_figure_style()
    _STYLE_CONFIGURED = True


@lru_cache(maxsize=1)
def signed_zero_white_cmap():
    """Return a diverging colormap with a white band centered at zero."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 257))
    mid = colors.shape[0] // 2
    colors[mid - 4 : mid + 5, :3] = 1.0
    return ListedColormap(colors, name="rdbu_white_center")


def label_axes(
    ax,
    *,
    xlabel: str = r"$x$",
    ylabel: str = r"$y$",
) -> None:
    """Apply math-text axis labels."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def format_scientific_ticks(
    ax,
    *,
    axes: tuple[str, ...] = ("x", "y"),
    powerlimits: tuple[int, int] = (-2, 2),
) -> None:
    """Use scientific-notation formatters with math text on selected axes."""
    from matplotlib.ticker import ScalarFormatter

    for axis_name in axes:
        if axis_name == "x":
            axis = ax.xaxis
        elif axis_name == "y":
            axis = ax.yaxis
        else:
            raise ValueError(f"Unknown axis {axis_name!r}")

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits(powerlimits)
        formatter.set_useOffset(False)
        axis.set_major_formatter(formatter)


def set_suptitle(
    fig,
    title: str,
    *,
    y: float = 0.985,
    fontsize: float = 16.0,
) -> None:
    """Set a compact figure title with consistent vertical placement."""
    fig.suptitle(title, y=min(y, 0.98), fontsize=min(fontsize, 10.0))


def load_parity_npz(stem: str) -> dict[str, np.ndarray]:
    """Load an extracted MATLAB parity NPZ bundle by stem."""
    path = PARITY_NPZ_DIR / f"{stem}.npz"
    if not path.exists():
        legacy_path = LEGACY_PARITY_NPZ_DIR / f"{stem}.npz"
        if legacy_path.exists():
            path = legacy_path
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def parity_npz_get(bundle: dict[str, np.ndarray], key: str) -> np.ndarray:
    """Read a logical HDF5-style field path from a flattened parity bundle."""
    flat_key = key.replace("/", "__")
    if flat_key not in bundle:
        raise KeyError(f"Missing parity field {key!r} ({flat_key!r})")
    return bundle[flat_key]


def mesh_view(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    used_ind: np.ndarray | None = None,
) -> SimpleNamespace:
    """Create a lightweight mesh-like object for baseflow reloads."""
    flat_size = int(np.asarray(X).size)
    if used_ind is None:
        used_ind = np.arange(flat_size, dtype=int)
    return SimpleNamespace(
        X=np.asarray(X, dtype=float),
        Y=np.asarray(Y, dtype=float),
        used_ind=np.asarray(used_ind, dtype=int),
    )


def plot_signed_panel(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    aspect: str | float | None = "auto",
    levels: int = 81,
    colorbar: bool = True,
    robust_percentile: float | None = 99.5,
    colorbar_kwargs: dict | None = None,
    title_pad: float = 5.0,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot a signed field with a symmetric color scale."""
    from examples._plot_flow import plot_flow

    configure_example_style()
    contour = plot_flow(
        X,
        Y,
        field,
        ax=ax,
        cmap=signed_zero_white_cmap(),
        levels=levels,
        colorbar=colorbar,
        colorbar_kwargs=colorbar_kwargs,
        symmetric=True,
        robust_percentile=robust_percentile,
        aspect=aspect,
        xlim=xlim,
        ylim=ylim,
        return_contour=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, pad=title_pad)
    return contour


def plot_scalar_panel(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    aspect: str | float | None = "auto",
    levels: int = 81,
    colorbar: bool = True,
    colorbar_kwargs: dict | None = None,
    title_pad: float = 5.0,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot a non-signed scalar field with standard example defaults."""
    from examples._plot_flow import plot_flow

    configure_example_style()
    contour = plot_flow(
        X,
        Y,
        field,
        ax=ax,
        cmap=cmap,
        levels=levels,
        colorbar=colorbar,
        colorbar_kwargs=colorbar_kwargs,
        aspect=aspect,
        xlim=xlim,
        ylim=ylim,
        return_contour=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, pad=title_pad)
    return contour


def plot_gain_bars(
    ax,
    gains: np.ndarray,
    *,
    ylabel: str = r"$\sigma_i$",
    matlab_gains: np.ndarray | None = None,
    matlab_label: str = "MATLAB",
) -> None:
    """Plot a compact gain bar chart."""
    configure_example_style()
    values = np.asarray(gains, dtype=float).reshape(-1)
    indices = np.arange(1, values.size + 1, dtype=int)
    ax.bar(indices, values, color="#a8add5", edgecolor="#394264", linewidth=0.9, label="Python")
    if matlab_gains is not None:
        ref = np.asarray(matlab_gains, dtype=float).reshape(-1)[: values.size]
        ax.plot(
            indices,
            ref,
            linestyle="none",
            marker="D",
            ms=6,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            label=matlab_label,
            zorder=5,
        )
    ax.set_xticks(indices)
    ax.set_xlabel(r"Mode number $i$")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    if matlab_gains is not None:
        ax.legend(loc="upper right", frameon=False, fontsize=9)


def top_imag_indices(values: np.ndarray, n: int) -> np.ndarray:
    """Return indices of the ``n`` largest imaginary parts."""
    arr = np.asarray(values, dtype=complex).reshape(-1)
    return np.argsort(np.imag(arr))[::-1][: min(n, arr.size)]


def smallest_abs_indices(values: np.ndarray, n: int) -> np.ndarray:
    """Return indices of the ``n`` smallest magnitudes."""
    arr = np.asarray(values, dtype=complex).reshape(-1)
    return np.argsort(np.abs(arr))[: min(n, arr.size)]


def hide_unused_axes(axes, used: int) -> None:
    """Hide axes in a flattened matplotlib axes collection after ``used``."""
    flat = np.ravel(axes)
    for ax in flat[used:]:
        ax.set_visible(False)


def format_complex(
    value: complex,
    *,
    real_precision: int = 3,
    imag_precision: int | None = None,
) -> str:
    """Format a complex scalar as ``a + bi`` with stable precision."""
    z = complex(value)
    imag_precision = real_precision if imag_precision is None else imag_precision
    real = f"{z.real:.{real_precision}f}"
    imag = f"{abs(z.imag):.{imag_precision}f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{real} {sign} {imag}i"
