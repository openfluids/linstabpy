"""Shared helpers for example scripts.

Environment-variable parsing, grid reconstruction, and other utilities
used across multiple examples.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FIGURE_DPI = 600
REPO_ROOT = Path(__file__).resolve().parents[1]
PARITY_NPZ_DIR = REPO_ROOT / "tools" / "parity" / "npz"


def env_bool(name: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """Read an environment variable as a positive integer."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        value = int(v)
    except (TypeError, ValueError):
        print(f"Invalid {name}={v!r}; using {default}.")
        return default
    return max(1, value)


def env_float(name: str, default: float) -> float:
    """Read an environment variable as a float."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        print(f"Invalid {name}={v!r}; using {default}.")
        return default


def env_complex(name: str, default: complex) -> complex:
    """Read an environment variable as a complex number."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return complex(v)
    except (TypeError, ValueError):
        print(f"Invalid {name}={v!r}; using {default}.")
        return default


def load_jsonc(path: str | Path) -> dict:
    """Load a JSONC file (JSON with // and /* */ comments)."""
    text = Path(path).read_text(encoding="utf-8")
    cleaned = _strip_jsonc_comments(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSONC object must be a dict: {path}")
    return data


def _strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving string literals."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    in_line_comment = False
    in_block_comment = False
    escape = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


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


def save_figure(fig, path: str | Path) -> None:
    """Save a matplotlib figure with the repo-wide example defaults."""
    import matplotlib.pyplot as plt

    output = Path(path)
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output}")


def load_parity_npz(stem: str) -> dict[str, np.ndarray]:
    """Load an extracted MATLAB parity NPZ bundle by stem."""
    path = PARITY_NPZ_DIR / f"{stem}.npz"
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
):
    """Plot a signed field with a symmetric color scale."""
    from linstab2d.utils.plotting import plot_flow

    contour = plot_flow(
        X,
        Y,
        field,
        ax=ax,
        cmap="RdBu_r",
        levels=levels,
        colorbar=colorbar,
        symmetric=True,
        robust_percentile=robust_percentile,
        aspect=aspect,
        xlim=xlim,
        ylim=ylim,
        return_contour=True,
    )
    ax.set_title(title)
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
):
    """Plot a non-signed scalar field with standard example defaults."""
    from linstab2d.utils.plotting import plot_flow

    contour = plot_flow(
        X,
        Y,
        field,
        ax=ax,
        cmap=cmap,
        levels=levels,
        colorbar=colorbar,
        aspect=aspect,
        xlim=xlim,
        ylim=ylim,
        return_contour=True,
    )
    ax.set_title(title)
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
