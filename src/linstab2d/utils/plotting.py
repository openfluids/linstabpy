"""Plotting helpers for field visualization."""

from __future__ import annotations

import numpy as np


def plot_flow(
    X: np.ndarray,
    Y: np.ndarray,
    field: np.ndarray,
    *,
    ax=None,
    component: str = "real",
    cmap: str = "RdBu_r",
    levels: int = 51,
    colorbar: bool = True,
    colorbar_kwargs: dict | None = None,
    symmetric: bool = False,
    robust_percentile: float | None = None,
    aspect: str | float | None = "equal",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    return_contour: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
):
    """Plot a scalar field as a filled contour with matching extent metadata.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate mesh arrays.
    field : ndarray
        Scalar field array of shape ``(ny, nx)``.
    ax : matplotlib axis, optional
        Existing axis to draw on (supports subplots).
    component : {'real','imag','abs','power'}
        Which field component to plot.
    cmap : str
        Matplotlib colormap name.
    levels : int
        Number of contour levels.
    colorbar : bool
        Whether to add a colorbar.
    colorbar_kwargs : dict, optional
        Extra kwargs for :func:`matplotlib.figure.Figure.colorbar`.
    symmetric : bool
        If true, use a zero-centered symmetric color scale.
    robust_percentile : float, optional
        Percentile used to clip the color scale. ``99`` keeps the central 99%
        of the finite values. When omitted, full data limits are used.
    aspect : {'equal', 'auto'} or float or None
        Axes aspect ratio. ``None`` leaves the current aspect unchanged.
    xlim, ylim : tuple, optional
        Axis limits applied after plotting.
    return_contour : bool
        If true, return the contour/pcolormesh object instead of the axis.
    vmin, vmax : float, optional
        Explicit color limits. Ignored when ``symmetric`` is true and both are
        omitted.

    Returns
    -------
    ax
        The axis used to draw the plot.
    """
    import matplotlib.pyplot as plt

    if colorbar_kwargs is None:
        colorbar_kwargs = {}

    data = np.asarray(field)
    if data.shape != np.asarray(X).shape:
        raise ValueError("field shape must match X and Y")

    if component == "real":
        plot_data = np.real(data)
    elif component == "imag":
        plot_data = np.imag(data)
    elif component == "abs":
        plot_data = np.abs(data)
    elif component == "power":
        plot_data = np.abs(data) ** 2
    else:
        raise ValueError("component must be one of 'real', 'imag', 'abs', or 'power'")

    if ax is None:
        _, ax = plt.subplots(1, 1)

    level_spec, vmin, vmax = _resolve_color_scale(
        plot_data,
        levels=levels,
        symmetric=symmetric,
        robust_percentile=robust_percentile,
        vmin=vmin,
        vmax=vmax,
    )

    has_nan = np.any(~np.isfinite(plot_data))

    render_kwargs = dict(kwargs)
    render_kwargs.pop("levels", None)
    if symmetric and "extend" not in render_kwargs:
        render_kwargs["extend"] = "both"

    if has_nan:
        contour = _tricontourf_masked(
            ax,
            X,
            Y,
            plot_data,
            levels=level_spec,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **render_kwargs,
        )
    else:
        try:
            contour = ax.contourf(
                X,
                Y,
                plot_data,
                levels=level_spec,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **render_kwargs,
            )
        except Exception:
            fallback_kwargs = render_kwargs.copy()
            fallback_kwargs.pop("shading", None)
            if "shading" not in fallback_kwargs:
                fallback_kwargs["shading"] = "auto"
            contour = ax.pcolormesh(
                X,
                Y,
                plot_data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **fallback_kwargs,
            )

    if aspect is not None:
        ax.set_aspect(aspect, adjustable="box")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if colorbar:
        cbar = ax.figure.colorbar(contour, ax=ax, **colorbar_kwargs)
        try:
            from matplotlib.ticker import MaxNLocator

            cbar.locator = MaxNLocator(nbins=6)
            cbar.update_ticks()
        except Exception:
            pass
    if return_contour:
        return contour
    return ax


def _resolve_color_scale(
    plot_data: np.ndarray,
    *,
    levels: int | np.ndarray,
    symmetric: bool,
    robust_percentile: float | None,
    vmin: float | None,
    vmax: float | None,
) -> tuple[int | np.ndarray, float | None, float | None]:
    """Resolve contour levels and color limits for a scalar field."""
    finite = np.asarray(plot_data, dtype=float)[np.isfinite(plot_data)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=float)

    use_explicit_scale = symmetric or robust_percentile is not None or vmin is not None or vmax is not None
    if not use_explicit_scale:
        return levels, vmin, vmax

    if symmetric:
        if robust_percentile is not None:
            limit = float(np.nanpercentile(np.abs(finite), robust_percentile))
        else:
            limit = float(np.max(np.abs(finite)))
        limit = max(limit, 1e-12)
        vmin = -limit
        vmax = limit
    else:
        if vmin is None or vmax is None:
            if robust_percentile is not None:
                lo = float(np.nanpercentile(finite, 100.0 - robust_percentile))
                hi = float(np.nanpercentile(finite, robust_percentile))
            else:
                lo = float(np.min(finite))
                hi = float(np.max(finite))
            if vmin is None:
                vmin = lo
            if vmax is None:
                vmax = hi
        if vmin is None or vmax is None:
            raise ValueError("vmin/vmax resolution failed")

    if np.isclose(vmin, vmax):
        delta = 1e-12 if vmin == 0.0 else max(abs(vmin), abs(vmax)) * 1e-6
        vmin -= delta
        vmax += delta

    if isinstance(levels, np.ndarray):
        return levels, vmin, vmax
    return np.linspace(vmin, vmax, max(int(levels), 3)), vmin, vmax


def _tricontourf_masked(
    ax,
    X,
    Y,
    plot_data,
    *,
    levels=51,
    cmap="RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
):
    """Render a NaN-holed structured field using grid-based triangulation.

    Builds triangles directly from the (ny, nx) grid adjacency: each quad
    cell whose 4 corners are all finite is split into 2 triangles.  This
    naturally excludes the NaN hole with no heuristic parameters.
    """
    import matplotlib.tri as mtri

    valid = np.isfinite(np.asarray(plot_data))
    ny, nx = valid.shape

    # Map (j, i) grid positions to contiguous vertex indices (-1 = invalid).
    vertex_idx = np.full((ny, nx), -1, dtype=int)
    vertex_idx[valid] = np.arange(valid.sum())

    # Build triangles from valid quad cells.
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = vertex_idx[j, i]
            v01 = vertex_idx[j, i + 1]
            v10 = vertex_idx[j + 1, i]
            v11 = vertex_idx[j + 1, i + 1]
            if v00 >= 0 and v01 >= 0 and v10 >= 0 and v11 >= 0:
                triangles.append((v00, v01, v10))
                triangles.append((v01, v11, v10))

    x_v = np.asarray(X)[valid]
    y_v = np.asarray(Y)[valid]
    z_v = np.asarray(plot_data)[valid]
    tri = mtri.Triangulation(x_v, y_v, triangles=np.array(triangles, dtype=int))

    return ax.tricontourf(tri, z_v, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
