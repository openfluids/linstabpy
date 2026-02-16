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

    has_nan = np.any(~np.isfinite(plot_data))

    render_kwargs = dict(kwargs)
    render_kwargs.pop("levels", None)

    if has_nan:
        contour = _tricontourf_masked(
            ax, X, Y, plot_data, levels=levels, cmap=cmap, **render_kwargs,
        )
    else:
        try:
            contour = ax.contourf(
                X, Y, plot_data, levels=levels, cmap=cmap, **render_kwargs,
            )
        except Exception:
            fallback_kwargs = render_kwargs.copy()
            fallback_kwargs.pop("shading", None)
            if "shading" not in fallback_kwargs:
                fallback_kwargs["shading"] = "auto"
            contour = ax.pcolormesh(
                X, Y, plot_data, cmap=cmap, **fallback_kwargs,
            )

    ax.set_aspect("equal", adjustable="box")
    if colorbar:
        ax.figure.colorbar(contour, ax=ax, **colorbar_kwargs)
    return ax


def _tricontourf_masked(ax, X, Y, plot_data, *, levels=51, cmap="RdBu_r", **kwargs):
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

    return ax.tricontourf(tri, z_v, levels=levels, cmap=cmap, **kwargs)
