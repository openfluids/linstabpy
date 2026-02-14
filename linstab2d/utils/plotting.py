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
    contour = ax.contourf(X, Y, plot_data, levels=levels, cmap=cmap, **kwargs)
    if colorbar:
        ax.figure.colorbar(contour, ax=ax, **colorbar_kwargs)
    return ax
