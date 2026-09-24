# SPDX-License-Identifier: GPL-3.0-only
"""Martini example constructors from LinStab2D MATLAB field files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, RectBivariateSpline, RegularGridInterpolator, griddata
from scipy.io import loadmat

from linstabpy.baseflow import BaseFlow

_BASEFLOW_DATA_ROOT = Path(
    os.environ.get(
        "LINSTABPY_BASEFLOW_DATA_DIR",
        Path(__file__).resolve().parent / "baseflows",
    )
)
if (_BASEFLOW_DATA_ROOT / "BaseFlows").is_dir():
    _BASEFLOW_DATA_DIR = _BASEFLOW_DATA_ROOT / "BaseFlows"
else:
    _BASEFLOW_DATA_DIR = _BASEFLOW_DATA_ROOT


def _load_baseflow_data(filename: str) -> dict[str, np.ndarray]:
    """Load a MATLAB base-flow file from the configured data directory."""
    path = _BASEFLOW_DATA_DIR / filename
    if not path.exists():
        if not _BASEFLOW_DATA_DIR.is_dir():
            raise FileNotFoundError(
                f"Reference base-flow data not found at {_BASEFLOW_DATA_DIR}. "
                "Set LINSTABPY_BASEFLOW_DATA_DIR to a local clone of "
                "https://github.com/LinStab2D/LinStab2D, or place .mat files in "
                "examples/linstab2d/baseflows/."
            ) from FileNotFoundError(
                f"Could not find base-flow data directory: {_BASEFLOW_DATA_DIR}"
            )
        raise FileNotFoundError(f"Could not find base-flow data file: {path}")
    return loadmat(path)


def _as_grid(values: np.ndarray, shape: tuple[int, int], name: str | None = None) -> np.ndarray:
    """Reshape MATLAB column vectors or matrices into a 2-D mesh array."""
    arr = np.asarray(values, dtype=float)
    if arr.size != np.prod(shape):
        if name is None:
            name = "field"
        raise ValueError(f"{name} has incompatible size: expected {shape} elements, got {arr.size}")
    return arr.reshape(shape, order="F")


def _interp_spline2d(
    Xq: np.ndarray,
    Yq: np.ndarray,
    Xgrid: np.ndarray,
    Ygrid: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """Interpolate a field with tensor-product cubic splines."""
    x = np.asarray(Xgrid, dtype=float)[0, :]
    y = np.asarray(Ygrid, dtype=float)[:, 0]
    spline = RectBivariateSpline(y, x, np.asarray(field, dtype=float))
    return spline.ev(Yq.ravel(order="F"), Xq.ravel(order="F")).reshape(Xq.shape, order="F")


def _interp_regular_grid_linear(
    Xq: np.ndarray,
    Yq: np.ndarray,
    Xgrid: np.ndarray,
    Ygrid: np.ndarray,
    field: np.ndarray,
    used_ind: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate a regular-grid field linearly on mesh query points."""
    x = np.asarray(Xgrid, dtype=float)[0, :]
    y = np.asarray(Ygrid, dtype=float)[:, 0]
    interpolator = RegularGridInterpolator(
        (y, x),
        np.asarray(field, dtype=float),
        bounds_error=False,
        fill_value=np.nan,
        method="linear",
    )

    flat_out = np.full(Xq.size, np.nan, dtype=float)
    Xf = Xq.ravel(order="F")
    Yf = Yq.ravel(order="F")
    used = np.asarray(used_ind, dtype=int) if used_ind is not None else np.arange(Xf.size)
    points = np.column_stack((Yf[used], Xf[used]))
    flat_out[used] = interpolator(points)
    return flat_out.reshape(Xq.shape, order="F")


def _interp_scattered_cubic(
    Xq: np.ndarray,
    Yq: np.ndarray,
    Xpts: np.ndarray,
    Ypts: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """Interpolate a scattered field with cubic `griddata`."""
    points = np.column_stack(
        (np.asarray(Xpts, dtype=float).ravel(), np.asarray(Ypts, dtype=float).ravel())
    )
    values = np.asarray(field, dtype=float).ravel()
    query = np.column_stack((Xq.ravel(order="F"), Yq.ravel(order="F")))
    flat = griddata(points, values, query, method="cubic", fill_value=np.nan).reshape(
        Xq.shape, order="F"
    )
    if np.isnan(flat).any():
        nearest = griddata(points, values, query, method="nearest", fill_value=np.nan).reshape(
            Xq.shape, order="F"
        )
        flat = np.where(np.isnan(flat), nearest, flat)
    if np.isnan(flat).any() or np.isinf(flat).any():
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat


# ---------------------------------------------------------------------------
# Blasius tabulated solution (White, *Viscous Fluid Flow*, p. 233, Table 4-1)
# ---------------------------------------------------------------------------
_ETA_BLASIUS = np.array(
    [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
        3.0,
        3.2,
        3.4,
        3.6,
        3.8,
        4.0,
        4.2,
        4.4,
        4.6,
        4.8,
        5.0,
        5.2,
        5.4,
        5.6,
        5.8,
        6.0,
    ]
)

_F_BLASIUS = np.array(
    [
        0.00000,
        0.00235,
        0.00939,
        0.02113,
        0.03755,
        0.05864,
        0.08439,
        0.11474,
        0.14967,
        0.18911,
        0.23299,
        0.28121,
        0.33366,
        0.39021,
        0.45072,
        0.51503,
        0.58296,
        0.65430,
        0.72887,
        0.80644,
        0.88680,
        1.05495,
        1.23153,
        1.41482,
        1.60328,
        1.79557,
        1.99058,
        2.18747,
        2.38559,
        2.58450,
        2.78388,
        2.98355,
        3.18338,
        3.38329,
        3.58325,
        3.78323,
        3.98322,
        4.18322,
        4.38322,
        4.58322,
        4.78322,
    ]
)

_FPRIME_BLASIUS = np.array(
    [
        0.00000,
        0.04696,
        0.09391,
        0.14081,
        0.18761,
        0.23423,
        0.28058,
        0.32653,
        0.37196,
        0.41672,
        0.46063,
        0.50354,
        0.54525,
        0.58559,
        0.62439,
        0.66147,
        0.69670,
        0.72993,
        0.76106,
        0.79000,
        0.81669,
        0.86330,
        0.90107,
        0.93060,
        0.95288,
        0.96905,
        0.98037,
        0.98797,
        0.99289,
        0.99594,
        0.99777,
        0.99882,
        0.99940,
        0.99970,
        0.99986,
        0.99994,
        0.999971,
        0.999981,
        0.999995,
        0.999998,
        0.999999,
    ]
)

_FPRIMEPRIME_BLASIUS = np.array(
    [
        0.46960,
        0.46956,
        0.46931,
        0.46861,
        0.46725,
        0.46503,
        0.46173,
        0.45718,
        0.45119,
        0.44363,
        0.43438,
        0.42337,
        0.41057,
        0.39598,
        0.37969,
        0.36180,
        0.34249,
        0.32195,
        0.30045,
        0.27825,
        0.25567,
        0.21058,
        0.16756,
        0.12861,
        0.09511,
        0.06771,
        0.04637,
        0.03054,
        0.01933,
        0.01176,
        0.00687,
        0.00386,
        0.00208,
        0.00108,
        0.00054,
        0.00026,
        0.000119,
        0.000052,
        0.000022,
        0.000009,
        0.000003,
    ]
)


def _blasius_profiles(
    X: np.ndarray,
    Y: np.ndarray,
    Ma: float,
    Pr: float,
    nu: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute compressible Blasius boundary-layer profiles.

    Parameters
    ----------
    X, Y : np.ndarray
        2-D coordinate arrays of shape ``(ny, nx)``.
    Ma : float
        Freestream Mach number.
    Pr : float
        Prandtl number.
    nu : float
        Kinematic viscosity, ``1 / Re``.
    kappa : float
        Heat capacity ratio.

    Returns
    -------
    U, V, T, RHO : np.ndarray
        Streamwise velocity, wall-normal velocity, temperature, and density.
        Fields sit on the ``(ny, nx)`` grid.

    Notes
    -----
    Based on the compressible self-similar boundary-layer theory in White's
    *Viscous Fluid Flow* (section 7-3.1).
    Also based on Anderson's *Hypersonic and High-Temperature Gas Dynamics*.
    The similarity variable transformation maps the incompressible Blasius
    solution to a compressible one.
    """
    ny, nx = X.shape

    # -- Self-similar profiles ------------------------------------------
    uue = _FPRIME_BLASIUS.copy()
    TTe = 1.0 + np.sqrt(Pr) * (kappa - 1.0) / 2.0 * Ma**2 * (1.0 - uue**2)
    rhorhoe = 1.0 / TTe

    # Map compressible eta -> incompressible eta
    eta_incomp = cumulative_trapezoid(TTe, _ETA_BLASIUS, initial=0.0)

    # -- Sampling points ------------------------------------------------
    x_flat = X.ravel()
    y_flat = Y.ravel()
    eta_interp = y_flat / np.sqrt(2.0 * nu * np.maximum(x_flat, 1e-30))

    # -- Cubic spline interpolation with freestream clamping ------------
    # MATLAB: interp1(eta_incomp, profile, eta_interp, 'spline', 1)
    # Extrapolation beyond tabulated range -> freestream value = 1.
    cs_u = CubicSpline(eta_incomp, uue, extrapolate=True)
    cs_T = CubicSpline(eta_incomp, TTe, extrapolate=True)
    cs_rho = CubicSpline(eta_incomp, rhorhoe, extrapolate=True)

    uue_g = cs_u(eta_interp)
    TTe_g = cs_T(eta_interp)
    rhorhoe_g = cs_rho(eta_interp)

    # Clamp to freestream values for points beyond the table
    outside = eta_interp > eta_incomp[-1]
    uue_g[outside] = 1.0
    TTe_g[outside] = 1.0
    rhorhoe_g[outside] = 1.0

    # Reshape to grid
    uue_g = uue_g.reshape(ny, nx)
    TTe_g = TTe_g.reshape(ny, nx)
    rhorhoe_g = rhorhoe_g.reshape(ny, nx)

    # -- Wall-normal velocity from compressible continuity --------------
    rhorhoeuue = rhorhoe_g * uue_g

    # d/dx(rho*u) along each row
    ddx_rhorhoeuue = np.empty_like(rhorhoeuue)
    for j in range(ny):
        ddx_rhorhoeuue[j, :] = np.gradient(rhorhoeuue[j, :], X[0, :])

    # rho*v = -integral_0^y d(rho*u)/dx dy   (column by column)
    rhorhoevue = np.empty_like(rhorhoeuue)
    for i in range(nx):
        rhorhoevue[:, i] = -cumulative_trapezoid(
            ddx_rhorhoeuue[:, i],
            Y[:, 0],
            initial=0.0,
        )

    vue_g = rhorhoevue / rhorhoe_g

    return uue_g, vue_g, TTe_g, rhorhoe_g


def load_blasius(
    X: np.ndarray,
    Y: np.ndarray,
    Re: float,
    Ma: float,
    Pr: float,
    kappa: float,
    T_0: float,
) -> BaseFlow:
    """Load a compressible Blasius boundary-layer base flow.

    Translates ``example_1_readbaseflow.m`` from the MATLAB codebase.

    Parameters
    ----------
    X, Y : np.ndarray
        2-D coordinate arrays of shape ``(ny, nx)``.
    Re : float
        Reynolds number.
    Ma : float
        Freestream Mach number.
    Pr : float
        Prandtl number.
    kappa : float
        Heat capacity ratio.
    T_0 : float
        Reference temperature [K].

    Returns
    -------
    BaseFlow
        Fully populated base flow (except for viscosity fields, which are
        set by :func:`~linstabpy.baseflow.viscosity.sutherland_air`).
    """
    nu = 1.0 / Re

    U, V, T, RHO = _blasius_profiles(X, Y, Ma, Pr, nu, kappa)
    W = np.zeros_like(X)

    return BaseFlow(
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        RHO=RHO,
        U=U,
        V=V,
        W=W,
        T=T,
    )


def load_jet(mesh) -> BaseFlow:
    """Load `M09jet.mat` (example 2: round jet base flow).

    Parameters
    ----------
    mesh : object
        Mesh containing ``X`` and ``Y`` coordinate arrays.

    Returns
    -------
    BaseFlow
        Base flow fields with axial/normal/spanwise velocity components arranged
        for Cartesian notation.
    """
    data = _load_baseflow_data("M09jet.mat")
    r_1D = np.asarray(data["r_1D"], dtype=float).ravel()
    z_1D = np.asarray(data["z_1D"], dtype=float).ravel()
    shape = (r_1D.size, z_1D.size)

    Xq = np.asarray(mesh.X, dtype=float)
    Yq = np.asarray(mesh.Y, dtype=float)
    X_data, Y_data = np.meshgrid(z_1D, r_1D, indexing="xy")

    def _load_field(name: str) -> np.ndarray:
        full = _as_grid(data[name], shape, name)
        if tuple(mesh.X.shape) == shape:
            return full
        return _interp_regular_grid_linear(
            Xq,
            Yq,
            X_data,
            Y_data,
            full,
        )

    RHO = _load_field("RHO")
    T = _load_field("T")
    # MATLAB base-flow file stores [radial, azimuthal, axial] as [U, V, W].
    # LinStab2D Python baseflow keeps [streamwise, radial, azimuthal].
    U_axial = _load_field("W")
    V_radial = _load_field("U")
    W_azimuthal = _load_field("V")

    return BaseFlow(
        Re=1.0e4,
        Ma=0.9,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=RHO,
        U=U_axial,
        V=V_radial,
        W=W_azimuthal,
        T=T,
    )


def load_streaky_bl(mesh) -> BaseFlow:
    """Load `BL_BF.mat` (example 3: streaky boundary layer).

    Parameters
    ----------
    mesh : object
        Mesh containing ``X`` and ``Y`` coordinate arrays.

    Returns
    -------
    BaseFlow
        Base flow interpolated to the provided mesh using cubic spline
        interpolation in both coordinates.
    """
    data = _load_baseflow_data("BL_BF.mat")

    X = np.asarray(data["x"], dtype=float)
    Y = np.asarray(data["y"], dtype=float)
    Xq = np.asarray(mesh.X, dtype=float)
    Yq = np.asarray(mesh.Y, dtype=float)

    # MATLAB BL_BF.mat stores the streak components as [spanwise, wall-normal,
    # streamwise] in [U, V, W]. Python BaseFlow uses [streamwise, wall-normal,
    # spanwise], so this loader remaps the components here.
    U = _interp_spline2d(Xq, Yq, X, Y, data["W"])
    V = _interp_spline2d(Xq, Yq, X, Y, data["V"])
    W = _interp_spline2d(Xq, Yq, X, Y, data["U"])
    RHO = _interp_spline2d(Xq, Yq, X, Y, data["rho"])
    T = _interp_spline2d(Xq, Yq, X, Y, data["T"])
    MU = _interp_spline2d(Xq, Yq, X, Y, data["mu"])
    dmudT = _interp_spline2d(Xq, Yq, X, Y, data["dmudT"])
    d2mudT2 = _interp_spline2d(Xq, Yq, X, Y, data["d2mudT2"])

    return BaseFlow(
        Re=28000.0,
        Ma=6.0,
        Pr=0.7,
        kappa=1.4,
        T_0=273.15,
        RHO=RHO,
        U=U,
        V=V,
        W=W,
        T=T,
        MU=MU,
        dmudT=dmudT,
        d2mudT2=d2mudT2,
    )


def load_cavity(mesh) -> BaseFlow:
    """Load `Step_BF.mat` (example 4: cavity base flow).

    Parameters
    ----------
    mesh : object
        Mesh containing ``X``, ``Y`` and ``used_ind`` arrays.

    Returns
    -------
    BaseFlow
        Base flow interpolated on ``mesh.used_ind`` using linear interpolation.
    """
    data = _load_baseflow_data("Step_BF.mat")

    Xq = np.asarray(mesh.X, dtype=float)
    Yq = np.asarray(mesh.Y, dtype=float)
    used_ind = np.asarray(mesh.used_ind, dtype=int)

    X_data = np.asarray(data["X"], dtype=float)
    Y_data = np.asarray(data["Y"], dtype=float)

    RHO = _interp_regular_grid_linear(Xq, Yq, X_data, Y_data, data["R_mean"], used_ind=used_ind)
    T = _interp_regular_grid_linear(Xq, Yq, X_data, Y_data, data["T_mean"], used_ind=used_ind)
    U = _interp_regular_grid_linear(Xq, Yq, X_data, Y_data, data["U_mean"], used_ind=used_ind)
    V = _interp_regular_grid_linear(Xq, Yq, X_data, Y_data, data["V_mean"], used_ind=used_ind)
    W = _interp_regular_grid_linear(Xq, Yq, X_data, Y_data, data["W_mean"], used_ind=used_ind)
    MU = np.ones_like(RHO, dtype=float)

    fp = None
    flow_parameters = data.get("flowParameters")
    if flow_parameters is not None:
        fp_container = np.asarray(flow_parameters).reshape(-1)
        if fp_container.size:
            fp = fp_container[0]
    if fp is not None:
        Re = float(np.asarray(fp["Re"]).reshape(-1)[0])
        Ma = float(np.asarray(fp["Ma"]).reshape(-1)[0])
        Pr = float(np.asarray(fp["Pr"]).reshape(-1)[0])
        kappa = float(np.asarray(fp["gamma"]).reshape(-1)[0])
        T_0 = float(np.asarray(fp["T0"]).reshape(-1)[0])
    else:
        Re = 1500.0
        Ma = 0.5
        Pr = 0.7
        kappa = 1.4
        T_0 = 300.0

    return BaseFlow(
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        RHO=RHO,
        U=U,
        V=V,
        W=W,
        T=T,
        MU=MU,
    )


def load_paraboloid(mesh) -> BaseFlow:
    """Load `Paraboloid_BF.mat` (example 5: parabolic geometry base flow).

    Parameters
    ----------
    mesh : object
        Mesh containing ``X`` and ``Y`` coordinate arrays.

    Returns
    -------
    BaseFlow
        Base flow interpolated to the mesh with cubic scattered interpolation.
    """
    data = _load_baseflow_data("Paraboloid_BF.mat")

    Xq = np.asarray(mesh.X, dtype=float)
    Yq = np.asarray(mesh.Y, dtype=float)

    U = _interp_scattered_cubic(
        Xq,
        Yq,
        data["x_BF"],
        data["y_BF"],
        data["U_BF"],
    )
    V = _interp_scattered_cubic(
        Xq,
        Yq,
        data["x_BF"],
        data["y_BF"],
        data["V_BF"],
    )
    RHO = np.ones_like(Xq, dtype=float)
    T = np.ones_like(Xq, dtype=float)
    W = np.zeros_like(Xq, dtype=float)
    MU = np.ones_like(Xq, dtype=float)

    return BaseFlow(
        Re=5e3,
        Ma=0.01,
        Pr=0.7,
        kappa=1.4,
        T_0=293.15,
        RHO=RHO,
        U=U,
        V=V,
        W=W,
        T=T,
        MU=MU,
    )
