from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from linstab2d.baseflow import BaseFlow
from linstab2d.baseflow import sutherland_air


def load_my_case(
    case_dir: str | Path,
    *,
    Re: float,
    Ma: float,
    Pr: float,
    kappa: float,
    T_0: float,
    stride_ksi: int = 1,
    stride_eta: int = 1,
    transpose_input: bool = False,
    zero_spanwise: bool = False,
) -> Tuple[np.ndarray, np.ndarray, BaseFlow, np.ndarray]:
    """
    Generic loader for user-supplied baseflows stored as .npy files.

    For the Ziming case, we expect all .npy fields to live in the
    ``baseflow`` subdirectory inside ``case_dir``:

        case_dir/
            baseflow/
                x.npy, y.npy
                rho.npy, u.npy, v.npy, w.npy, T.npy
                sponge.npy

    All arrays must be 2-D with the same shape.
    """
    root = Path(case_dir) / "baseflow"

    def _load(name: str) -> np.ndarray:
        path = root / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing field file: {path}")
        arr = np.asarray(np.load(path), dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{path} must be 2-D, got shape {arr.shape}")
        return arr

    X = _load("x")
    Y = _load("y")
    rho = _load("rho")
    u = _load("u")
    v = _load("v")
    w = _load("w")
    T = _load("T")
    sponge = _load("sponge")

    if transpose_input:
        X, Y, rho, u, v, w, T, sponge = [a.T for a in (X, Y, rho, u, v, w, T, sponge)]

    if stride_ksi < 1 or stride_eta < 1:
        raise ValueError("stride_ksi and stride_eta must be >= 1")

    if stride_ksi > 1 or stride_eta > 1:
        X = X[::stride_ksi, ::stride_eta]
        Y = Y[::stride_ksi, ::stride_eta]
        rho = rho[::stride_ksi, ::stride_eta]
        u = u[::stride_ksi, ::stride_eta]
        v = v[::stride_ksi, ::stride_eta]
        w = w[::stride_ksi, ::stride_eta]
        T = T[::stride_ksi, ::stride_eta]
        sponge = sponge[::stride_ksi, ::stride_eta]

    if zero_spanwise:
        w = np.zeros_like(u)

    shape = X.shape
    for name, arr in {
        "Y": Y,
        "rho": rho,
        "u": u,
        "v": v,
        "w": w,
        "T": T,
        "sponge": sponge,
    }.items():
        if arr.shape != shape:
            raise ValueError(f"Field '{name}' shape {arr.shape} does not match x shape {shape}")

    baseflow = BaseFlow(
        Re=float(Re),
        Ma=float(Ma),
        Pr=float(Pr),
        kappa=float(kappa),
        T_0=float(T_0),
        RHO=rho,
        U=u,
        V=v,
        W=w,
        T=T,
    )

    # Populate viscosity and derived transport coefficients
    sutherland_air(baseflow)

    return X, Y, baseflow, sponge

