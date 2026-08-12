# SPDX-License-Identifier: GPL-3.0-only
"""Load a rectilinear primitive field set from a npy directory or one npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from typing import Any

from .baseflow import BaseFlow
from .preprocess import apply_preprocess

FIELD_NAMES = ("x", "y", "rho", "u", "v", "w", "T", "sponge")


def _as_field(name: str, arr: np.ndarray, source: Path) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{source} field {name} must be 2-D, got shape {out.shape}")
    return out


def read_grid_fields(source: str | Path) -> dict[str, np.ndarray]:
    """Read x, y, rho, u, v, w, T, sponge from an npz file or a npy directory."""
    root = Path(source)
    if root.is_file():
        if root.suffix != ".npz":
            raise ValueError(f"grid field file must be .npz, got {root}")
        packed = np.load(root)
        missing = [name for name in FIELD_NAMES if name not in packed.files]
        if missing:
            raise FileNotFoundError(f"{root} is missing fields: {missing}")
        return {name: _as_field(name, packed[name], root) for name in FIELD_NAMES}

    if not root.is_dir():
        raise FileNotFoundError(f"grid field source not found: {root}")

    fields: dict[str, np.ndarray] = {}
    for name in FIELD_NAMES:
        path = root / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing field file: {path}")
        fields[name] = _as_field(name, np.load(path), path)
    return fields


def load_case_fields(
    source: str | Path,
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
    refuse_asymmetric_stride: bool = False,
    preprocess: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, BaseFlow, np.ndarray, dict[str, Any]]:
    """Build coordinates, a BaseFlow, and the sponge from a packed field.

    ``source`` is a directory of eight ``.npy`` files or one ``.npz`` with
    the same keys. After an optional transpose the arrays are ``(ny, nx)``:
    ``stride_eta`` walks the first index and ``stride_ksi`` walks the second.
    """
    fields = read_grid_fields(source)
    if transpose_input:
        fields = {name: arr.T for name, arr in fields.items()}

    sx = int(stride_ksi)
    sy = int(stride_eta)
    if sx < 1 or sy < 1:
        raise ValueError("stride_ksi and stride_eta must be >= 1")
    if refuse_asymmetric_stride and sx != sy:
        raise ValueError(
            f"Asymmetric strides are not supported on this mesh "
            f"(stride_ksi={sx}, stride_eta={sy})."
        )
    if sx > 1 or sy > 1:
        fields = {name: arr[::sy, ::sx] for name, arr in fields.items()}

    shape = fields["x"].shape
    for name, arr in fields.items():
        if arr.shape != shape:
            raise ValueError(f"Field '{name}' shape {arr.shape} does not match x shape {shape}")

    X = fields["x"]
    Y = fields["y"]
    rho = fields["rho"]
    u = fields["u"]
    v = fields["v"]
    w = fields["w"]
    T = fields["T"]
    sponge = fields["sponge"]

    X, Y, rho, u, v, w, T, sponge, info = apply_preprocess(
        X, Y, rho, u, v, w, T, sponge, preprocess
    )

    if zero_spanwise:
        w = np.zeros_like(u)

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
    return X, Y, baseflow, sponge, info


def load_grid_fields(
    source: str | Path,
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
) -> tuple[np.ndarray, np.ndarray, BaseFlow, np.ndarray]:
    """Load a packed field and drop the preprocess report."""
    X, Y, baseflow, sponge, _info = load_case_fields(
        source,
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        stride_ksi=stride_ksi,
        stride_eta=stride_eta,
        transpose_input=transpose_input,
        zero_spanwise=zero_spanwise,
    )
    return X, Y, baseflow, sponge
