# SPDX-License-Identifier: GPL-3.0-only
"""Load a rectilinear primitive field set from a npy directory or one npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .baseflow import BaseFlow

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
    """Build coordinates, a BaseFlow, and the sponge from a field set.

    ``source`` is either a directory of eight ``.npy`` files or one ``.npz``
    with the same keys. Asymmetric strides are allowed here. The Hamada
    wrapper still refuses them on the curved turbine mesh.
    """
    fields = read_grid_fields(source)
    if transpose_input:
        fields = {name: arr.T for name, arr in fields.items()}

    sx = int(stride_ksi)
    sy = int(stride_eta)
    if sx < 1 or sy < 1:
        raise ValueError("stride_ksi and stride_eta must be >= 1")
    if sx > 1 or sy > 1:
        # Arrays are (ny, nx) after an optional transpose. stride_eta
        # walks the first index; stride_ksi walks the second.
        fields = {name: arr[::sy, ::sx] for name, arr in fields.items()}

    shape = fields["x"].shape
    for name, arr in fields.items():
        if arr.shape != shape:
            raise ValueError(f"Field '{name}' shape {arr.shape} does not match x shape {shape}")

    spanwise = np.zeros_like(fields["u"]) if zero_spanwise else fields["w"]
    baseflow = BaseFlow(
        Re=float(Re),
        Ma=float(Ma),
        Pr=float(Pr),
        kappa=float(kappa),
        T_0=float(T_0),
        RHO=fields["rho"],
        U=fields["u"],
        V=fields["v"],
        W=spanwise,
        T=fields["T"],
    )
    return fields["x"], fields["y"], baseflow, fields["sponge"]
