"""Loader for Hamada turbine-side case arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .baseflow import BaseFlow


def _load_field(case_dir: Path, name: str) -> np.ndarray:
    path = case_dir / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing Hamada case file: {path}")
    field = np.asarray(np.load(path), dtype=float)
    if field.ndim != 2:
        raise ValueError(f"{path} must be 2-D, got shape {field.shape}")
    return field


def _validate_stride(value: int, name: str) -> int:
    stride = int(value)
    if stride < 1:
        raise ValueError(f"{name} must be >= 1")
    return stride


def load_hamada_case(
    case_dir: str | Path,
    *,
    Re: float = 395000.0,
    Ma: float = 2.0,
    Pr: float = 0.747,
    kappa: float = 1.31,
    T_0: float = 1540.0,
    stride_ksi: int = 1,
    stride_eta: int = 1,
    transpose_input: bool = True,
    zero_spanwise: bool = False,
) -> tuple[np.ndarray, np.ndarray, BaseFlow, np.ndarray]:
    """Load Hamada case arrays and build a ``BaseFlow`` object."""
    root = Path(case_dir)
    fields = {
        name: _load_field(root, name)
        for name in ("x", "y", "rho", "u", "v", "w", "T", "sponge")
    }

    if transpose_input:
        fields = {name: arr.T for name, arr in fields.items()}

    sx = _validate_stride(stride_ksi, "stride_ksi")
    sy = _validate_stride(stride_eta, "stride_eta")
    if sx > 1 or sy > 1:
        fields = {name: arr[::sx, ::sy] for name, arr in fields.items()}

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
