"""Loader for Hamada turbine-side case arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .baseflow import BaseFlow
from .sponge import case1_sponge_from_config


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


def _load_sponge_aux(case_inputs_mat: Path) -> np.ndarray:
    """Load MATLAB ``sponge_aux`` field from a case-inputs .mat file."""
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MATLAB sponge_aux loading requires optional dependency 'h5py'. "
            "Install the 'linstabpy[hdf5]' extra or rerun install.sh with "
            "LINSTAB2D_WITH_H5PY=1."
        ) from exc
    with h5py.File(case_inputs_mat, "r") as handle:
        if "sponge_aux" not in handle:
            raise KeyError(f"'sponge_aux' not found in {case_inputs_mat}")
        sponge = np.asarray(handle["sponge_aux"], dtype=float)
    if sponge.ndim != 2:
        raise ValueError(f"sponge_aux in {case_inputs_mat} must be 2-D, got shape {sponge.shape}")
    return sponge


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
    sponge_mat_path: str | Path | None = None,
    prefer_matlab_sponge_aux: bool = False,
    native_sponge_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, BaseFlow, np.ndarray]:
    """Load Hamada case arrays and build a ``BaseFlow`` object."""
    root = Path(case_dir)
    fields = {
        name: _load_field(root, name)
        for name in ("x", "y", "rho", "u", "v", "w", "T", "sponge")
    }

    if transpose_input:
        fields = {name: arr.T for name, arr in fields.items()}

    if native_sponge_cfg is not None:
        fields["sponge"] = case1_sponge_from_config(
            fields["sponge"].shape,
            native_sponge_cfg,
            X=fields["x"],
            Y=fields["y"],
        )
    elif prefer_matlab_sponge_aux and sponge_mat_path is not None:
        sponge_aux = _load_sponge_aux(Path(sponge_mat_path))
        if sponge_aux.shape != fields["sponge"].shape:
            if sponge_aux.T.shape == fields["sponge"].shape:
                sponge_aux = sponge_aux.T
            else:
                raise ValueError(
                    "sponge_aux shape mismatch: "
                    f"{sponge_aux.shape} (or transpose {sponge_aux.T.shape}) "
                    f"vs field shape {fields['sponge'].shape}"
                )
        fields["sponge"] = sponge_aux

    sx = _validate_stride(stride_ksi, "stride_ksi")
    sy = _validate_stride(stride_eta, "stride_eta")
    # Naming caveat: after transpose_input=True the post-load arrays are
    # (ny, nx) in Mesh convention, i.e. (η, ξ). The stride application
    # below therefore decimates axis 0 (η, wall-normal) by stride_ksi and
    # axis 1 (ξ, streamwise) by stride_eta — the opposite of what the
    # parameter names literally suggest. For the Case 2 curved turbine
    # suction-side mesh we only use symmetric strides (sx == sy), so the
    # naming mismatch is observationally invisible. This assertion guards
    # against future asymmetric use on this loader, where the result
    # would be a location-dependent physical cell aspect ratio (not a
    # natural coarsening of a body-fitted domain); see
    # `studies/case2/PLAN.md` (F2, I1) and `worklog.md` for the record.
    if sx != sy:
        raise ValueError(
            f"Asymmetric strides are not supported by load_hamada_case "
            f"(stride_ksi={sx}, stride_eta={sy}). In the curved body-fitted "
            "mesh this produces a location-dependent cell aspect ratio, "
            "which is not a natural coarsening of the physical problem. "
            "Use symmetric strides only (e.g. stride_ksi == stride_eta)."
        )
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
