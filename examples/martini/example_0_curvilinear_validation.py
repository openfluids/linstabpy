#!/usr/bin/env python3
"""
Example 0: Curvilinear derivative validation.

Translates `example_0_TestCurvilinearDerivative.m` directly:
- build the symmetric computational mesh
- deform it to the parabolic physical domain
- evaluate `f = cos(x)` and `df = Dx @ f`

OUTPUTS (``examples/martini/``):
  - example_0_curvilinear_validation.npz
  - example_0_curvilinear_mesh.png
  - example_0_curvilinear_validation.png
  - example_0_field_df_triptych.png  (LinStab2D fixture under LinStab2DResults/ vs Python NPZ)

USAGE:   python example_0_curvilinear_validation.py
         rm example_0_curvilinear_validation.npz && python example_0_curvilinear_validation.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from linstabpy.mesh import Mesh
from linstabpy.utils.yaml_io import load_config

from examples._helpers import load_or_compute
from examples.martini.example_0_plots import plot_example0_figures

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
ARTIFACT_JSON = SCRIPT.parents[2] / "artifacts" / "curvilinear_derivative_parity.json"


def _build_computational_mesh(cfg: dict) -> Mesh:
    """Build the Example 0 computational mesh exactly as in MATLAB."""
    return Mesh.create(
        xrange=tuple(cfg["xrange"]),
        yrange=tuple(cfg["yrange"]),
        nx=cfg["nx"],
        ny=cfg["ny"],
        fd_order=cfg["fd_order"],
        y_symmetry=cfg["y_symmetry"],
        x_periodicity=cfg["x_periodicity"],
        alpha_filter=cfg["alpha_filter"],
    )


def _build_reference_mesh(cfg: dict) -> Mesh:
    """Build the Example 0 curvilinear mesh exactly as in MATLAB."""
    cmesh = _build_computational_mesh(cfg)
    d = cfg["conformal_scale"]
    x = cmesh.X * d - 0.5
    z = x + 1j * cmesh.Y
    X = -np.real(z**2)
    Y = -np.imag(z**2)
    return cmesh.deform(X_new=X, Y_new=Y)


def _compute_reference_fields(cfg: dict) -> dict[str, np.ndarray | float | bool]:
    """Compute the MATLAB-like Example 0 fields and export orientation."""
    mesh = _build_reference_mesh(cfg)
    x = mesh.X
    y = mesh.Y

    f = np.cos(x)
    df_an = -np.sin(x)
    df = (mesh.Dx @ f.ravel(order="F")).reshape(mesh.X.shape, order="F")

    df_linf = float(np.max(np.abs(df - df_an)))
    df_rel_l2 = float(np.linalg.norm(df - df_an) / max(np.linalg.norm(df_an), 1e-30))

    # MATLAB exports these arrays transposed relative to our storage.
    return {
        "x": x.T,
        "y": y.T,
        "f": f.T,
        "df": df.T,
        "df_an": df_an.T,
        "df_linf": df_linf,
        "df_rel_l2": df_rel_l2,
        "overall_pass": bool(df_linf <= 1e-10 and df_rel_l2 <= 1e-10),
    }


def _write_artifact_report(fields: dict[str, np.ndarray | float | bool]) -> None:
    """Write the local Example 0 diagnostic report inside the repo."""
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "computed",
        "note": (
            "This artifact stores Python-versus-analytic derivative errors only. "
            "Use examples/martini/martini_parity_latest.json for MATLAB parity status."
        ),
        "df_linf": float(fields["df_linf"]),
        "df_rel_l2": float(fields["df_rel_l2"]),
        "analytical_self_check_pass": bool(fields["overall_pass"]),
    }

    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")


def compute(cfg: dict) -> None:
    fields = _compute_reference_fields(cfg)
    _write_artifact_report(fields)

    np.savez_compressed(
        OUTPUT_NPZ,
        x=np.asarray(fields["x"], dtype=float),
        y=np.asarray(fields["y"], dtype=float),
        f=np.asarray(fields["f"], dtype=float),
        df=np.asarray(fields["df"], dtype=float),
        df_an=np.asarray(fields["df_an"], dtype=float),
        df_linf=np.array([fields["df_linf"]], dtype=float),
        df_rel_l2=np.array([fields["df_rel_l2"]], dtype=float),
        overall_pass=np.array([fields["overall_pass"]], dtype=bool),
    )
    print(f"Saved fields: {OUTPUT_NPZ}")
    print(f"Saved artifact: {ARTIFACT_JSON}")


def plot(data: dict[str, np.ndarray], cfg: dict) -> None:
    mesh_path, validation_path, triptych_path = plot_example0_figures(
        data,
        example_dir=SCRIPT.parent,
        cfg=cfg,
    )
    print(f"Saved figure: {mesh_path}")
    print(f"Saved figure: {validation_path}")
    print(f"Saved figure: {triptych_path}")


def ensure_artifact(data: dict[str, np.ndarray]) -> None:
    """Ensure the local analytic-error artifact exists inside the repo."""
    if ARTIFACT_JSON.exists():
        return
    _write_artifact_report(
        {
            "df_linf": float(np.asarray(data["df_linf"]).reshape(-1)[0]),
            "df_rel_l2": float(np.asarray(data["df_rel_l2"]).reshape(-1)[0]),
            "overall_pass": bool(np.asarray(data["overall_pass"]).reshape(-1)[0]),
        }
    )


def main() -> None:
    cfg = load_config(SCRIPT.with_suffix(".yaml"))
    raw = load_or_compute(
        OUTPUT_NPZ,
        lambda: compute(cfg),
        required_keys={"x", "y", "f", "df", "df_an", "overall_pass"},
        allow_pickle=False,
        verbose=False,
    )
    data = {key: raw[key] for key in raw.files}
    ensure_artifact(data)
    plot(data, cfg)


if __name__ == "__main__":
    main()
