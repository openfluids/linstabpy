#!/usr/bin/env python3
"""
Example 0: Curvilinear derivative validation.

Translates `example_0_TestCurvilinearDerivative.m` directly.

- build the symmetric computational mesh.
- deform it to the parabolic physical domain.
- evaluate `f = cos(x)` and `df = Dx @ f`.

OUTPUTS (``examples/linstab2d/``).
  - example_0_curvilinear_validation.npz.
  - example_0_curvilinear_validation.log.

USAGE:   python example_0_curvilinear_validation.py
         rm example_0_curvilinear_validation.npz && python example_0_curvilinear_validation.py
         python example_0_plots.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from examples._helpers import load_or_compute
from linstabpy.mesh import Mesh
from linstabpy.utils.yaml_io import load_config

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
LOG_PATH = SCRIPT.with_suffix(".log")


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

    # The slim LinStab2DResults/ bundles keep MATLAB's (ny, nx) layout, the same
    # as mesh.X.
    # The earlier root-level export used the transpose, hence the old .T.
    return {
        "x": x,
        "y": y,
        "f": f,
        "df": df,
        "df_an": df_an,
        "df_linf": df_linf,
        "df_rel_l2": df_rel_l2,
        "overall_pass": bool(df_linf <= 1e-10 and df_rel_l2 <= 1e-10),
    }


def _log(line: str) -> None:
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def compute(cfg: dict) -> None:
    fields = _compute_reference_fields(cfg)

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


def main() -> None:
    cfg = load_config(SCRIPT.with_suffix(".yaml"))
    t0 = time.perf_counter()
    load_or_compute(
        OUTPUT_NPZ,
        lambda: compute(cfg),
        required_keys={"x", "y", "f", "df", "df_an", "overall_pass"},
        allow_pickle=False,
        verbose=False,
    )
    _log(f"example_0_curvilinear_validation  {time.perf_counter() - t0:.3f} s")


if __name__ == "__main__":
    main()
