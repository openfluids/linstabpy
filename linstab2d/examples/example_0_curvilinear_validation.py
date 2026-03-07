#!/usr/bin/env python3
"""
Example 0: Curvilinear derivative validation.

Translates `example_0_TestCurvilinearDerivative.m` directly:
- build the symmetric computational mesh
- deform it to the parabolic physical domain
- evaluate `f = cos(x)` and `df = Dx @ f`

OUTPUTS: example_0_curvilinear_validation.npz, example_0_curvilinear_validation.png
USAGE:   python example_0_curvilinear_validation.py
         rm example_0_curvilinear_validation.npz && python example_0_curvilinear_validation.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from linstab2d.mesh import Mesh
from linstab2d.utils.plotting import plot_flow

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
ARTIFACT_JSON = SCRIPT.parents[2] / "artifacts" / "curvilinear_derivative_parity.json"


def _build_reference_mesh() -> Mesh:
    """Build the Example 0 curvilinear mesh exactly as in MATLAB."""
    cmesh = Mesh.create(
        xrange=(-1.0, 0.0),
        yrange=(0.0, 1.0),
        nx=50,
        ny=50,
        fd_order=4,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )

    d = 0.2
    x = cmesh.X * d - 0.5
    z = x + 1j * cmesh.Y
    X = -np.real(z**2)
    Y = -np.imag(z**2)
    return cmesh.deform(X_new=X, Y_new=Y)


def _compute_reference_fields() -> dict[str, np.ndarray | float | bool]:
    """Compute the MATLAB-like Example 0 fields and export orientation."""
    mesh = _build_reference_mesh()
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


def compute() -> None:
    fields = _compute_reference_fields()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if bool(fields["overall_pass"]) else "fail",
        "df_linf": float(fields["df_linf"]),
        "df_rel_l2": float(fields["df_rel_l2"]),
    }

    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

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


def plot(data: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    f = np.asarray(data["f"], dtype=float)
    df = np.asarray(data["df"], dtype=float)
    df_an = np.asarray(data["df_an"], dtype=float)

    fig = plt.figure(figsize=(12, 8))
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    fields = [
        (f, r"$f$"),
        (df_an, r"$df_{analytical}$"),
        (df, r"$df_{numerical}$"),
        (df - df_an, r"$df_{error}$"),
    ]

    for ax, (field, title) in zip(axes, fields):
        plot_flow(x, y, field, ax=ax)
        ax.set_title(title)

    overall_pass = bool(np.asarray(data["overall_pass"]).reshape(-1)[0])
    fig.suptitle(f"Curvilinear Derivative Validation ({'PASS' if overall_pass else 'FAIL'})")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {OUTPUT_PNG}")


def main() -> None:
    try:
        raw = np.load(OUTPUT_NPZ, allow_pickle = False)
        required = {"x", "y", "f", "df", "df_an", "overall_pass"}
        if not required.issubset(set(raw.files)):
            raise KeyError("stale cache schema")
    except (FileNotFoundError, EOFError, OSError, ValueError):
        compute()
        raw = np.load(OUTPUT_NPZ, allow_pickle = False)
    except KeyError:
        compute()
        raw = np.load(OUTPUT_NPZ, allow_pickle = False)

    data = {key: raw[key] for key in raw.files}
    plot(data)


if __name__ == "__main__":
    main()
