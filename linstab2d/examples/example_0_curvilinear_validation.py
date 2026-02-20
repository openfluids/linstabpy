#!/usr/bin/env python3
"""
Example 0: Curvilinear derivative validation.

Validates transformed derivative operators for deformation modes:
`none`, `x_only`, `y_only`, and `full_2d`.

OUTPUTS: example_0_curvilinear_validation.npz, example_0_curvilinear_validation.png
USAGE:   python example_0_curvilinear_validation.py
         rm example_0_curvilinear_validation.npz && python example_0_curvilinear_validation.py
"""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np

from linstab2d.mesh import Mesh

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
ARTIFACT_JSON = SCRIPT.parents[2] / "artifacts" / "curvilinear_derivative_parity.json"


def _analytic_field(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ax = 2.2
    by = 1.7
    f = np.sin(ax * X) * np.cos(by * Y)
    dfdx = ax * np.cos(ax * X) * np.cos(by * Y)
    dfdy = -by * np.sin(ax * X) * np.sin(by * Y)
    d2fdxdy = -ax * by * np.cos(ax * X) * np.sin(by * Y)
    return f, dfdx, dfdy, d2fdxdy


def _flatten_used(mesh: Mesh, q: np.ndarray) -> np.ndarray:
    return q.ravel(order="F")[mesh.used_ind]


def _normal_derivative_errors(mesh: Mesh, f_vec: np.ndarray, dfdx_e: np.ndarray, dfdy_e: np.ndarray) -> dict[str, float]:
    x_flat = mesh.X.ravel(order="F")[mesh.used_ind]
    y_flat = mesh.Y.ravel(order="F")[mesh.used_ind]

    gradX_x = mesh.Dx @ x_flat
    gradX_y = mesh.Dy @ x_flat
    gradY_x = mesh.Dx @ y_flat
    gradY_y = mesh.Dy @ y_flat

    def _err(ids: np.ndarray, nx: float, ny: float) -> float:
        if ids.size == 0:
            return 0.0
        num = (
            (gradX_x[ids] * nx + gradY_x[ids] * ny) * (mesh.Dx @ f_vec)[ids]
            + (gradX_y[ids] * nx + gradY_y[ids] * ny) * (mesh.Dy @ f_vec)[ids]
        )
        exact = nx * dfdx_e[ids] + ny * dfdy_e[ids]
        return float(np.max(np.abs(num - exact)))

    return {
        "left": _err(mesh.idx["li"], 1.0, 0.0),
        "right": _err(mesh.idx["ri"], 1.0, 0.0),
        "bottom": _err(mesh.idx["bi"], 0.0, 1.0),
        "top": _err(mesh.idx["ti"], 0.0, 1.0),
    }


def _branch_error_metrics(mesh: Mesh) -> dict[str, float]:
    f, dfdx, dfdy, d2fdxdy = _analytic_field(mesh.X, mesh.Y)
    f_v = _flatten_used(mesh, f)
    dfdx_e = _flatten_used(mesh, dfdx)
    dfdy_e = _flatten_used(mesh, dfdy)
    d2fdxdy_e = _flatten_used(mesh, d2fdxdy)

    dfdx_n = mesh.Dx @ f_v
    dfdy_n = mesh.Dy @ f_v
    d2xy_n = mesh.Dxy @ f_v
    d2yx_n = mesh.Dyx @ f_v

    def _linf(num: np.ndarray, ref: np.ndarray) -> float:
        return float(np.max(np.abs(num - ref)))

    def _l2rel(num: np.ndarray, ref: np.ndarray) -> float:
        return float(np.linalg.norm(num - ref) / max(np.linalg.norm(ref), 1e-30))

    normal_errors = _normal_derivative_errors(mesh, f_v, dfdx_e, dfdy_e)

    return {
        "dfdx_inf": _linf(dfdx_n, dfdx_e),
        "dfdy_inf": _linf(dfdy_n, dfdy_e),
        "d2xy_inf": _linf(d2xy_n, d2fdxdy_e),
        "d2yx_inf": _linf(d2yx_n, d2fdxdy_e),
        "dfdx_l2rel": _l2rel(dfdx_n, dfdx_e),
        "dfdy_l2rel": _l2rel(dfdy_n, dfdy_e),
        "d2xy_l2rel": _l2rel(d2xy_n, d2fdxdy_e),
        "d2yx_l2rel": _l2rel(d2yx_n, d2fdxdy_e),
        "normal_left_inf": normal_errors["left"],
        "normal_right_inf": normal_errors["right"],
        "normal_bottom_inf": normal_errors["bottom"],
        "normal_top_inf": normal_errors["top"],
    }


def _build_deformed_meshes() -> dict[str, Mesh]:
    mesh0 = Mesh.create(
        xrange=(0.0, 1.0),
        yrange=(0.0, 1.0),
        nx=84,
        ny=68,
        fd_order=4,
    )
    X = mesh0.X
    Y = mesh0.Y

    x_only = mesh0.deform(
        X_new=X + 0.08 * np.sin(2.0 * np.pi * X) * (0.3 + Y)
    )
    y_only = mesh0.deform(
        Y_new=Y + 0.06 * np.sin(2.0 * np.pi * Y) * (0.2 + X)
    )
    full_2d = mesh0.deform(
        X_new=X + 0.05 * np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y),
        Y_new=Y + 0.05 * np.cos(np.pi * X) * np.sin(2.0 * np.pi * Y),
    )
    return {
        "none": mesh0,
        "x_only": x_only,
        "y_only": y_only,
        "full_2d": full_2d,
    }


def _status_from_metrics(metrics: dict[str, float]) -> bool:
    # Conservative tolerances for curved mappings on a moderate grid.
    return (
        metrics["dfdx_inf"] <= 1.5e-1
        and metrics["dfdy_inf"] <= 1.5e-1
        and metrics["d2xy_inf"] <= 8.5e-1
        and metrics["d2yx_inf"] <= 8.5e-1
        and metrics["normal_left_inf"] <= 2.5e-1
        and metrics["normal_right_inf"] <= 2.5e-1
        and metrics["normal_bottom_inf"] <= 2.5e-1
        and metrics["normal_top_inf"] <= 2.5e-1
    )


def compute():
    meshes = _build_deformed_meshes()
    branch_names = ["none", "x_only", "y_only", "full_2d"]
    records = []

    for name in branch_names:
        metrics = _branch_error_metrics(meshes[name])
        passed = _status_from_metrics(metrics)
        records.append(
            {
                "branch": name,
                "pass": passed,
                "metrics": metrics,
            }
        )

    overall_pass = all(r["pass"] for r in records)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if overall_pass else "fail",
        "branches": records,
    }

    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    np.savez_compressed(
        OUTPUT_NPZ,
        branch_names=np.array(branch_names),
        dfdx_inf=np.array([r["metrics"]["dfdx_inf"] for r in records]),
        dfdy_inf=np.array([r["metrics"]["dfdy_inf"] for r in records]),
        d2xy_inf=np.array([r["metrics"]["d2xy_inf"] for r in records]),
        d2yx_inf=np.array([r["metrics"]["d2yx_inf"] for r in records]),
        normal_left_inf=np.array([r["metrics"]["normal_left_inf"] for r in records]),
        normal_right_inf=np.array([r["metrics"]["normal_right_inf"] for r in records]),
        normal_bottom_inf=np.array([r["metrics"]["normal_bottom_inf"] for r in records]),
        normal_top_inf=np.array([r["metrics"]["normal_top_inf"] for r in records]),
        pass_flags=np.array([r["pass"] for r in records], dtype=bool),
        overall_pass=np.array([overall_pass], dtype=bool),
    )
    print(f"Saved metrics: {OUTPUT_NPZ}")
    print(f"Saved artifact: {ARTIFACT_JSON}")


def plot(data):
    import matplotlib.pyplot as plt

    names = [str(s) for s in data["branch_names"]]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    series = [
        ("dfdx_inf", "dfdy_inf", "First derivatives (L_inf)"),
        ("d2xy_inf", "d2yx_inf", "Mixed derivatives (L_inf)"),
        ("normal_left_inf", "normal_right_inf", "Normal x-boundaries (L_inf)"),
        ("normal_bottom_inf", "normal_top_inf", "Normal y-boundaries (L_inf)"),
    ]

    for ax, (k1, k2, title) in zip(axes.ravel(), series):
        y1 = np.asarray(data[k1], dtype=float)
        y2 = np.asarray(data[k2], dtype=float)
        ax.semilogy(x, y1 + 1e-16, "o-", label=k1)
        ax.semilogy(x, y2 + 1e-16, "s-", label=k2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(loc="best")

    overall_pass = bool(np.asarray(data["overall_pass"]).ravel()[0])
    fig.suptitle(
        f"Curvilinear Derivative Validation ({'PASS' if overall_pass else 'FAIL'})"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {OUTPUT_PNG}")


def main():
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    except (FileNotFoundError, EOFError, OSError, ValueError):
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
