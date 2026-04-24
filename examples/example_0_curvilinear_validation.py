#!/usr/bin/env python3
"""
Example 0: Curvilinear derivative validation.

Translates `example_0_TestCurvilinearDerivative.m` directly:
- build the symmetric computational mesh
- deform it to the parabolic physical domain
- evaluate `f = cos(x)` and `df = Dx @ f`

OUTPUTS:
  - example_0_curvilinear_validation.npz
  - example_0_curvilinear_validation.png
  - example_0_curvilinear_validation_baseflow.png
  - example_0_curvilinear_validation_support_error.png
USAGE:   python example_0_curvilinear_validation.py
         rm example_0_curvilinear_validation.npz && python example_0_curvilinear_validation.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from linstab2d.mesh import Mesh
from linstab2d.utils.jsonc import load_jsonc

from ._helpers import (
    apply_figure_layout,
    format_scientific_ticks,
    label_axes,
    load_or_compute,
    plot_scalar_panel,
    plot_signed_panel,
    save_figure,
    set_suptitle,
)

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
OUTPUT_BASEFLOW = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_baseflow.png")
OUTPUT_SUPPORT = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_support_error.png")
ARTIFACT_JSON = SCRIPT.parents[1] / "artifacts" / "curvilinear_derivative_parity.json"


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
            "Use validation/martini2024/reports/martini_parity_latest.json for MATLAB parity status."
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


def plot(data: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    f = np.asarray(data["f"], dtype=float)
    df = np.asarray(data["df"], dtype=float)
    df_an = np.asarray(data["df_an"], dtype=float)
    df_linf = float(np.asarray(data["df_linf"]).reshape(-1)[0])
    df_rel_l2 = float(np.asarray(data["df_rel_l2"]).reshape(-1)[0])
    error = df - df_an

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 8.4),
        gridspec_kw={"wspace": 0.34, "hspace": 0.24},
    )
    plot_scalar_panel(
        axes[0, 0],
        x,
        y,
        f,
        title=r"$f(x,y)=\cos(x)$",
        cmap="viridis",
        aspect="equal",
        title_pad=4.0,
    )
    plot_signed_panel(
        axes[0, 1],
        x,
        y,
        df_an,
        title=r"$\partial_x f_{\mathrm{analytical}}$",
        aspect="equal",
        title_pad=4.0,
    )
    plot_signed_panel(
        axes[1, 0],
        x,
        y,
        df,
        title=r"$\partial_x f_{\mathrm{numerical}}$",
        aspect="equal",
        title_pad=4.0,
    )
    plot_signed_panel(
        axes[1, 1],
        x,
        y,
        error,
        title=r"$\partial_x f_{\mathrm{num}}-\partial_x f_{\mathrm{an}}$",
        aspect="equal",
        title_pad=4.0,
    )
    for ax in axes.flat:
        label_axes(ax)
    set_suptitle(fig, "Curvilinear derivative verification", y=0.962)
    fig.text(
        0.5,
        0.028,
        rf"$\|\partial_x f_{{\mathrm{{num}}}}-\partial_x f_{{\mathrm{{an}}}}\|_\infty = {df_linf:.3e}$"
        "\n"
        rf"$\|\partial_x f_{{\mathrm{{num}}}}-\partial_x f_{{\mathrm{{an}}}}\|_2 / \|\partial_x f_{{\mathrm{{an}}}}\|_2 = {df_rel_l2:.3e}$",
        ha="center",
        va="bottom",
        fontsize=14,
        linespacing=1.35,
    )
    apply_figure_layout(fig, rect=(0.0, 0.08, 1.0, 0.93), pad=0.8)
    save_figure(fig, OUTPUT_PNG)

    cfg = load_jsonc(SCRIPT.with_suffix(".jsonc"))
    cmesh = _build_computational_mesh(cfg)
    fig_mesh, axes_mesh = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        gridspec_kw={"wspace": 0.08},
    )
    step = 5
    mesh_specs = [
        (axes_mesh[0], cmesh.X, cmesh.Y, "Computational mesh", step),
        (axes_mesh[1], x.T, y.T, "Mapped physical mesh", 1),
    ]
    for ax, Xg, Yg, title, local_step in mesh_specs:
        ax.plot(Xg[:, ::local_step], Yg[:, ::local_step], color="#4a5878", lw=0.7, alpha=0.9)
        ax.plot(Xg[::local_step, :].T, Yg[::local_step, :].T, color="#4a5878", lw=0.7, alpha=0.9)
        ax.plot(Xg[:, 0], Yg[:, 0], color="#394264", lw=1.0)
        ax.plot(Xg[:, -1], Yg[:, -1], color="#394264", lw=1.0)
        ax.plot(Xg[0, :], Yg[0, :], color="#394264", lw=1.0)
        ax.plot(Xg[-1, :], Yg[-1, :], color="#394264", lw=1.0)
        ax.set_title(title, pad=4.0)
        ax.set_aspect("equal", adjustable="box")
        label_axes(ax)
    apply_figure_layout(fig_mesh, pad=0.6)
    save_figure(fig_mesh, OUTPUT_BASEFLOW)

    centerline = y.shape[0] // 2
    fig_support, axes_support = plt.subplots(1, 2, figsize=(12, 4.5))
    axes_support[0].plot(x[centerline, :], df_an[centerline, :], label="Analytical", lw=2.0)
    axes_support[0].plot(x[centerline, :], df[centerline, :], "--", label="Numerical", lw=1.8)
    axes_support[0].set_title("Centerline derivative comparison", pad=4.0)
    label_axes(axes_support[0], xlabel=r"$x$", ylabel=r"$\partial_x f$")
    format_scientific_ticks(axes_support[0], axes=("y",))
    axes_support[0].grid(True, alpha=0.25)
    axes_support[0].legend(loc="best", frameon=False)

    abs_error = np.abs(error[np.isfinite(error)])
    axes_support[1].hist(abs_error, bins=30, color="#7a86b8", edgecolor="#394264")
    axes_support[1].set_title("Absolute error distribution", pad=4.0)
    axes_support[1].set_xlabel(r"$\left|\partial_x f_{\mathrm{num}}-\partial_x f_{\mathrm{an}}\right|$")
    axes_support[1].set_ylabel("Count")
    axes_support[1].set_yscale("log")
    format_scientific_ticks(axes_support[1], axes=("x",))
    axes_support[1].grid(True, alpha=0.25)
    apply_figure_layout(fig_support, pad=0.7)
    save_figure(fig_support, OUTPUT_SUPPORT)


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
    cfg = load_jsonc(SCRIPT.with_suffix(".jsonc"))
    raw = load_or_compute(
        OUTPUT_NPZ,
        lambda: compute(cfg),
        required_keys={"x", "y", "f", "df", "df_an", "overall_pass"},
        allow_pickle=False,
        verbose=False,
    )
    data = {key: raw[key] for key in raw.files}
    ensure_artifact(data)
    plot(data)


if __name__ == "__main__":
    main()
