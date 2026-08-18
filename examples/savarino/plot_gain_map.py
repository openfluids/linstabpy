#!/usr/bin/env python3
"""Figure S6 — resolvent gain map (Savarino2026 tensor/PCHIP style).

Reads a run-record CSV and renders :math:`\\sigma_1^2` with colorbar ticks in
units of :math:`10^{-4}` (physical :math:`\\sigma_1^2` = tick :math:`\\times 10^4`).
Interpolation is 1-D along :math:`\\log_{10} f` per :math:`\\beta` line, then
1-D across :math:`\\beta` — no scattered ``griddata`` / Delaunay fill.

Default input: ``output/savarino_map2_200pt.csv``.

Usage:
    python plot_gain_map.py
    python plot_gain_map.py --summary path/to/summary.csv --out path/to/map.png
    python plot_gain_map.py --show-points --baseline-summary output/savarino_map2_200pt.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from reference_paths import GAIN_MAP_CSV, GAIN_MAP_PNG  # noqa: E402

REF_COLORS = [
    "#311442",
    "#466de5",
    "#27bbea",
    "#31f197",
    "#a2fc3c",
    "#ebd039",
    "#fa8021",
    "#d33204",
    "#7d0401",
]

LIFT_UP_PEAK = {
    "f": 1.0e-4,
    "beta": 163.0,
    "label": "Low-frequency\nlift-up mode",
    "marker": "D",
    "xytext": (42, 0),
    "ha": "left",
    "color": "white",
}

GLOBAL_PEAK_LABEL = "Medium-frequency\n1st Mack mode"
GLOBAL_PEAK_XYTEXT = (-42, 0)
DEFAULT_MAP_TITLE = r"$\theta = 30.8^\circ$"

_MAP_STYLE_APPLIED = False
MAP_FONT = {
    "title": 14,
    "label": 14,
    "tick": 12,
    "cbar_label": 15,
    "cbar_tick": 11,
    "annotate": 13,
    "legend": 11,
}


def _latex_available() -> bool:
    fig = None
    previous = matplotlib.rcParams.get("text.usetex", False)
    try:
        matplotlib.rcParams["text.usetex"] = True
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$x$")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.tell() > 0
    except Exception:
        return False
    finally:
        matplotlib.rcParams["text.usetex"] = previous
        if fig is not None:
            plt.close(fig)


def _apply_map_style() -> None:
    global _MAP_STYLE_APPLIED
    if _MAP_STYLE_APPLIED:
        return
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
            "text.usetex": _latex_available(),
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "axes.titlesize": MAP_FONT["title"],
            "axes.labelsize": MAP_FONT["label"],
            "xtick.labelsize": MAP_FONT["tick"],
            "ytick.labelsize": MAP_FONT["tick"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    _MAP_STYLE_APPLIED = True


def _colorbar_ticks(vmax: float) -> list[float]:
    if vmax <= 0.0:
        return [0.0]
    if vmax <= 5.0:
        return [0.0, 1.0, 2.0, 3.0]
    raw = MaxNLocator(nbins=5, min_n_ticks=4).tick_values(0.0, vmax)
    ticks = sorted({0.0, *(float(t) for t in raw if 0.0 <= float(t) <= vmax)})
    if len(ticks) < 2:
        ticks = [0.0, vmax]
    return ticks


def _load_rows(summary_path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"empty summary: {summary_path}")
    return rows


def _row_gain(row: dict[str, str]) -> float:
    if row.get("gain"):
        return float(row["gain"])
    if row.get("sigma1"):
        return float(row["sigma1"])
    if row.get("sigma1_sq"):
        return float(np.sqrt(float(row["sigma1_sq"])))
    raise KeyError(f"no gain column in row keys {list(row)}")


def _row_f_beta(row: dict[str, str]) -> tuple[float, float]:
    for fk in ("f_Xsh_Uinf", "f_XSh_over_Uinf", "f"):
        if row.get(fk) not in (None, ""):
            f = float(row[fk])
            break
    else:
        raise KeyError(f"no frequency column in {list(row)}")
    for bk in ("beta_Xsh", "beta_XSh", "beta"):
        if row.get(bk) not in (None, ""):
            beta = float(row[bk])
            break
    else:
        raise KeyError(f"no beta column in {list(row)}")
    return f, beta


def tensor_grid(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, dict]:
    pts: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("status") not in (None, "", "ok", "done"):
            continue
        f, beta = _row_f_beta(row)
        gain = _row_gain(row)
        if f > 0.0:
            pts.setdefault(beta, []).append((np.log10(f), gain * gain * 1.0e-4))

    gx = np.linspace(-4.0, 1.0, 501)
    betas = sorted(pts)
    lines: dict[float, np.ndarray] = {}
    for beta in betas:
        samples = sorted(pts[beta])
        xs = np.array([p[0] for p in samples], dtype=float)
        ys = np.array([p[1] for p in samples], dtype=float)
        if xs.size == 1:
            col = np.full(gx.size, np.nan)
            col[np.argmin(np.abs(gx - xs[0]))] = ys[0]
        elif xs.size < 4:
            col = np.interp(gx, xs, ys)
            col[(gx < xs.min()) | (gx > xs.max())] = np.nan
        else:
            col = PchipInterpolator(xs, ys)(gx)
            col[(gx < xs.min()) | (gx > xs.max())] = np.nan
        lines[beta] = col

    gy = np.linspace(0.0, 200.0, 501)
    gz = np.full((gy.size, gx.size), np.nan)
    for j in range(gx.size):
        beta_vals = np.array([b for b in betas if np.isfinite(lines[b][j])])
        if beta_vals.size < 2:
            continue
        vals = np.array([lines[b][j] for b in beta_vals])
        if beta_vals.size < 4:
            col = np.interp(gy, beta_vals, vals)
        else:
            col = PchipInterpolator(beta_vals, vals)(gy)
        col[(gy < beta_vals.min()) | (gy > beta_vals.max())] = np.nan
        gz[:, j] = col

    return gx, gy, np.ma.masked_invalid(gz), pts


def _log_f_axis(f: float) -> float:
    if f <= 0.0:
        return -4.0
    return float(np.log10(f))


def _overlay_points(
    ax,
    rows: list[dict[str, str]],
    *,
    baseline_keys: set[tuple[float, float]] | None = None,
) -> int:
    n = 0
    for row in rows:
        if row.get("status") not in (None, "", "ok", "done"):
            continue
        f, beta = _row_f_beta(row)
        key = (beta, f)
        x = _log_f_axis(f)
        if baseline_keys is None:
            ax.plot(
                x,
                beta,
                "o",
                ms=3.2,
                mfc="white",
                mec="black",
                mew=0.45,
                alpha=0.9,
                zorder=7,
            )
        elif key in baseline_keys:
            ax.plot(
                x,
                beta,
                "o",
                ms=3.0,
                mfc="white",
                mec="black",
                mew=0.45,
                alpha=0.95,
                zorder=7,
            )
        else:
            ax.plot(
                x,
                beta,
                "o",
                ms=2.8,
                mfc="none",
                mec="0.15",
                mew=0.55,
                alpha=0.85,
                zorder=7,
            )
        n += 1
    return n


def plot_map(
    rows: list[dict[str, str]],
    out: Path,
    *,
    title: str = DEFAULT_MAP_TITLE,
    annotate_peaks: bool = True,
    show_points: bool = False,
    baseline_keys: set[tuple[float, float]] | None = None,
) -> dict[str, float | int]:
    _apply_map_style()
    gx, gy, gz, pts = tensor_grid(rows)
    if not np.any(np.isfinite(gz)):
        raise SystemExit("tensor grid is empty after interpolation")

    vmax = float(np.nanmax(gz))
    fig = plt.figure(figsize=(7.6, 5.4))
    ax = fig.add_axes([0.11, 0.14, 0.66, 0.78])
    cax = fig.add_axes([0.80, 0.14, 0.04, 0.78])

    levels = np.linspace(0.0, vmax, 10)
    cf = ax.contourf(gx, gy, gz, levels=levels, colors=REF_COLORS)
    for coll in cf.collections:
        coll.set_edgecolor("face")

    ax.set_xlim(-4.0, 1.0)
    ax.set_ylim(0.0, 200.0)
    ax.set_xticks(range(-4, 2))
    ax.set_xticklabels([rf"$10^{{{t}}}$" for t in range(-4, 2)])
    ax.set_yticks([0, 50, 100, 150, 200])
    minor = [np.log10(k * 10.0**e) for e in range(-4, 1) for k in range(2, 10)]
    ax.set_xticks(minor, minor=True)
    ax.tick_params(
        which="both",
        direction="out",
        top=True,
        right=True,
        labelsize=MAP_FONT["tick"],
    )
    ax.tick_params(which="major", length=5, pad=4)
    ax.tick_params(which="minor", length=2.5, pad=2)
    ax.set_xlabel(r"$f X_{\mathrm{sh}}/U_\infty$", fontsize=MAP_FONT["label"])
    ax.set_ylabel(r"$\beta X_{\mathrm{sh}}$", fontsize=MAP_FONT["label"])
    ax.set_title(title, pad=8, fontsize=MAP_FONT["title"])

    smooth = LinearSegmentedColormap.from_list("ref", REF_COLORS)
    sm = plt.cm.ScalarMappable(cmap=smooth, norm=plt.Normalize(0.0, vmax))
    cbar_ticks = _colorbar_ticks(vmax)
    cb = fig.colorbar(sm, cax=cax, ticks=cbar_ticks)
    cb.ax.tick_params(direction="out", labelsize=MAP_FONT["cbar_tick"], pad=3)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    cb.set_label(r"$\sigma_1^2$", fontsize=MAP_FONT["cbar_label"], labelpad=8)
    cb.ax.set_title(r"$\times 10^{4}$", pad=6, fontsize=MAP_FONT["cbar_tick"])

    peak: dict[str, float | int] = {"n_points": len(rows), "vmax_scaled": vmax}
    if show_points:
        peak["n_points_drawn"] = _overlay_points(ax, rows, baseline_keys=baseline_keys)
        if baseline_keys is not None:
            ax.plot([], [], "o", ms=4, mfc="white", mec="black", label="author 200")
            ax.plot([], [], "o", ms=4, mfc="none", mec="0.15", label="ABC densify")
            ax.legend(loc="lower right", fontsize=MAP_FONT["legend"], framealpha=0.85)
    if annotate_peaks and not show_points:
        all_vals = [(v, b, x) for b in pts for x, v in sorted(pts[b])]
        _, beta_max, logf_max = max(all_vals)
        peak["global_logf"] = float(logf_max)
        peak["global_beta"] = float(beta_max)
        peak["global_f"] = float(10.0**logf_max)
        ax.plot(
            logf_max,
            beta_max,
            "s",
            ms=9,
            mfc="red",
            mec="black",
            mew=0.8,
            zorder=6,
        )
        ax.annotate(
            GLOBAL_PEAK_LABEL,
            xy=(logf_max, beta_max),
            xytext=GLOBAL_PEAK_XYTEXT,
            textcoords="offset points",
            fontsize=MAP_FONT["annotate"],
            va="center",
            ha="right",
            color="white",
            arrowprops=dict(arrowstyle="-", color="white", lw=0.8),
        )

        fp = float(LIFT_UP_PEAK["f"])
        bp = float(LIFT_UP_PEAK["beta"])
        ax.plot(
            np.log10(fp),
            bp,
            marker=str(LIFT_UP_PEAK["marker"]),
            ms=7,
            mfc="red",
            mec="k",
            mew=0.6,
            zorder=6,
        )
        ax.annotate(
            str(LIFT_UP_PEAK["label"]),
            xy=(np.log10(fp), bp),
            xytext=tuple(LIFT_UP_PEAK["xytext"]),
            textcoords="offset points",
            fontsize=MAP_FONT["annotate"],
            va="center",
            ha=str(LIFT_UP_PEAK["ha"]),
            color=str(LIFT_UP_PEAK["color"]),
            arrowprops=dict(arrowstyle="-", color="white", lw=0.8),
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return peak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=GAIN_MAP_CSV, help="run-record CSV")
    ap.add_argument("--out", type=Path, default=GAIN_MAP_PNG, help="output PNG")
    ap.add_argument("--title", type=str, default=DEFAULT_MAP_TITLE)
    ap.add_argument(
        "--show-points",
        action="store_true",
        help="overlay sample locations (filled=author grid, open=extra when --baseline-summary set)",
    )
    ap.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="reference summary for distinguishing author vs densified points",
    )
    ap.add_argument(
        "--no-annotate",
        action="store_true",
        help="skip Mack / lift-up peak markers",
    )
    args = ap.parse_args()

    summary = args.summary.expanduser().resolve()
    out = args.out.expanduser().resolve()
    rows = _load_rows(summary)
    baseline_keys = None
    if args.baseline_summary is not None:
        baseline_keys = {
            _row_f_beta(r)[::-1]
            for r in _load_rows(args.baseline_summary.expanduser().resolve())
            if r.get("status") in (None, "", "ok", "done")
        }
    peak = plot_map(
        rows,
        out,
        title=args.title,
        show_points=args.show_points,
        baseline_keys=baseline_keys,
        annotate_peaks=not args.no_annotate,
    )
    extra = ""
    if "n_points_drawn" in peak:
        extra = f", points_drawn={peak['n_points_drawn']}"
    print(f"wrote {out} (n={peak['n_points']}, vmax×1e4={peak['vmax_scaled']:.3g}{extra})")
    if "global_f" in peak:
        print(
            f"  global peak f={peak['global_f']:g}, beta={peak['global_beta']:g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
